"""5-Fold Out-Of-Fold (OOF) cross-validation and official whitespace-tokenized METEOR evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever
from src.common.normalize import clean_legal_text, normalize_question
from src.common.reranker import BGEReranker
from src.task2.article_stitcher import ArticleStitcher
from src.task2.generator import QwenGenerator
from src.task2.predict import LegalQAPipeline
from src.task2.qa_memory import QAMemory
from src.task2.source_snap import (
    generate_candidate_ensemble,
    select_best_answer_candidate,
    snap_facts_to_evidence,
)

try:
    nltk.data.find("corpora/wordnet.zip")
except LookupError:
    try:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception:
        pass


def calculate_official_meteor(references: List[str], predictions: List[str]) -> float:
    """Compute official whitespace-tokenized METEOR score matching competition scoring."""
    scores = []
    for r, p in zip(references, predictions):
        r_tokens = str(r).split()
        p_tokens = str(p).split()
        scores.append(meteor_score([r_tokens], p_tokens))
    return float(np.mean(scores)) if scores else 0.0


def calculate_rouge_l(references: List[str], predictions: List[str]) -> float:
    """Compute ROUGE-L f-measure without stemming."""
    if rouge_scorer is None or not references:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(str(r), str(p))["rougeL"].fmeasure for r, p in zip(references, predictions)]
    return float(np.mean(scores))


def run_oof_validation(
    qa_path: str = "artifacts/task2/data/qa_unique.parquet",
    fold_path: str = "artifacts/task2/data/fold_assignments.parquet",
    chunks_path: str = "artifacts/task2/data/legal_chunks.parquet",
    bm25_dir: str = "artifacts/task2/indexes/bm25",
    dek21_dir: str = "artifacts/task2/indexes/dek21",
    eval_output_dir: str = "artifacts/task2/evaluations",
    num_eval_samples: Optional[int] = 100,
    n_splits: int = 5,
    mode: str = "fast",  # "fast" or "full"
    model_path: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: Optional[str] = None,
    reranker_checkpoint: str = "BAAI/bge-reranker-v2-m3",
    held_out_fold: Optional[int] = None,
    device: Optional[str] = None,
    gen_device: Optional[str] = None,
    retrieval_device: Optional[str] = None,
    max_new_tokens: int = 384,
) -> Dict[str, Any]:
    print(f"=== Starting LegalQA Task 2 OOF Validation (Mode: {mode.upper()}) ===")
    if mode == "fast":
        print("*******************************************************************************")
        print("  DIAGNOSTIC ONLY — NOT VALID FOR MODEL QUALITY, CHECKPOINT VERIFICATION OR PROMOTION")
        print("*******************************************************************************")
    print(f"Loading QA dataset from {qa_path}...")
    df_qa = pd.read_parquet(qa_path)

    if os.path.exists(fold_path):
        df_folds = pd.read_parquet(fold_path)
        if "fold_id" in df_folds.columns:
            fold_map = dict(zip(df_folds["qa_id"], df_folds["fold_id"]))
            df_qa["fold_id"] = df_qa["qa_id"].map(fold_map).fillna(0).astype(int)

    if "fold_id" not in df_qa.columns:
        def _hash_q(q: str) -> int:
            norm = normalize_question(q)
            return int(hashlib.md5(f"42_{norm}".encode("utf-8")).hexdigest(), 16) % n_splits
        df_qa["fold_id"] = df_qa["question_norm"].apply(_hash_q)

    # 1. Sparse BM25
    print(f"Loading BM25 index from {bm25_dir}...")
    bm25 = BM25Retriever.load(bm25_dir, corpus_path=chunks_path) if os.path.exists(bm25_dir) else BM25Retriever()
    if not bm25.corpus and os.path.exists(chunks_path):
        df_c = pd.read_parquet(chunks_path)
        bm25.fit(df_c.to_dict("records"))

    # 2. Dense DEk21
    r_dev = retrieval_device or device
    if mode == "full" and os.path.exists(dek21_dir) and os.path.exists(os.path.join(dek21_dir, "embeddings.npy")):
        print(f"Loading real DEk21 embeddings on {r_dev}...")
        dense = DEk21Retriever.load_index(dek21_dir, corpus_path=chunks_path, device=r_dev)
    else:
        print("Using mock DEk21 dense retriever for fast validation...")
        dense = DEk21Retriever(model_name="mock", device=r_dev)
        if bm25.corpus:
            dense.fit_mock(bm25.corpus)

    # 3. Reranker
    if mode == "full" and device != "cpu":
        print(f"Loading Neural Cross-Encoder Reranker ({reranker_checkpoint}) on {r_dev}...")
        reranker = BGEReranker(model_name=reranker_checkpoint, device=r_dev)
    else:
        print("Using fast lexical reranker for validation...")
        reranker = BGEReranker(model_name="mock", device="cpu")

    # 4. Article Stitcher
    stitcher = ArticleStitcher(bm25.corpus if bm25.corpus else [])

    # 5. Generator
    g_dev = gen_device or device
    if mode == "full" and g_dev != "cpu":
        print(f"Loading Qwen2.5-3B Generator on {g_dev}...")
        generator = QwenGenerator.load(
            model_path=model_path,
            adapter_path=adapter_path,
            device=g_dev,
            runtime="torch",
        )
    else:
        print("Using extractive fallback generator for fast validation...")
        generator = QwenGenerator(runtime="fallback")

    all_records = df_qa.to_dict("records")
    full_memory = QAMemory.from_records(all_records)

    all_oof_results = []
    fold_meteors = []
    fold_rouges = []

    candidate_families = [
        "focused_extract",
        "stitched_extract",
        "generated",
        "snapped",
        "strategy_f_300",
        "strategy_f_600",
        "strategy_f_1000",
        "strategy_f_1500",
        "selected",
        "oracle_best",
    ]
    family_scores: Dict[str, List[float]] = {f: [] for f in candidate_families}

    samples_per_fold = (num_eval_samples // n_splits) if num_eval_samples else None
    print(f"\nEvaluating {'all' if not samples_per_fold else samples_per_fold} samples per fold across {n_splits} folds...")

    for fold_id in range(n_splits):
        fold_records = df_qa[df_qa["fold_id"] == fold_id]
        if samples_per_fold:
            val_subset = fold_records.head(samples_per_fold)
        else:
            val_subset = fold_records

        # Strict zero-leakage fold memory: exclude ALL records assigned to this validation fold
        all_fold_qa_ids = set(fold_records["qa_id"].astype(str))
        all_fold_questions = set(fold_records["question_raw"].astype(str))
        isolated_mem = full_memory.filter_fold(val_qa_ids=all_fold_qa_ids, val_questions=all_fold_questions)
        pipeline = LegalQAPipeline(isolated_mem, bm25, dense, reranker, stitcher, generator)

        fold_preds = []
        fold_refs = []

        for _, row in tqdm(val_subset.iterrows(), total=len(val_subset), desc=f"Fold {fold_id}"):
            qid = str(row["qa_id"])
            q = str(row["question_raw"])
            ref_ans = str(row["answer_raw"])
            ref_tokens = ref_ans.split()

            selected, cands, ev = pipeline.predict_single(
                qid, q, max_new_tokens=max_new_tokens, return_candidates=True
            )
            cands["selected"] = selected

            # Score each candidate against reference
            cand_meteors = {}
            best_cand_score = 0.0
            best_cand_name = "selected"

            for name, cand_text in cands.items():
                sc = meteor_score([ref_tokens], str(cand_text).split())
                cand_meteors[name] = sc
                if name in family_scores:
                    family_scores[name].append(sc)
                if sc > best_cand_score:
                    best_cand_score = sc
                    best_cand_name = name

            family_scores["oracle_best"].append(best_cand_score)

            fold_preds.append(selected)
            fold_refs.append(ref_ans)

            sc_selected = cand_meteors.get("selected", 0.0)
            all_oof_results.append({
                "qa_id": qid,
                "fold_id": fold_id,
                "question": q,
                "reference": ref_ans,
                "prediction": selected,
                "meteor": sc_selected,
                "oracle_best_candidate": best_cand_name,
                "oracle_best_meteor": best_cand_score,
                "candidate_scores": cand_meteors,
            })

        fold_m = calculate_official_meteor(fold_refs, fold_preds)
        fold_r = calculate_rouge_l(fold_refs, fold_preds)
        fold_meteors.append(fold_m)
        fold_rouges.append(fold_r)
        print(f"Fold {fold_id} -> METEOR: {fold_m:.4f} | ROUGE-L: {fold_r:.4f}")

    mean_meteor = float(np.mean(fold_meteors))
    std_meteor = float(np.std(fold_meteors))
    mean_rouge = float(np.mean(fold_rouges))

    print("\n=======================================================")
    print(f"5-Fold OOF METEOR:  {mean_meteor:.4f} ± {std_meteor:.4f}")
    print(f"5-Fold OOF ROUGE-L: {mean_rouge:.4f}")
    print("-------------------------------------------------------")
    print("Candidate Family Breakdown (Mean METEOR across all OOF):")
    cand_summary = {}
    for f_name in candidate_families:
        scores = family_scores.get(f_name, [])
        avg = float(np.mean(scores)) if scores else 0.0
        cand_summary[f_name] = round(avg, 4)
        print(f" - {f_name:18s}: {avg:.4f}")
    print("=======================================================")

    os.makedirs(eval_output_dir, exist_ok=True)
    df_oof = pd.DataFrame(all_oof_results)
    oof_out = os.path.join(eval_output_dir, "oof_predictions.parquet")
    df_oof.to_parquet(oof_out, index=False)

    summary = {
        "mode": mode,
        "num_folds": n_splits,
        "total_evaluated": len(df_oof),
        "mean_meteor": mean_meteor,
        "std_meteor": std_meteor,
        "mean_rouge_l": mean_rouge,
        "fold_meteors": fold_meteors,
        "fold_rouges": fold_rouges,
        "candidate_family_meteors": cand_summary,
    }
    with open(os.path.join(eval_output_dir, "oof_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved OOF predictions to {oof_out} and summary to {eval_output_dir}/oof_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description="LegalQA Task 2 5-Fold OOF Validation")
    parser.add_argument("--qa_path", default="artifacts/task2/data/qa_unique.parquet")
    parser.add_argument("--fold_path", default="artifacts/task2/data/fold_assignments.parquet")
    parser.add_argument("--chunks_path", default="artifacts/task2/data/legal_chunks.parquet")
    parser.add_argument("--bm25_dir", default="artifacts/task2/indexes/bm25")
    parser.add_argument("--dek21_dir", default="artifacts/task2/indexes/dek21")
    parser.add_argument("--eval_output_dir", default="artifacts/task2/evaluations")
    parser.add_argument("--samples", type=int, default=50, help="Total sample count across folds (or 0 for all)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mode", default="fast", choices=["fast", "full"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--reranker_checkpoint", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--held_out_fold", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    args = parser.parse_args()

    num_samples = args.samples if args.samples > 0 else None
    run_oof_validation(
        qa_path=args.qa_path,
        fold_path=args.fold_path,
        chunks_path=args.chunks_path,
        bm25_dir=args.bm25_dir,
        dek21_dir=args.dek21_dir,
        eval_output_dir=args.eval_output_dir,
        num_eval_samples=num_samples,
        n_splits=args.folds,
        mode=args.mode,
        model_path=args.model,
        adapter_path=args.adapter,
        reranker_checkpoint=args.reranker_checkpoint,
        held_out_fold=args.held_out_fold,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
