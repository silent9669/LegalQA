"""5-Fold Out-Of-Fold (OOF) cross-validation and official whitespace-tokenized METEOR evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    adapter_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
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

    print(f"Loading retrieval indexes from {bm25_dir} and {chunks_path}...")
    bm25 = BM25Retriever.load(bm25_dir, corpus_path=chunks_path) if os.path.exists(bm25_dir) else BM25Retriever()

    dense = None
    if os.path.exists(dek21_dir) and os.path.exists(os.path.join(dek21_dir, "embeddings.npy")):
        dense = DEk21Retriever.load_index(dek21_dir, corpus_path=chunks_path, device=device)
    else:
        dense = DEk21Retriever(model_name="mock", device=device)
        if bm25.corpus:
            dense.fit_mock(bm25.corpus)

    reranker = BGEReranker(model_name="mock" if device == "cpu" else "BAAI/bge-reranker-v2-m3", device=device)
    stitcher = ArticleStitcher(bm25.corpus if bm25.corpus else [])
    generator = QwenGenerator(runtime="fallback")

    all_oof_results = []
    fold_meteors = []
    fold_rouges = []

    samples_per_fold = (num_eval_samples // n_splits) if num_eval_samples else None
    print(f"Evaluating {'all' if not samples_per_fold else samples_per_fold} samples per fold across {n_splits} folds...")

    for fold_id in range(n_splits):
        fold_records = df_qa[df_qa["fold_id"] == fold_id]
        if samples_per_fold:
            val_subset = fold_records.head(samples_per_fold)
        else:
            val_subset = fold_records

        train_subset = df_qa[df_qa["fold_id"] != fold_id]

        # Zero-leakage fold memory: val queries cannot lookup themselves
        fold_mem = QAMemory.from_records(train_subset.to_dict("records"))
        pipeline = LegalQAPipeline(fold_mem, bm25, dense, reranker, stitcher, generator)

        fold_preds = []
        fold_refs = []

        for _, row in tqdm(val_subset.iterrows(), total=len(val_subset), desc=f"Fold {fold_id}"):
            qid = str(row["qa_id"])
            q = str(row["question_raw"])
            ref_ans = str(row["answer_raw"])

            pred = pipeline.predict_single(qid, q)
            fold_preds.append(pred)
            fold_refs.append(ref_ans)

            sc_meteor = meteor_score([ref_ans.split()], pred.split())
            all_oof_results.append({
                "qa_id": qid,
                "fold_id": fold_id,
                "question": q,
                "reference": ref_ans,
                "prediction": pred,
                "meteor": sc_meteor,
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
    print("=======================================================")

    os.makedirs(eval_output_dir, exist_ok=True)
    df_oof = pd.DataFrame(all_oof_results)
    oof_out = os.path.join(eval_output_dir, "oof_predictions.parquet")
    df_oof.to_parquet(oof_out, index=False)

    summary = {
        "num_folds": n_splits,
        "total_evaluated": len(df_oof),
        "mean_meteor": mean_meteor,
        "std_meteor": std_meteor,
        "mean_rouge_l": mean_rouge,
        "fold_meteors": fold_meteors,
        "fold_rouges": fold_rouges,
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
    parser.add_argument("--samples", type=int, default=100, help="Total sample count across folds (or 0 for all)")
    parser.add_argument("--folds", type=int, default=5)
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
    )


if __name__ == "__main__":
    main()
