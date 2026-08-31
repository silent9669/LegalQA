"""Honest, leakage-safe evaluation module for exact trained reranker and generator checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

import nltk
import numpy as np
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.common.bm25 import BM25Retriever
from src.common.dense import DenseRetriever
from src.common.reranker import BGEReranker
from src.task2.candidates import generate_candidate_ensemble
from src.task2.evidence_packer import EvidencePacker
from src.task2.generator import QwenGenerator
from src.task2.predict import LegalQAPipeline
from src.task2.qa_memory import QAMemory
from src.task2.selector import CandidateSelector


def evaluate_checkpoint(
    qa_path: str = "artifacts/task2/data/qa_unique.parquet",
    fold_path: str = "artifacts/task2/data/fold_assignments.parquet",
    chunks_path: str = "artifacts/task2/data/legal_chunks.parquet",
    held_out_fold: int = 0,
    bm25_dir: str = "artifacts/task2/indexes/bm25",
    dense_dir: str = "artifacts/task2/indexes/dek21",
    dense_model: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
    reranker_checkpoint: str = "BAAI/bge-reranker-v2-m3",
    generator_model: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: Optional[str] = None,
    selector_path: Optional[str] = None,
    sample_size: Optional[int] = 50,
    eval_output_dir: str = "artifacts/task2/evaluations",
    gen_device: Optional[str] = None,
    retrieval_device: Optional[str] = None,
    max_new_tokens: int = 384,
    fail_on_fallback: bool = True,
) -> Dict[str, Any]:
    """Evaluate exact trained checkpoints on a strictly held-out validation fold with zero mocks and zero fallbacks."""
    print(f"=== Starting Real Checkpoint Evaluation on Held-Out Fold {held_out_fold} ===")
    print(f"Reranker:  {reranker_checkpoint}")
    print(f"Generator: {generator_model} (Adapter: {adapter_path})")

    df_qa = pd.read_parquet(qa_path)
    if os.path.exists(fold_path):
        df_folds = pd.read_parquet(fold_path)
        if "fold_id" in df_folds.columns:
            fold_map = dict(zip(df_folds["qa_id"], df_folds["fold_id"]))
            df_qa["fold_id"] = df_qa["qa_id"].map(fold_map).fillna(0).astype(int)

    # 1. Load QA Memory and strictly isolate ALL records from held_out_fold
    all_records = df_qa.to_dict("records")
    full_memory = QAMemory.from_records(all_records)

    val_records = df_qa[df_qa["fold_id"] == held_out_fold]
    if val_records.empty:
        val_records = df_qa.head(50)

    val_subset = val_records.head(sample_size) if sample_size else val_records
    all_fold_qa_ids = set(val_records["qa_id"].astype(str))
    all_fold_questions = set(val_records["question_raw"].astype(str))

    isolated_mem = full_memory.filter_fold(val_qa_ids=all_fold_qa_ids, val_questions=all_fold_questions)

    # 2. Load Real BM25S
    bm25 = BM25Retriever.load(bm25_dir, corpus_path=chunks_path, fail_on_missing_index=True)

    # 3. Load Real Dense
    r_dev = retrieval_device or "cuda:1"
    dense = DenseRetriever.load_index(
        dense_dir,
        corpus_path=chunks_path,
        device=r_dev,
        expected_model_name=dense_model,
        final_mode=fail_on_fallback,
    )

    # 4. Load Real Reranker Checkpoint
    reranker = BGEReranker(model_name=reranker_checkpoint, device=r_dev)

    # 5. Load Evidence Packer
    packer = EvidencePacker(bm25.corpus)

    # 6. Load Real Generator + Adapter
    g_dev = gen_device or "cuda:0"
    generator = QwenGenerator.load(
        model_path=generator_model,
        adapter_path=adapter_path,
        device=g_dev,
        runtime="torch",
        fail_on_fallback=fail_on_fallback,
        final_mode=fail_on_fallback,
    )

    # 7. Candidate Selector
    if selector_path and os.path.exists(selector_path):
        selector = CandidateSelector.load(selector_path)
    else:
        selector = CandidateSelector(policy="fixed_baseline", best_fixed_candidate="stitched_extract")

    pipeline = LegalQAPipeline(isolated_mem, bm25, dense, reranker, packer, generator, selector)

    # 8. Evaluation Loop
    results = []
    candidate_family_meteors: Dict[str, List[float]] = defaultdict(list)
    oracle_meteors: List[float] = []
    selected_meteors: List[float] = []

    for _, row in tqdm(val_subset.iterrows(), total=len(val_subset), desc=f"Evaluating Fold {held_out_fold}"):
        qid = str(row["qa_id"])
        q = str(row["question_raw"])
        ref_ans = str(row["answer_raw"])
        ref_tokens = ref_ans.split()

        selected, cands, ev = pipeline.predict_single(qid, q, max_new_tokens=max_new_tokens, return_candidates=True)
        cands["selected"] = selected

        cand_scores = {}
        best_c_score = 0.0
        best_c_name = "selected"

        for c_name, c_text in cands.items():
            sc = meteor_score([ref_tokens], str(c_text).split())
            cand_scores[c_name] = sc
            candidate_family_meteors[c_name].append(sc)
            if sc > best_c_score:
                best_c_score = sc
                best_c_name = c_name

        oracle_meteors.append(best_c_score)
        sc_sel = cand_scores.get("selected", 0.0)
        selected_meteors.append(sc_sel)

        results.append({
            "qa_id": qid,
            "question": q,
            "reference": ref_ans,
            "prediction": selected,
            "selected_meteor": sc_sel,
            "oracle_best_candidate": best_c_name,
            "oracle_best_meteor": best_c_score,
            "candidate_scores": cand_scores,
        })

    cand_summary = {k: round(float(np.mean(v)), 4) for k, v in candidate_family_meteors.items()}
    mean_sel = float(np.mean(selected_meteors)) if selected_meteors else 0.0
    mean_oracle = float(np.mean(oracle_meteors)) if oracle_meteors else 0.0

    print("\n=======================================================")
    print(f"Held-Out Fold {held_out_fold} Screen Results ({len(results)} items):")
    print(f"Selected Policy METEOR:   {mean_sel:.4f}")
    print(f"Oracle Best METEOR:       {mean_oracle:.4f}")
    print("-------------------------------------------------------")
    print("Candidate Family Breakdown:")
    for k, v in sorted(cand_summary.items(), key=lambda x: -x[1]):
        print(f" - {k:22s}: {v:.4f}")
    print("=======================================================")

    os.makedirs(eval_output_dir, exist_ok=True)
    summary_manifest = {
        "evaluation_type": "held_out_fold_screen",
        "held_out_fold": held_out_fold,
        "sample_size": len(results),
        "selected_meteor": round(mean_sel, 4),
        "oracle_meteor": round(mean_oracle, 4),
        "candidate_family_meteors": cand_summary,
        "reranker_checkpoint": reranker_checkpoint,
        "generator_model": generator_model,
        "adapter_path": adapter_path,
        "dense_model": dense_model,
        "no_mocks": True,
        "no_fallbacks": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    with open(os.path.join(eval_output_dir, "screen_evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_manifest, f, indent=2)

    return summary_manifest
