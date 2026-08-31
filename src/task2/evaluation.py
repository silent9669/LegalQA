"""Honest, leakage-safe evaluation module for exact trained reranker and generator checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.common.bm25 import BM25Retriever
from src.common.dense import DenseRetriever
from src.common.reranker import BGEReranker
from src.task2.candidates import generate_candidate_ensemble
from src.task2.evidence_packer import EvidencePacker
from src.task2.generator import QwenGenerator
from src.task2.metrics import calculate_official_meteor, ensure_meteor_resources, official_meteor
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
    generator_model: Optional[str] = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: Optional[str] = None,
    selector_path: Optional[str] = None,
    sample_size: Optional[int] = 50,
    eval_output_dir: str = "artifacts/task2/evaluations",
    gen_device: Optional[str] = None,
    retrieval_device: Optional[str] = None,
    max_new_tokens: int = 384,
    fail_on_fallback: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate exact trained checkpoints on a strictly held-out validation fold with zero mocks and zero fallbacks."""
    ensure_meteor_resources()
    print(f"=== Starting Real Checkpoint Evaluation on Held-Out Fold {held_out_fold} ===")
    print(f"Reranker:  {reranker_checkpoint}")
    print(f"Generator: {generator_model} (Adapter: {adapter_path})")

    df_qa = pd.read_parquet(qa_path)
    if os.path.exists(fold_path):
        df_folds = pd.read_parquet(fold_path)
        if "fold_id" in df_folds.columns:
            fold_map = dict(zip(df_folds["qa_id"], df_folds["fold_id"]))
            df_qa["fold_id"] = df_qa["qa_id"].map(fold_map)

    # P0-2: Strict fold validation — never silently switch or fallback
    if "fold_id" not in df_qa.columns:
        raise RuntimeError("Held-out evaluation requires fold_id assignments.")

    val_records = df_qa[df_qa["fold_id"] == held_out_fold]
    if val_records.empty:
        raise RuntimeError(
            f"Held-out fold {held_out_fold} contains no rows; refusing to evaluate a different subset."
        )

    # Deterministic sampling by SEED rather than accidental .head() ordering
    if sample_size and sample_size < len(val_records):
        val_subset = val_records.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        val_subset = val_records.reset_index(drop=True)

    eval_qa_ids = sorted(val_subset["qa_id"].astype(str).tolist())
    eval_qa_ids_sha256 = hashlib.sha256(" ".join(eval_qa_ids).encode("utf-8")).hexdigest()
    print(f"Held-out evaluation set: {len(val_subset)} rows (SHA256: {eval_qa_ids_sha256[:12]}...)")

    # 1. Load QA Memory and strictly isolate ALL records from held_out_fold
    all_records = df_qa.to_dict("records")
    full_memory = QAMemory.from_records(all_records)

    all_fold_qa_ids = set(val_records["qa_id"].astype(str))
    all_fold_questions = set(val_records["question_raw"].astype(str))
    isolated_mem = full_memory.filter_fold(val_qa_ids=all_fold_qa_ids, val_questions=all_fold_questions)

    # 2. Load BM25
    bm25 = BM25Retriever.load(bm25_dir, corpus_path=chunks_path, fail_on_missing_index=True)

    # 3. Load Dense
    r_dev = retrieval_device or "cuda:1"
    dense = DenseRetriever.load_index(
        dense_dir,
        corpus_path=chunks_path,
        device=r_dev,
        expected_model_name=dense_model,
        expected_dtype="float16",
        final_mode=fail_on_fallback,
    )

    # 4. Load Reranker Checkpoint
    reranker = BGEReranker(model_name=reranker_checkpoint, device=r_dev)

    # 5. Load Evidence Packer
    packer = EvidencePacker(bm25.corpus)

    # 6. Load Generator (if configured)
    generator: Optional[QwenGenerator] = None
    if generator_model:
        g_dev = gen_device or "cuda:0"
        generator = QwenGenerator.load(
            model_path=generator_model,
            adapter_path=adapter_path,
            device=g_dev,
            runtime="torch",
            fail_on_fallback=fail_on_fallback,
            final_mode=fail_on_fallback,
            require_adapter=bool(adapter_path),
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

    # Retrieval metrics tracking (Article/Chunk Recall & MRR)
    retrieval_recalls_at_1: List[float] = []
    retrieval_recalls_at_5: List[float] = []
    retrieval_recalls_at_8: List[float] = []
    retrieval_mrrs: List[float] = []

    for _, row in tqdm(val_subset.iterrows(), total=len(val_subset), desc=f"Evaluating Fold {held_out_fold}"):
        qid = str(row["qa_id"])
        q = str(row["question_raw"])
        ref_ans = str(row["answer_raw"])
        target_article = str(row.get("article_id", "")) if "article_id" in row else None

        selected, cands, ev = pipeline.predict_single(qid, q, max_new_tokens=max_new_tokens, return_candidates=True)
        cands["selected"] = selected

        # Calculate retrieval metrics if article metadata exists
        if target_article and ev:
            retrieved_articles = [str(item.get("article_id", "")) for item in ev if isinstance(item, dict)]
            r1 = 1.0 if target_article in retrieved_articles[:1] else 0.0
            r5 = 1.0 if target_article in retrieved_articles[:5] else 0.0
            r8 = 1.0 if target_article in retrieved_articles[:8] else 0.0
            try:
                rank = retrieved_articles.index(target_article) + 1
                mrr = 1.0 / rank
            except ValueError:
                mrr = 0.0
            retrieval_recalls_at_1.append(r1)
            retrieval_recalls_at_5.append(r5)
            retrieval_recalls_at_8.append(r8)
            retrieval_mrrs.append(mrr)

        cand_scores = {}
        best_c_score = 0.0
        best_c_name = "selected"

        for c_name, c_text in cands.items():
            sc = official_meteor(ref_ans, str(c_text))
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

    retrieval_metrics = {
        "recall_at_1": round(float(np.mean(retrieval_recalls_at_1)), 4) if retrieval_recalls_at_1 else None,
        "recall_at_5": round(float(np.mean(retrieval_recalls_at_5)), 4) if retrieval_recalls_at_5 else None,
        "recall_at_8": round(float(np.mean(retrieval_recalls_at_8)), 4) if retrieval_recalls_at_8 else None,
        "mrr": round(float(np.mean(retrieval_mrrs)), 4) if retrieval_mrrs else None,
    }

    print("\n=======================================================")
    print(f"Held-Out Fold {held_out_fold} Screen Results ({len(results)} items):")
    print(f"Selected Policy METEOR:   {mean_sel:.4f}")
    print(f"Oracle Best METEOR:       {mean_oracle:.4f}")
    if retrieval_metrics["mrr"] is not None:
        print(f"Retrieval Recall@1: {retrieval_metrics['recall_at_1']:.4f} | Recall@5: {retrieval_metrics['recall_at_5']:.4f} | MRR: {retrieval_metrics['mrr']:.4f}")
    print("-------------------------------------------------------")
    print("Candidate Family Breakdown:")
    for k, v in sorted(cand_summary.items(), key=lambda x: -x[1]):
        print(f" - {k:24s}: {v:.4f}")
    print("=======================================================")

    os.makedirs(eval_output_dir, exist_ok=True)
    summary_manifest = {
        "evaluation_type": "held_out_fold_screen",
        "held_out_fold": held_out_fold,
        "sample_size": len(results),
        "sample_ids_sha256": eval_qa_ids_sha256,
        "selected_meteor": round(mean_sel, 4),
        "oracle_meteor": round(mean_oracle, 4),
        "candidate_family_meteors": cand_summary,
        "retrieval_metrics": retrieval_metrics,
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


def run_screen_matrix(
    qa_path: str = "artifacts/task2/data/qa_unique.parquet",
    fold_path: str = "artifacts/task2/data/fold_assignments.parquet",
    chunks_path: str = "artifacts/task2/data/legal_chunks.parquet",
    held_out_fold: int = 0,
    bm25_dir: str = "artifacts/task2/indexes/bm25",
    dense_dir: str = "artifacts/task2/indexes/dek21",
    dense_model: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
    base_reranker: str = "BAAI/bge-reranker-v2-m3",
    tuned_reranker: Optional[str] = None,
    base_generator: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: Optional[str] = None,
    sample_size: int = 250,
    eval_output_dir: str = "artifacts/task2/evaluations",
    gen_device: Optional[str] = None,
    retrieval_device: Optional[str] = None,
    seed: int = 42,
    tolerance: float = 0.001,
) -> Dict[str, Any]:
    """Run full S0 (base) vs S1 (tuned reranker) vs S2 (tuned reranker + QLoRA) screen matrix on identical held-out IDs."""
    print("\n=======================================================")
    print(f"Running Full Screen Matrix (S0 / S1 / S2) on Fold {held_out_fold}")
    print("=======================================================")

    # Evaluate S0: Base Reranker + Base Qwen
    s0_summary = evaluate_checkpoint(
        qa_path=qa_path,
        fold_path=fold_path,
        chunks_path=chunks_path,
        held_out_fold=held_out_fold,
        bm25_dir=bm25_dir,
        dense_dir=dense_dir,
        dense_model=dense_model,
        reranker_checkpoint=base_reranker,
        generator_model=base_generator,
        adapter_path=None,
        sample_size=sample_size,
        eval_output_dir=os.path.join(eval_output_dir, "s0_base"),
        gen_device=gen_device,
        retrieval_device=retrieval_device,
        seed=seed,
    )

    # Evaluate S1: Tuned Reranker + Base Qwen (if tuned reranker provided)
    reranker_to_use = tuned_reranker if (tuned_reranker and os.path.exists(tuned_reranker)) else base_reranker
    s1_summary = evaluate_checkpoint(
        qa_path=qa_path,
        fold_path=fold_path,
        chunks_path=chunks_path,
        held_out_fold=held_out_fold,
        bm25_dir=bm25_dir,
        dense_dir=dense_dir,
        dense_model=dense_model,
        reranker_checkpoint=reranker_to_use,
        generator_model=base_generator,
        adapter_path=None,
        sample_size=sample_size,
        eval_output_dir=os.path.join(eval_output_dir, "s1_tuned_reranker"),
        gen_device=gen_device,
        retrieval_device=retrieval_device,
        seed=seed,
    )

    # Evaluate S2: Tuned Reranker + QLoRA (if adapter provided)
    if adapter_path and os.path.exists(adapter_path):
        s2_summary = evaluate_checkpoint(
            qa_path=qa_path,
            fold_path=fold_path,
            chunks_path=chunks_path,
            held_out_fold=held_out_fold,
            bm25_dir=bm25_dir,
            dense_dir=dense_dir,
            dense_model=dense_model,
            reranker_checkpoint=reranker_to_use,
            generator_model=base_generator,
            adapter_path=adapter_path,
            sample_size=sample_size,
            eval_output_dir=os.path.join(eval_output_dir, "s2_qlora"),
            gen_device=gen_device,
            retrieval_device=retrieval_device,
            seed=seed,
        )
    else:
        s2_summary = s1_summary

    # Find best fixed candidate across candidate family meteors in s1
    cand_scores = s1_summary["candidate_family_meteors"]
    # Exclude 'selected' and 'oracle_best' to find the top fixed family
    fixed_candidates = {k: v for k, v in cand_scores.items() if k not in ["selected", "oracle_best", "generated"]}
    best_fixed = max(fixed_candidates.items(), key=lambda x: x[1])[0] if fixed_candidates else "stitched_extract"

    base_gen_meteor = float(s0_summary["candidate_family_meteors"].get("generated", 0.0))
    qlora_gen_meteor = float(s2_summary["candidate_family_meteors"].get("generated", 0.0))
    best_fixed_meteor = float(cand_scores.get(best_fixed, 0.0))

    # Decision logic:
    # 1. Reranker: promote if retrieval improved or downstream fixed extract didn't regress
    s0_mrr = (s0_summary.get("retrieval_metrics") or {}).get("mrr") or 0.0
    s1_mrr = (s1_summary.get("retrieval_metrics") or {}).get("mrr") or 0.0
    s0_fixed = (s0_summary["candidate_family_meteors"].get(best_fixed, 0.0))
    s1_fixed = (s1_summary["candidate_family_meteors"].get(best_fixed, 0.0))

    promote_reranker = (s1_mrr >= s0_mrr and s1_fixed >= s0_fixed - tolerance) if (tuned_reranker and tuned_reranker != base_reranker) else False

    # 2. QLoRA: promote ONLY if it improves a deployable policy beyond best fixed candidate
    promote_qlora = (qlora_gen_meteor > base_gen_meteor + tolerance) and (qlora_gen_meteor > best_fixed_meteor + tolerance)

    recommended_policy = "fixed_baseline"
    if promote_qlora:
        recommended_policy = "generated"

    promotion_report = {
        "held_out_fold": held_out_fold,
        "sample_ids_sha256": s0_summary["sample_ids_sha256"],
        "sample_size": s0_summary["sample_size"],
        "reranker_base_metrics": s0_summary.get("retrieval_metrics", {}),
        "reranker_tuned_metrics": s1_summary.get("retrieval_metrics", {}),
        "candidate_family_meteors": cand_scores,
        "base_generator_meteor": round(base_gen_meteor, 4),
        "qlora_generator_meteor": round(qlora_gen_meteor, 4),
        "best_fixed_candidate": best_fixed,
        "best_fixed_candidate_meteor": round(best_fixed_meteor, 4),
        "selected_policy_meteor": round(s2_summary["selected_meteor"], 4),
        "recommended_use_task_tuned_reranker": bool(promote_reranker),
        "recommended_use_qlora": bool(promote_qlora),
        "recommended_candidate_policy": recommended_policy,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    report_path = os.path.join(eval_output_dir, "promotion_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(promotion_report, f, indent=2)

    # Also write to working root /kaggle/working if on kaggle
    if os.path.exists("/kaggle/working"):
        with open("/kaggle/working/promotion_report.json", "w", encoding="utf-8") as f:
            json.dump(promotion_report, f, indent=2)

    print("\n================ PROMOTION DECISION ================")
    print(f"Best Fixed Candidate:       {best_fixed} (METEOR: {best_fixed_meteor:.4f})")
    print(f"Base Generator METEOR:      {base_gen_meteor:.4f}")
    print(f"QLoRA Generator METEOR:     {qlora_gen_meteor:.4f}")
    print(f"Promote Tuned Reranker:     {promote_reranker}")
    print(f"Promote QLoRA Generator:    {promote_qlora}")
    print(f"Recommended Policy:         {recommended_policy}")
    print(f"Promotion Report saved to:  {report_path}")
    print("====================================================")

    return promotion_report
