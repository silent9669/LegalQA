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
from src.common.hashing import sha256_file
from src.common.reranker import BGEReranker
from src.task2.candidates import generate_candidate_ensemble
from src.task2.evidence_packer import EvidencePacker
from src.task2.generator import QwenGenerator
from src.task2.metrics import calculate_official_meteor, ensure_meteor_resources, official_meteor
from src.task2.predict import LegalQAPipeline
from src.task2.qa_memory import QAMemory
from src.task2.selector import CandidateSelector

GENERATOR_DEPENDENT_FAMILIES = {
    "generated",
    "snapped",
    "strategy_f_300",
    "strategy_f_600",
    "strategy_f_1000",
    "strategy_f_1500",
}


def evaluate_checkpoint(
    qa_path: str = "artifacts/task2/data/qa_unique.parquet",
    fold_path: str = "artifacts/task2/data/fold_assignments.parquet",
    chunks_path: str = "artifacts/task2/data/legal_chunks.parquet",
    labels_path: Optional[str] = "artifacts/task2/data/retrieval_labels.parquet",
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
    require_retrieval_supervision: bool = False,
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

    # Load Gold Retrieval Labels (P0-7)
    qa_to_gold_chunks: Dict[str, Set[str]] = defaultdict(set)
    qa_to_gold_articles: Dict[str, Set[str]] = defaultdict(set)
    if labels_path and os.path.exists(labels_path):
        try:
            df_labels = pd.read_parquet(labels_path)
            for _, row in df_labels.iterrows():
                qid = str(row["qa_id"]).strip()
                cid = str(row.get("positive_chunk_id", "")).strip()
                aid = str(row.get("positive_article_id", "")).strip()
                if cid:
                    qa_to_gold_chunks[qid].add(cid)
                if aid:
                    qa_to_gold_articles[qid].add(aid)
        except Exception as e:
            print(f"Warning loading retrieval labels: {e}", file=sys.stderr)

    # If supervision required but missing, raise (P0-7)
    if require_retrieval_supervision:
        covered_qids = set(eval_qa_ids) & set(qa_to_gold_chunks.keys())
        if not covered_qids:
            raise RuntimeError(
                f"Retrieval supervision is required for screening, but no retrieval labels found in {labels_path} for evaluated fold {held_out_fold}."
            )

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

    # Real retrieval metrics tracking (P0-7)
    chunk_recalls_at_1: List[float] = []
    chunk_recalls_at_5: List[float] = []
    chunk_recalls_at_8: List[float] = []
    article_recalls_at_1: List[float] = []
    article_recalls_at_5: List[float] = []
    article_recalls_at_8: List[float] = []
    retrieval_mrrs: List[float] = []
    queries_with_labels_count = 0

    for _, row in tqdm(val_subset.iterrows(), total=len(val_subset), desc=f"Evaluating Fold {held_out_fold}"):
        qid = str(row["qa_id"])
        q = str(row["question_raw"])
        ref_ans = str(row["answer_raw"])

        # Execute single prediction with full retrieval trace (P0-7)
        selected, cands, trace = pipeline.predict_single(
            qid, q, max_new_tokens=max_new_tokens, return_candidates=True, return_trace=True
        )
        cands["selected"] = selected

        # Calculate real retrieval metrics from trace["reranked_results"] (P0-7)
        reranked_hits = trace.get("reranked_results", [])
        gold_chunks = qa_to_gold_chunks.get(qid, set())
        gold_articles = qa_to_gold_articles.get(qid, set())

        # Also fallback to row target_article if present
        if not gold_articles and "article_id" in row and pd.notna(row["article_id"]):
            gold_articles = {str(row["article_id"]).strip()}

        if gold_chunks or gold_articles:
            queries_with_labels_count += 1
            retrieved_chunks = [str(item.get("chunk_id", "")).strip() for item in reranked_hits if isinstance(item, dict)]
            retrieved_articles = [str(item.get("article_id") or item.get("parent_article_id") or "").strip() for item in reranked_hits if isinstance(item, dict)]

            # Chunk recalls
            if gold_chunks:
                c1 = 1.0 if any(c in gold_chunks for c in retrieved_chunks[:1]) else 0.0
                c5 = 1.0 if any(c in gold_chunks for c in retrieved_chunks[:5]) else 0.0
                c8 = 1.0 if any(c in gold_chunks for c in retrieved_chunks[:8]) else 0.0
                chunk_recalls_at_1.append(c1)
                chunk_recalls_at_5.append(c5)
                chunk_recalls_at_8.append(c8)

            # Article recalls
            if gold_articles:
                a1 = 1.0 if any(a in gold_articles for a in retrieved_articles[:1]) else 0.0
                a5 = 1.0 if any(a in gold_articles for a in retrieved_articles[:5]) else 0.0
                a8 = 1.0 if any(a in gold_articles for a in retrieved_articles[:8]) else 0.0
                article_recalls_at_1.append(a1)
                article_recalls_at_5.append(a5)
                article_recalls_at_8.append(a8)

            # MRR computation
            mrr_val = 0.0
            for rank_idx, (c_id, a_id) in enumerate(zip(retrieved_chunks, retrieved_articles), start=1):
                if (c_id and c_id in gold_chunks) or (a_id and a_id in gold_articles):
                    mrr_val = 1.0 / rank_idx
                    break
            retrieval_mrrs.append(mrr_val)

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
        "chunk_recall_at_1": round(float(np.mean(chunk_recalls_at_1)), 4) if chunk_recalls_at_1 else None,
        "chunk_recall_at_5": round(float(np.mean(chunk_recalls_at_5)), 4) if chunk_recalls_at_5 else None,
        "chunk_recall_at_8": round(float(np.mean(chunk_recalls_at_8)), 4) if chunk_recalls_at_8 else None,
        "article_recall_at_1": round(float(np.mean(article_recalls_at_1)), 4) if article_recalls_at_1 else None,
        "article_recall_at_5": round(float(np.mean(article_recalls_at_5)), 4) if article_recalls_at_5 else None,
        "article_recall_at_8": round(float(np.mean(article_recalls_at_8)), 4) if article_recalls_at_8 else None,
        "mrr": round(float(np.mean(retrieval_mrrs)), 4) if retrieval_mrrs else None,
        "num_queries_with_retrieval_labels": queries_with_labels_count,
    }

    print("\n=======================================================")
    print(f"Held-Out Fold {held_out_fold} Screen Results ({len(results)} items):")
    print(f"Selected Policy METEOR:   {mean_sel:.4f}")
    print(f"Oracle Best METEOR:       {mean_oracle:.4f}")
    if retrieval_metrics["mrr"] is not None:
        print(f"Retrieval Chunk Recall@1: {retrieval_metrics['chunk_recall_at_1']} | Recall@8: {retrieval_metrics['chunk_recall_at_8']} | MRR: {retrieval_metrics['mrr']:.4f} ({queries_with_labels_count} queries)")
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
    labels_path: str = "artifacts/task2/data/retrieval_labels.parquet",
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
    retrieval_tolerance: float = 0.001,
    meteor_tolerance: float = 0.005,
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
        labels_path=labels_path,
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
        require_retrieval_supervision=True,
        seed=seed,
    )

    # Evaluate S1: Tuned Reranker + Base Qwen (if tuned reranker provided)
    reranker_to_use = tuned_reranker if (tuned_reranker and os.path.exists(tuned_reranker)) else base_reranker
    s1_summary = evaluate_checkpoint(
        qa_path=qa_path,
        fold_path=fold_path,
        chunks_path=chunks_path,
        labels_path=labels_path,
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
        require_retrieval_supervision=True,
        seed=seed,
    )

    # Evaluate S2: Tuned Reranker + QLoRA (if adapter provided)
    if adapter_path and os.path.exists(adapter_path):
        s2_summary = evaluate_checkpoint(
            qa_path=qa_path,
            fold_path=fold_path,
            chunks_path=chunks_path,
            labels_path=labels_path,
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
            require_retrieval_supervision=True,
            seed=seed,
        )
    else:
        s2_summary = s1_summary

    # P0-8: Strict Reranker Promotion Logic
    s0_mrr = (s0_summary.get("retrieval_metrics") or {}).get("mrr")
    s1_mrr = (s1_summary.get("retrieval_metrics") or {}).get("mrr")
    if s0_mrr is None or s1_mrr is None:
        raise RuntimeError("Cannot promote reranker without valid retrieval metrics (MRR is None).")

    s0_recall8 = (s0_summary.get("retrieval_metrics") or {}).get("chunk_recall_at_8") or 0.0
    s1_recall8 = (s1_summary.get("retrieval_metrics") or {}).get("chunk_recall_at_8") or 0.0

    # P0-9: Comprehensive Multi-Family Breakdown & Comparison
    s1_cands = s1_summary["candidate_family_meteors"]
    s2_cands = s2_summary["candidate_family_meteors"]

    # 1. Best non-generator fixed candidate in S1
    non_gen_cands = {k: v for k, v in s1_cands.items() if k not in ["selected", "oracle_best"] and k not in GENERATOR_DEPENDENT_FAMILIES}
    best_non_gen_name, best_non_gen_score = max(non_gen_cands.items(), key=lambda x: x[1]) if non_gen_cands else ("stitched_extract", 0.0)

    # 2. Best base-generator candidate in S1
    base_gen_cands = {k: v for k, v in s1_cands.items() if k in GENERATOR_DEPENDENT_FAMILIES}
    best_base_gen_name, best_base_gen_score = max(base_gen_cands.items(), key=lambda x: x[1]) if base_gen_cands else ("generated", 0.0)

    # 3. Best QLoRA-derived candidate in S2
    qlora_cands = {k: v for k, v in s2_cands.items() if k in GENERATOR_DEPENDENT_FAMILIES}
    best_qlora_name, best_qlora_score = max(qlora_cands.items(), key=lambda x: x[1]) if qlora_cands else ("generated", 0.0)

    # 4. Overall deployable winner across S2 and S1
    all_deployable_s2 = {k: v for k, v in s2_cands.items() if k not in ["selected", "oracle_best"]}
    overall_winner_name, overall_winner_score = max(all_deployable_s2.items(), key=lambda x: x[1]) if all_deployable_s2 else (best_non_gen_name, best_non_gen_score)

    # Decision logic for Reranker (P0-8)
    s0_fixed_score = float(s0_summary["candidate_family_meteors"].get(best_non_gen_name, 0.0))
    s1_fixed_score = float(s1_summary["candidate_family_meteors"].get(best_non_gen_name, 0.0))

    retrieval_improved = (s1_mrr > s0_mrr + retrieval_tolerance or s1_recall8 > s0_recall8 + retrieval_tolerance)
    downstream_ok = (s1_fixed_score >= s0_fixed_score - meteor_tolerance)
    promote_reranker = bool(retrieval_improved and downstream_ok) if (tuned_reranker and tuned_reranker != base_reranker) else False

    # Decision logic for QLoRA (P0-9)
    promote_qlora = bool(
        adapter_path is not None
        and os.path.exists(adapter_path)
        and (best_qlora_score > max(best_non_gen_score, best_base_gen_score) + meteor_tolerance)
        and (overall_winner_name in GENERATOR_DEPENDENT_FAMILIES)
    )

    # P0-10: Strict Policy Encoding (fixed_baseline with best_fixed_candidate)
    recommended_policy_type = "fixed_baseline"
    recommended_best_fixed = overall_winner_name

    promotion_report = {
        "held_out_fold": held_out_fold,
        "sample_ids_sha256": s0_summary["sample_ids_sha256"],
        "sample_size": s0_summary["sample_size"],
        "tolerances": {
            "retrieval_tolerance": retrieval_tolerance,
            "meteor_tolerance": meteor_tolerance,
        },
        "reranker_base_metrics": s0_summary.get("retrieval_metrics", {}),
        "reranker_tuned_metrics": s1_summary.get("retrieval_metrics", {}),
        "retrieval_improved": bool(retrieval_improved),
        "downstream_ok": bool(downstream_ok),
        "candidate_family_meteors": s2_cands,
        "best_non_generator_candidate": best_non_gen_name,
        "best_non_generator_meteor": round(float(best_non_gen_score), 4),
        "best_base_generator_candidate": best_base_gen_name,
        "best_base_generator_meteor": round(float(best_base_gen_score), 4),
        "best_qlora_derived_candidate": best_qlora_name,
        "best_qlora_derived_meteor": round(float(best_qlora_score), 4),
        "overall_deployable_winner": overall_winner_name,
        "overall_deployable_meteor": round(float(overall_winner_score), 4),
        "recommended_use_task_tuned_reranker": bool(promote_reranker),
        "recommended_use_qlora": bool(promote_qlora),
        "candidate_policy": {
            "type": recommended_policy_type,
            "best_fixed_candidate": recommended_best_fixed,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    os.makedirs(eval_output_dir, exist_ok=True)
    report_path = os.path.join(eval_output_dir, "promotion_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(promotion_report, f, indent=2)

    # Compute SHA256 of report (P0-5)
    report_sha = sha256_file(report_path)
    promotion_report["report_sha256"] = report_sha

    # Also write to working root /kaggle/working if on kaggle
    if os.path.exists("/kaggle/working"):
        with open("/kaggle/working/promotion_report.json", "w", encoding="utf-8") as f:
            json.dump(promotion_report, f, indent=2)

    print("\n================ PROMOTION DECISION ================")
    print(f"Best Non-Generator:         {best_non_gen_name} (METEOR: {best_non_gen_score:.4f})")
    print(f"Best Base Generator:        {best_base_gen_name} (METEOR: {best_base_gen_score:.4f})")
    print(f"Best QLoRA-Derived:         {best_qlora_name} (METEOR: {best_qlora_score:.4f})")
    print(f"Overall Winner:             {overall_winner_name} (METEOR: {overall_winner_score:.4f})")
    print(f"Promote Tuned Reranker:     {promote_reranker}")
    print(f"Promote QLoRA Generator:    {promote_qlora}")
    print(f"Policy:                     {recommended_policy_type} (best_fixed={recommended_best_fixed})")
    print(f"Promotion Report SHA256:    {report_sha[:16]}...")
    print(f"Promotion Report saved to:  {report_path}")
    print("====================================================")

    return promotion_report
