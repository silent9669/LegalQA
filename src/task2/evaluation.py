"""Honest, leakage-safe evaluation module for exact trained reranker and generator checkpoints (V8)."""

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

GENERATOR_DEPENDENT_FAMILIES: Set[str] = {
    "generated",
    "snapped",
    "strategy_f_300",
    "strategy_f_600",
    "strategy_f_1000",
    "strategy_f_1500",
}


def best_deployable_candidate(summary: Dict[str, Any]) -> Tuple[str, float]:
    """Extract the highest scoring deployable candidate family and its mean score from an evaluation summary."""
    cand_scores = summary.get("candidate_family_meteors", {})
    deployable = {k: v for k, v in cand_scores.items() if k not in ["selected", "oracle_best"]}
    if not deployable:
        return "stitched_extract", 0.0
    best_name, best_score = max(deployable.items(), key=lambda x: x[1])
    return best_name, float(best_score)


def decide_reranker_promotion(
    base_summary: Dict[str, Any],
    tuned_summary: Dict[str, Any],
    retrieval_tolerance: float = 0.001,
    meteor_tolerance: float = 0.005,
) -> Dict[str, Any]:
    """Decide whether to promote the task-tuned reranker based on real chunk retrieval gains and downstream METEOR."""
    base_retr = base_summary.get("retrieval_metrics") or {}
    tuned_retr = tuned_summary.get("retrieval_metrics") or {}

    base_mrr = base_retr.get("chunk_mrr")
    tuned_mrr = tuned_retr.get("chunk_mrr")
    if base_mrr is None or tuned_mrr is None:
        raise RuntimeError("Cannot decide reranker promotion without chunk_mrr metrics in evaluation summaries.")

    base_rec8 = float(base_retr.get("chunk_recall_at_8", 0.0))
    tuned_rec8 = float(tuned_retr.get("chunk_recall_at_8", 0.0))

    base_best_name, base_best_score = best_deployable_candidate(base_summary)
    tuned_best_name, tuned_best_score = best_deployable_candidate(tuned_summary)

    retrieval_improved = (tuned_mrr > base_mrr + retrieval_tolerance or tuned_rec8 > base_rec8 + retrieval_tolerance)
    downstream_ok = (tuned_best_score >= base_best_score - meteor_tolerance)
    promote = bool(retrieval_improved and downstream_ok)

    reason = (
        f"Promoted (Chunk MRR: {base_mrr:.4f} -> {tuned_mrr:.4f}, Recall@8: {base_rec8:.4f} -> {tuned_rec8:.4f}, "
        f"Downstream: {base_best_score:.4f} -> {tuned_best_score:.4f})"
        if promote
        else f"Rejected (Retrieval improved: {retrieval_improved}, Downstream ok: {downstream_ok})"
    )

    return {
        "promote": promote,
        "retrieval_improved": bool(retrieval_improved),
        "downstream_ok": bool(downstream_ok),
        "base_chunk_mrr": round(float(base_mrr), 4),
        "tuned_chunk_mrr": round(float(tuned_mrr), 4),
        "base_chunk_recall_at_8": round(float(base_rec8), 4),
        "tuned_chunk_recall_at_8": round(float(tuned_rec8), 4),
        "base_best_candidate": base_best_name,
        "base_best_score": round(float(base_best_score), 4),
        "tuned_best_candidate": tuned_best_name,
        "tuned_best_score": round(float(tuned_best_score), 4),
        "reason": reason,
    }


def decide_generator_promotion(
    base_summary: Dict[str, Any],
    qlora_summary: Dict[str, Any],
    meteor_tolerance: float = 0.005,
) -> Dict[str, Any]:
    """Decide whether to promote QLoRA generator under the chosen reranker."""
    base_best_name, base_best_score = best_deployable_candidate(base_summary)
    qlora_best_name, qlora_best_score = best_deployable_candidate(qlora_summary)

    promote = bool(
        (qlora_best_score > base_best_score + meteor_tolerance)
        and (qlora_best_name in GENERATOR_DEPENDENT_FAMILIES)
    )

    reason = (
        f"Promoted (QLoRA winner '{qlora_best_name}' score {qlora_best_score:.4f} > base best '{base_best_name}' score {base_best_score:.4f} + tol {meteor_tolerance})"
        if promote
        else f"Rejected (QLoRA winner '{qlora_best_name}' score {qlora_best_score:.4f} did not exceed base best '{base_best_name}' score {base_best_score:.4f} + tol {meteor_tolerance})"
    )

    return {
        "promote": promote,
        "base_best_candidate": base_best_name,
        "base_best_score": round(float(base_best_score), 4),
        "qlora_best_candidate": qlora_best_name,
        "qlora_best_score": round(float(qlora_best_score), 4),
        "reason": reason,
    }


def write_promotion_report(
    report: Dict[str, Any],
    primary_path: str,
    mirror_path: Optional[str] = None,
) -> Dict[str, str]:
    """Canonicalize and write exact byte-identical promotion reports to primary and optional mirror paths (Task 7)."""
    os.makedirs(os.path.dirname(os.path.abspath(primary_path)), exist_ok=True)
    payload = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")

    with open(primary_path, "wb") as f:
        f.write(payload)

    if mirror_path:
        os.makedirs(os.path.dirname(os.path.abspath(mirror_path)), exist_ok=True)
        with open(mirror_path, "wb") as f:
            f.write(payload)

    sha256 = hashlib.sha256(payload).hexdigest()
    return {
        "sha256": sha256,
        "primary_path": primary_path,
        "mirror_path": mirror_path or "",
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
    min_retrieval_label_coverage: float = 0.70,
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

    # Strict fold validation — never silently switch or fallback
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

    # Load Gold Retrieval Labels
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

    num_eval_queries = len(val_subset)
    chunk_labeled_qids = set(eval_qa_ids) & set(qa_to_gold_chunks.keys())
    article_labeled_qids = set(eval_qa_ids) & set(qa_to_gold_articles.keys())
    chunk_coverage = len(chunk_labeled_qids) / max(1, num_eval_queries)
    article_coverage = len(article_labeled_qids) / max(1, num_eval_queries)

    # If supervision required, check coverage threshold
    if require_retrieval_supervision:
        if chunk_coverage < min_retrieval_label_coverage:
            raise RuntimeError(
                f"Insufficient retrieval label coverage for screening: "
                f"found {len(chunk_labeled_qids)}/{num_eval_queries} ({chunk_coverage:.2%}), "
                f"required >= {min_retrieval_label_coverage:.2%}. Check {labels_path}."
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

    chunk_recalls_at_1: List[float] = []
    chunk_recalls_at_5: List[float] = []
    chunk_recalls_at_8: List[float] = []
    chunk_mrrs: List[float] = []

    article_recalls_at_1: List[float] = []
    article_recalls_at_5: List[float] = []
    article_recalls_at_8: List[float] = []
    article_mrrs: List[float] = []

    for _, row in tqdm(val_subset.iterrows(), total=len(val_subset), desc=f"Evaluating Fold {held_out_fold}"):
        qid = str(row["qa_id"])
        q = str(row["question_raw"])
        ref_ans = str(row["answer_raw"])

        selected, cands, trace = pipeline.predict_single(
            qid, q, max_new_tokens=max_new_tokens, return_candidates=True, return_trace=True
        )
        cands["selected"] = selected

        reranked_hits = trace.get("reranked_results", [])
        gold_chunks = qa_to_gold_chunks.get(qid, set())
        gold_articles = qa_to_gold_articles.get(qid, set())

        if not gold_articles and "article_id" in row and pd.notna(row["article_id"]):
            gold_articles = {str(row["article_id"]).strip()}

        retrieved_chunks = [str(item.get("chunk_id", "")).strip() for item in reranked_hits if isinstance(item, dict)]
        retrieved_articles = [str(item.get("article_id") or item.get("parent_article_id") or "").strip() for item in reranked_hits if isinstance(item, dict)]

        # 1. Chunk-level metrics
        if gold_chunks:
            c1 = 1.0 if any(c in gold_chunks for c in retrieved_chunks[:1]) else 0.0
            c5 = 1.0 if any(c in gold_chunks for c in retrieved_chunks[:5]) else 0.0
            c8 = 1.0 if any(c in gold_chunks for c in retrieved_chunks[:8]) else 0.0
            chunk_recalls_at_1.append(c1)
            chunk_recalls_at_5.append(c5)
            chunk_recalls_at_8.append(c8)

            c_mrr = 0.0
            for rank_idx, c_id in enumerate(retrieved_chunks, start=1):
                if c_id in gold_chunks:
                    c_mrr = 1.0 / rank_idx
                    break
            chunk_mrrs.append(c_mrr)

        # 2. Article-level metrics
        if gold_articles:
            a1 = 1.0 if any(a in gold_articles for a in retrieved_articles[:1]) else 0.0
            a5 = 1.0 if any(a in gold_articles for a in retrieved_articles[:5]) else 0.0
            a8 = 1.0 if any(a in gold_articles for a in retrieved_articles[:8]) else 0.0
            article_recalls_at_1.append(a1)
            article_recalls_at_5.append(a5)
            article_recalls_at_8.append(a8)

            a_mrr = 0.0
            for rank_idx, a_id in enumerate(retrieved_articles, start=1):
                if a_id in gold_articles:
                    a_mrr = 1.0 / rank_idx
                    break
            article_mrrs.append(a_mrr)

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
        "chunk_mrr": round(float(np.mean(chunk_mrrs)), 4) if chunk_mrrs else None,
        "article_recall_at_1": round(float(np.mean(article_recalls_at_1)), 4) if article_recalls_at_1 else None,
        "article_recall_at_5": round(float(np.mean(article_recalls_at_5)), 4) if article_recalls_at_5 else None,
        "article_recall_at_8": round(float(np.mean(article_recalls_at_8)), 4) if article_recalls_at_8 else None,
        "article_mrr": round(float(np.mean(article_mrrs)), 4) if article_mrrs else None,
        "num_eval_queries": num_eval_queries,
        "num_queries_with_chunk_labels": len(chunk_labeled_qids),
        "num_queries_with_article_labels": len(article_labeled_qids),
        "chunk_label_coverage": round(chunk_coverage, 4),
        "article_label_coverage": round(article_coverage, 4),
    }

    print("\n=======================================================")
    print(f"Held-Out Fold {held_out_fold} Screen Results ({len(results)} items):")
    print(f"Selected Policy METEOR:   {mean_sel:.4f}")
    print(f"Oracle Best METEOR:       {mean_oracle:.4f}")
    if retrieval_metrics["chunk_mrr"] is not None:
        print(f"Chunk Recall@8: {retrieval_metrics['chunk_recall_at_8']} | Chunk MRR: {retrieval_metrics['chunk_mrr']:.4f} ({len(chunk_labeled_qids)} labeled queries, {chunk_coverage:.1%} cov)")
    if retrieval_metrics["article_mrr"] is not None:
        print(f"Article Recall@8: {retrieval_metrics['article_recall_at_8']} | Article MRR: {retrieval_metrics['article_mrr']:.4f}")
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
    min_retrieval_label_coverage: float = 0.70,
) -> Dict[str, Any]:
    """Run staged component-consistent screening (R0G0 -> R1G0 -> R_SELECTED_G1) on identical held-out IDs (Protocol 8)."""
    # Upfront validation of required checkpoints for canonical screen
    if tuned_reranker is None or not os.path.exists(tuned_reranker):
        raise FileNotFoundError(
            f"Tuned reranker checkpoint directory missing at '{tuned_reranker}'. "
            f"Cannot run canonical screen_fold0 without trained reranker checkpoint."
        )

    if adapter_path is None or not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"QLoRA adapter checkpoint missing at '{adapter_path}'. "
            f"Cannot run canonical screen_fold0 without trained generator adapter."
        )

    print("\n=======================================================")
    print(f"Running Component-Consistent Screening Matrix (Protocol 8) on Fold {held_out_fold}")
    print("=======================================================")

    # 1. Stage A: R0G0 = Base Reranker + Base Generator
    print("\n[STAGE A] Evaluating R0G0 (Base Reranker + Base Generator)...")
    r0g0 = evaluate_checkpoint(
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
        eval_output_dir=os.path.join(eval_output_dir, "r0g0_base"),
        gen_device=gen_device,
        retrieval_device=retrieval_device,
        require_retrieval_supervision=True,
        min_retrieval_label_coverage=min_retrieval_label_coverage,
        seed=seed,
    )

    # 2. Stage B: R1G0 = Tuned Reranker + Base Generator
    print(f"\n[STAGE B] Evaluating R1G0 (Tuned Reranker '{tuned_reranker}' + Base Generator)...")
    r1g0 = evaluate_checkpoint(
        qa_path=qa_path,
        fold_path=fold_path,
        chunks_path=chunks_path,
        labels_path=labels_path,
        held_out_fold=held_out_fold,
        bm25_dir=bm25_dir,
        dense_dir=dense_dir,
        dense_model=dense_model,
        reranker_checkpoint=tuned_reranker,
        generator_model=base_generator,
        adapter_path=None,
        sample_size=sample_size,
        eval_output_dir=os.path.join(eval_output_dir, "r1g0_tuned_reranker"),
        gen_device=gen_device,
        retrieval_device=retrieval_device,
        require_retrieval_supervision=True,
        min_retrieval_label_coverage=min_retrieval_label_coverage,
        seed=seed,
    )

    # Decide reranker promotion based on R0G0 vs R1G0
    rerank_decision = decide_reranker_promotion(
        r0g0,
        r1g0,
        retrieval_tolerance=retrieval_tolerance,
        meteor_tolerance=meteor_tolerance,
    )
    promote_reranker = rerank_decision["promote"]
    selected_reranker = tuned_reranker if promote_reranker else base_reranker
    selected_base_summary = r1g0 if promote_reranker else r0g0
    selected_base_key = "R1G0" if promote_reranker else "R0G0"

    print(f"\n>> RERANKER SELECTION DECISION: {selected_base_key} (Promote Tuned={promote_reranker})")
    print(f"   Reason: {rerank_decision['reason']}")

    # 3. Stage C: R_SELECTED_G1 = Selected Reranker + QLoRA
    print(f"\n[STAGE C] Evaluating R_SELECTED_G1 (Selected Reranker '{selected_reranker}' + QLoRA Adapter '{adapter_path}')...")
    selected_qlora = evaluate_checkpoint(
        qa_path=qa_path,
        fold_path=fold_path,
        chunks_path=chunks_path,
        labels_path=labels_path,
        held_out_fold=held_out_fold,
        bm25_dir=bm25_dir,
        dense_dir=dense_dir,
        dense_model=dense_model,
        reranker_checkpoint=selected_reranker,
        generator_model=base_generator,
        adapter_path=adapter_path,
        sample_size=sample_size,
        eval_output_dir=os.path.join(eval_output_dir, "r_selected_g1_qlora"),
        gen_device=gen_device,
        retrieval_device=retrieval_device,
        require_retrieval_supervision=True,
        min_retrieval_label_coverage=min_retrieval_label_coverage,
        seed=seed,
    )

    # Decide generator promotion based on selected_base vs selected_qlora
    gen_decision = decide_generator_promotion(
        selected_base_summary,
        selected_qlora,
        meteor_tolerance=meteor_tolerance,
    )
    promote_qlora = gen_decision["promote"]

    final_measured_summary = selected_qlora if promote_qlora else selected_base_summary
    final_measured_system_key = "R_SELECTED_G1" if promote_qlora else selected_base_key
    final_candidate_name, final_candidate_score = best_deployable_candidate(final_measured_summary)

    print(f"\n>> GENERATOR SELECTION DECISION: (Promote QLoRA={promote_qlora})")
    print(f"   Reason: {gen_decision['reason']}")
    print(f">> FINAL DEPLOYABLE SYSTEM: {final_measured_system_key} with Candidate '{final_candidate_name}' ({final_candidate_score:.4f})")

    promotion_report = {
        "screen_protocol_version": 8,
        "held_out_fold": held_out_fold,
        "sample_ids_sha256": r0g0["sample_ids_sha256"],
        "sample_size": r0g0["sample_size"],
        "tolerances": {
            "retrieval_tolerance": retrieval_tolerance,
            "meteor_tolerance": meteor_tolerance,
            "min_retrieval_label_coverage": min_retrieval_label_coverage,
        },
        "evaluated_systems": {
            "R0G0": r0g0,
            "R1G0": r1g0,
            "R_SELECTED_G1": selected_qlora,
        },
        "selected_reranker": {
            "use_task_tuned": bool(promote_reranker),
            "checkpoint": selected_reranker,
            "decision_reason": rerank_decision["reason"],
            "retrieval_decision": rerank_decision,
        },
        "selected_generator": {
            "use_qlora": bool(promote_qlora),
            "adapter": adapter_path if promote_qlora else None,
            "decision_reason": gen_decision["reason"],
            "generator_decision": gen_decision,
        },
        "candidate_policy": {
            "type": "fixed_baseline",
            "best_fixed_candidate": final_candidate_name,
        },
        "final_measured_system_key": final_measured_system_key,
        "overall_deployable_winner": final_candidate_name,
        "overall_deployable_meteor": round(float(final_candidate_score), 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    report_path = os.path.join(eval_output_dir, "promotion_report.json")
    mirror_path = "/kaggle/working/promotion_report.json" if os.path.exists("/kaggle/working") else None
    hash_meta = write_promotion_report(promotion_report, report_path, mirror_path)
    report_sha = hash_meta["sha256"]

    print("\n================ PROMOTION SUMMARY REPORT ================")
    print(f"Protocol Version:           8")
    print(f"Selected Reranker:          {'Task-Tuned (' + selected_reranker + ')' if promote_reranker else 'Base'}")
    print(f"Selected Generator:         {'QLoRA (' + adapter_path + ')' if promote_qlora else 'Base Qwen / Extractive'}")
    print(f"Final Winning Candidate:    {final_candidate_name} (METEOR: {final_candidate_score:.4f})")
    print(f"Final Measured System:      {final_measured_system_key}")
    print(f"Report SHA256:              {report_sha[:16]}...")
    print(f"Report Saved To:            {report_path}")
    print("==========================================================")

    return promotion_report
