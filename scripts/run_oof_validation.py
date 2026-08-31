"""5-Fold Out-Of-Fold (OOF) cross-validation and official whitespace-tokenized METEOR evaluation (V9)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever
from src.common.hashing import sha256_file
from src.common.normalize import clean_legal_text, normalize_question
from src.common.reranker import BGEReranker
from src.task2.article_stitcher import ArticleStitcher
from src.task2.checkpoint_manifest import load_generator_manifest, load_reranker_manifest
from src.task2.generator import QwenGenerator
from src.task2.metrics import (
    calculate_official_meteor,
    calculate_rouge_l,
    ensure_meteor_resources,
    official_meteor,
)
from src.task2.predict import LegalQAPipeline
from src.task2.qa_memory import QAMemory
from src.task2.source_snap import (
    generate_candidate_ensemble,
    select_best_answer_candidate,
    snap_facts_to_evidence,
)


def assert_fold_reranker_checkpoint(
    checkpoint_path: str,
    fold_id: int,
    expected_base_model: str = "BAAI/bge-reranker-v2-m3",
) -> Dict[str, Any]:
    """Validate fold-specific reranker checkpoint provenance (Task 6)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Reranker checkpoint directory missing: {checkpoint_path}")
    manifest = load_reranker_manifest(checkpoint_path)
    if manifest.get("smoke_only"):
        raise ValueError(f"Fold {fold_id} reranker checkpoint at {checkpoint_path} is marked as smoke_only!")
    exc = manifest.get("val_fold_excluded", manifest.get("val_fold"))
    if exc != fold_id:
        raise ValueError(
            f"Fold {fold_id} reranker checkpoint at {checkpoint_path} has val_fold_excluded={exc} != {fold_id}."
        )
    scope = manifest.get("training_scope")
    if scope != f"folds_excluding_{fold_id}":
        raise ValueError(
            f"Fold {fold_id} reranker training_scope '{scope}' != 'folds_excluding_{fold_id}'."
        )
    base_m = manifest.get("base_model_id") or manifest.get("base_model") or manifest.get("base_model_name_or_path")
    if not base_m or (expected_base_model and base_m != expected_base_model):
        raise ValueError(
            f"Fold {fold_id} reranker base model mismatch: expected '{expected_base_model}', found '{base_m}'."
        )
    return manifest


def assert_fold_generator_checkpoint(
    adapter_path: str,
    fold_id: int,
    expected_base_model: str = "Qwen/Qwen2.5-3B-Instruct",
) -> Dict[str, Any]:
    """Validate fold-specific generator adapter provenance (Task 6)."""
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Generator adapter checkpoint directory missing: {adapter_path}")
    manifest = load_generator_manifest(adapter_path)
    if manifest.get("smoke_only"):
        raise ValueError(f"Fold {fold_id} generator adapter at {adapter_path} is marked as smoke_only!")
    exc = manifest.get("val_fold_excluded", manifest.get("val_fold"))
    if exc != fold_id:
        raise ValueError(
            f"Fold {fold_id} generator adapter at {adapter_path} has val_fold_excluded={exc} != {fold_id}."
        )
    scope = manifest.get("training_scope")
    if scope != f"folds_excluding_{fold_id}":
        raise ValueError(
            f"Fold {fold_id} generator training_scope '{scope}' != 'folds_excluding_{fold_id}'."
        )
    base_m = manifest.get("base_model_id") or manifest.get("base_model") or manifest.get("base_model_name_or_path")
    if not base_m or (expected_base_model and base_m != expected_base_model):
        raise ValueError(
            f"Fold {fold_id} generator base model mismatch: expected '{expected_base_model}', found '{base_m}'."
        )
    return manifest


def validate_full_mode_contract(
    *,
    bm25_dir: str,
    dek21_dir: str,
    held_out_fold: Optional[int],
    fold_checkpoint_map: Optional[Dict[int, Dict[str, str]]],
    n_splits: int,
    reranker_checkpoint: str,
    adapter_path: Optional[str],
    retrieval_device: Optional[str],
    gen_device: Optional[str],
) -> None:
    """Strictly validate prerequisites for mode='full' (Task 4)."""
    if held_out_fold is None and not fold_checkpoint_map:
        raise RuntimeError(
            "True full neural OOF requires one checkpoint per fold via fold_checkpoint_map "
            "or an explicit single held_out_fold. Reusing one fold checkpoint across all folds is invalid."
        )

    if not os.path.isdir(bm25_dir):
        raise RuntimeError(f"Full mode requires BM25 index directory at: {bm25_dir}")

    if not os.path.isdir(dek21_dir):
        raise RuntimeError(f"Full mode requires DEk21 dense index directory at: {dek21_dir}")

    emb_file = os.path.join(dek21_dir, "embeddings.npy")
    if not os.path.isfile(emb_file):
        raise RuntimeError(f"Full mode requires real DEk21 embeddings.npy at: {emb_file}")

    dense_man = os.path.join(dek21_dir, "dense_manifest.json")
    if not os.path.isfile(dense_man):
        dense_man = os.path.join(dek21_dir, "dek21_manifest.json")
    if not os.path.isfile(dense_man):
        raise RuntimeError(f"Full mode requires DEk21 dense manifest under: {dek21_dir}")

    r_dev = str(retrieval_device or "")
    if not r_dev.startswith("cuda"):
        raise RuntimeError(f"Full mode requires retrieval_device to be a CUDA device, found: '{retrieval_device}'.")

    g_dev = str(gen_device or "")
    if not g_dev.startswith("cuda"):
        raise RuntimeError(f"Full mode requires gen_device to be a CUDA device, found: '{gen_device}'.")


def validate_fold_checkpoint_map(
    fold_checkpoint_map: Dict[int, Dict[str, str]],
    *,
    target_folds: List[int],
    require_reranker: bool = False,
    require_adapter: bool = False,
) -> None:
    """Strictly validate that fold_checkpoint_map covers all target folds upfront (Task 5)."""
    missing = sorted(set(target_folds) - set(fold_checkpoint_map.keys()))
    if missing:
        raise RuntimeError(f"fold_checkpoint_map missing folds: {missing}. Expected all folds in {target_folds}.")

    for f_id in target_folds:
        f_info = fold_checkpoint_map[f_id]
        if require_reranker and "reranker" not in f_info:
            raise RuntimeError(f"fold_checkpoint_map missing 'reranker' checkpoint for fold {f_id}.")
        if require_adapter and "adapter" not in f_info:
            raise RuntimeError(f"fold_checkpoint_map missing 'adapter' checkpoint for fold {f_id}.")


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
    fold_checkpoint_map: Optional[Dict[int, Dict[str, str]]] = None,
    device: Optional[str] = None,
    gen_device: Optional[str] = None,
    retrieval_device: Optional[str] = None,
    max_new_tokens: int = 384,
) -> Dict[str, Any]:
    ensure_meteor_resources()
    print(f"=== Starting LegalQA Task 2 OOF Validation (Mode: {mode.upper()}) ===")

    r_dev = retrieval_device or device or "cuda:1"
    g_dev = gen_device or device or "cuda:0"

    if mode == "fast":
        print("*******************************************************************************")
        print("  DIAGNOSTIC ONLY — NOT VALID FOR MODEL QUALITY, CHECKPOINT VERIFICATION OR PROMOTION")
        print("*******************************************************************************")
    elif mode == "full":
        validate_full_mode_contract(
            bm25_dir=bm25_dir,
            dek21_dir=dek21_dir,
            held_out_fold=held_out_fold,
            fold_checkpoint_map=fold_checkpoint_map,
            n_splits=n_splits,
            reranker_checkpoint=reranker_checkpoint,
            adapter_path=adapter_path,
            retrieval_device=r_dev,
            gen_device=g_dev,
        )
        if held_out_fold is not None:
            if reranker_checkpoint and reranker_checkpoint != "BAAI/bge-reranker-v2-m3" and os.path.exists(reranker_checkpoint):
                assert_fold_reranker_checkpoint(reranker_checkpoint, held_out_fold, "BAAI/bge-reranker-v2-m3")
            if adapter_path and os.path.exists(adapter_path):
                assert_fold_generator_checkpoint(adapter_path, held_out_fold, model_path)

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
    if mode == "full":
        bm25 = BM25Retriever.load(bm25_dir, corpus_path=chunks_path, fail_on_missing_index=True)
    else:
        bm25 = BM25Retriever.load(bm25_dir, corpus_path=chunks_path) if os.path.exists(bm25_dir) else BM25Retriever()
        if not bm25.corpus and os.path.exists(chunks_path):
            df_c = pd.read_parquet(chunks_path)
            bm25.fit(df_c.to_dict("records"))

    # 2. Dense DEk21
    if mode == "full":
        print(f"Loading real DEk21 embeddings on {r_dev}...")
        dense = DEk21Retriever.load_index(
            dek21_dir,
            corpus_path=chunks_path,
            device=r_dev,
            expected_model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
            expected_dtype="float16",
            final_mode=True,
        )
    else:
        print("Using fast lexical retriever for fast validation...")
        dense = DEk21Retriever(model_name="mock", device=r_dev)
        if bm25.corpus:
            dense.fit_mock(bm25.corpus)

    # 3. Reranker
    if mode == "full":
        print(f"Loading Neural Cross-Encoder Reranker ({reranker_checkpoint}) on {r_dev}...")
        reranker = BGEReranker(model_name=reranker_checkpoint, device=r_dev)
    else:
        print("Using fast lexical reranker for validation...")
        reranker = BGEReranker(model_name="mock", device="cpu")

    # 4. Article Stitcher
    stitcher = ArticleStitcher(bm25.corpus if bm25.corpus else [])

    # 5. Generator
    if mode == "full":
        print(f"Loading Qwen2.5-3B Generator on {g_dev}...")
        generator = QwenGenerator.load(
            model_path=model_path,
            adapter_path=adapter_path,
            device=g_dev,
            runtime="torch",
            fail_on_fallback=True,
            final_mode=True,
            require_adapter=bool(adapter_path),
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

    # Strictly respect held_out_fold when passed
    if held_out_fold is not None:
        target_folds = [held_out_fold]
    else:
        target_folds = list(range(n_splits))

    # Upfront validation of fold checkpoint map if provided (Task 5)
    if mode == "full" and fold_checkpoint_map:
        validate_fold_checkpoint_map(
            fold_checkpoint_map,
            target_folds=target_folds,
            require_reranker=bool(reranker_checkpoint and reranker_checkpoint != "BAAI/bge-reranker-v2-m3"),
            require_adapter=bool(adapter_path),
        )

    samples_per_fold = (num_eval_samples // len(target_folds)) if num_eval_samples else None
    print(f"\nEvaluating {'all' if not samples_per_fold else samples_per_fold} samples per fold across {len(target_folds)} folds: {target_folds}...")

    for fold_id in target_folds:
        fold_records = df_qa[df_qa["fold_id"] == fold_id]
        if fold_records.empty:
            raise RuntimeError(f"Fold {fold_id} has no records to evaluate.")

        if samples_per_fold and samples_per_fold < len(fold_records):
            val_subset = fold_records.sample(n=samples_per_fold, random_state=42).reset_index(drop=True)
        else:
            val_subset = fold_records.reset_index(drop=True)

        # Strict zero-leakage fold memory: exclude ALL records assigned to this validation fold
        all_fold_qa_ids = set(fold_records["qa_id"].astype(str))
        all_fold_questions = set(fold_records["question_raw"].astype(str))
        isolated_mem = full_memory.filter_fold(val_qa_ids=all_fold_qa_ids, val_questions=all_fold_questions)

        # If per-fold checkpoints are provided, reload fold-specific models
        current_reranker = reranker
        current_generator = generator
        if fold_checkpoint_map and fold_id in fold_checkpoint_map:
            ckpt_info = fold_checkpoint_map[fold_id]
            if "reranker" in ckpt_info:
                r_path = ckpt_info["reranker"]
                if mode == "full":
                    assert_fold_reranker_checkpoint(r_path, fold_id, "BAAI/bge-reranker-v2-m3")
                current_reranker = BGEReranker(model_name=r_path, device=r_dev)

            if "adapter" in ckpt_info:
                a_path = ckpt_info["adapter"]
                if mode == "full":
                    assert_fold_generator_checkpoint(a_path, fold_id, model_path)
                current_generator = QwenGenerator.load(
                    model_path=model_path,
                    adapter_path=a_path,
                    device=g_dev,
                    runtime="torch",
                    fail_on_fallback=True,
                    final_mode=True,
                    require_adapter=True,
                )

        pipeline = LegalQAPipeline(isolated_mem, bm25, dense, current_reranker, stitcher, current_generator)

        fold_preds = []
        fold_refs = []

        for _, row in tqdm(val_subset.iterrows(), total=len(val_subset), desc=f"Fold {fold_id}"):
            qid = str(row["qa_id"])
            q = str(row["question_raw"])
            ref_ans = str(row["answer_raw"])

            selected, cands, ev = pipeline.predict_single(
                qid, q, max_new_tokens=max_new_tokens, return_candidates=True
            )
            cands["selected"] = selected

            # Score each candidate against reference
            cand_meteors = {}
            best_cand_score = 0.0
            best_cand_name = "selected"

            for name, cand_text in cands.items():
                sc = official_meteor(ref_ans, str(cand_text))
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
    print(f"OOF METEOR:  {mean_meteor:.4f} ± {std_meteor:.4f}")
    print(f"OOF ROUGE-L: {mean_rouge:.4f}")
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
        "evaluated_folds": target_folds,
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
