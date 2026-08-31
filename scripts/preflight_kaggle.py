"""Canonical preflight validation script for Kaggle Dual-T4 LegalQA execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.dense import compute_chunk_ids_hash
from src.common.security import assert_no_secrets_in_workspace
from src.task2.production_config import load_production_selection
from scripts.audit_parameters import audit_parameter_budget, verify_config_consistency, load_config_file


def run_preflight_checks(
    pipeline_config_path: str = "configs/pipeline.yaml",
    models_config_path: str = "configs/models.yaml",
    production_config_path: str = "configs/production_selection.yaml",
    require_cuda: bool = False,
    expected_gpu_count: int = 2,
    allow_single_gpu: bool = False,
    check_dataset_files: bool = False,
    data_dir: str = "artifacts/task2/data",
    bm25_dir: Optional[str] = "artifacts/task2/indexes/bm25",
    dek21_dir: Optional[str] = "artifacts/task2/indexes/dek21",
    public_path: Optional[str] = "artifacts/raw/public-official.json",
    stack: str = "stack_a",
    require_training_files: bool = False,
    verify_dense_hash: bool = False,
    expected_dense_model: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
) -> Dict[str, Any]:
    """Perform comprehensive preflight checks and return diagnostic status."""
    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}

    print("=== Running LegalQA Kaggle Preflight Diagnostics ===")

    # 1. Security Check
    try:
        assert_no_secrets_in_workspace(Path.cwd())
        details["security"] = "PASS: No unwhitelisted secrets found."
    except Exception as e:
        errors.append(f"Security preflight failure: {e}")

    # 2. Config existence and consistency
    if not os.path.exists(pipeline_config_path):
        errors.append(f"Missing pipeline config: {pipeline_config_path}")
    if not os.path.exists(models_config_path):
        errors.append(f"Missing models config: {models_config_path}")
    if os.path.exists(production_config_path):
        try:
            prod_cfg = load_production_selection(production_config_path)
            details["production_selection"] = {
                "status": prod_cfg.status,
                "stack": prod_cfg.stack,
                "use_task_tuned_reranker": prod_cfg.use_task_tuned_reranker,
                "use_qlora": prod_cfg.use_qlora,
                "candidate_policy": prod_cfg.candidate_policy,
                "best_fixed_candidate": prod_cfg.best_fixed_candidate,
            }
        except Exception as e:
            errors.append(f"Invalid production selection config ({production_config_path}): {e}")
    else:
        warnings.append(f"Production selection config not found at {production_config_path}")

    if not errors:
        cons = verify_config_consistency(pipeline_config_path, models_config_path)
        details["config_consistency"] = cons
        if not cons["is_consistent"]:
            warnings.extend(cons["issues"])

        audit = audit_parameter_budget(models_config_path, stack=stack)
        details["parameter_audit"] = audit
        if not audit["is_compliant"]:
            errors.append(
                f"Parameter budget exceeded! Total: {audit['total_learned_parameters']:,} >= {audit['limit']:,}"
            )
        else:
            print(f"Parameter Budget: {audit['total_learned_parameters']:,} / {audit['limit']:,} (Margin: {audit['margin']:,}) - COMPLIANT")

    # 3. Hardware & Dual-T4 GPU Checks (P0-11)
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_cnt = torch.cuda.device_count() if cuda_avail else 0
        details["cuda"] = {"available": cuda_avail, "gpu_count": gpu_cnt}
        if require_cuda:
            if not cuda_avail:
                errors.append("CUDA is required but not available.")
            elif gpu_cnt < expected_gpu_count and not allow_single_gpu:
                errors.append(
                    f"Production preflight requires {expected_gpu_count} CUDA GPUs (Dual-T4), found {gpu_cnt}. "
                    f"Set allow_single_gpu=True only for explicit single-GPU smoke runs."
                )
        if cuda_avail:
            for i in range(gpu_cnt):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                print(f" - GPU {i}: {name} (Compute: {cap[0]}.{cap[1]})")
    except ImportError:
        if require_cuda:
            errors.append("PyTorch is not installed.")

    # 4. Dataset Files Check
    chunks_path = os.path.join(data_dir, "legal_chunks.parquet")
    corpus_row_count = None
    corpus_chunk_ids_hash = None

    if check_dataset_files:
        required_data_files = [
            chunks_path,
            os.path.join(data_dir, "qa_unique.parquet"),
            os.path.join(data_dir, "known_qa.json"),
            os.path.join(data_dir, "fold_assignments.parquet"),
        ]
        if require_training_files:
            required_data_files.append(os.path.join(data_dir, "reranker_training_pairs.parquet"))

        for df_path in required_data_files:
            if not os.path.exists(df_path):
                errors.append(f"Missing required data file: {df_path}")
            else:
                sz = os.path.getsize(df_path) / (1024 * 1024)
                print(f" - Data file: {os.path.basename(df_path)} ({sz:.2f} MB)")

        if os.path.exists(chunks_path):
            try:
                df_chunks = pd.read_parquet(chunks_path)
                corpus_row_count = len(df_chunks)
                chunk_ids = [str(c) for c in df_chunks["chunk_id"]]
                corpus_chunk_ids_hash = compute_chunk_ids_hash(chunk_ids)
                print(f" - Verified legal_chunks.parquet: {corpus_row_count} rows (Chunk Hash: {corpus_chunk_ids_hash[:12]}...)")
            except Exception as e:
                errors.append(f"Failed to read legal_chunks.parquet: {e}")

    # 5. BM25 Index Validation (P0-12)
    if bm25_dir and os.path.exists(bm25_dir):
        bm25_manifest_path = os.path.join(bm25_dir, "bm25_manifest.json")
        if not os.path.exists(bm25_manifest_path):
            warnings.append(f"BM25 manifest not found at: {bm25_manifest_path}")
        else:
            try:
                with open(bm25_manifest_path, "r", encoding="utf-8") as f:
                    bm25_meta = json.load(f)
                bm25_corpus_size = bm25_meta.get("corpus_size")
                if corpus_row_count is not None and bm25_corpus_size is not None and bm25_corpus_size != corpus_row_count:
                    errors.append(
                        f"BM25 index corpus size ({bm25_corpus_size}) does not match legal_chunks rows ({corpus_row_count})."
                    )
                else:
                    print(f" - BM25 Index: validated ({bm25_corpus_size or 'ok'} docs)")
            except Exception as e:
                errors.append(f"Failed to read BM25 manifest: {e}")

    # 6. DEk21 Dense Index Validation (P0-12, P0-13)
    if dek21_dir and os.path.exists(dek21_dir):
        dense_manifest_path = os.path.join(dek21_dir, "dense_manifest.json")
        if not os.path.exists(dense_manifest_path):
            dense_manifest_path = os.path.join(dek21_dir, "dek21_manifest.json")

        emb_file = os.path.join(dek21_dir, "embeddings.npy")
        if not os.path.exists(emb_file):
            errors.append(f"Dense embeddings file missing: {emb_file}")
        elif not os.path.exists(dense_manifest_path):
            errors.append(f"Dense manifest missing: {dense_manifest_path}")
        else:
            try:
                with open(dense_manifest_path, "r", encoding="utf-8") as f:
                    dense_meta = json.load(f)

                model_id = dense_meta.get("model_id") or dense_meta.get("model_name")
                dtype = dense_meta.get("dtype")
                dim = dense_meta.get("dim")
                manifest_rows = dense_meta.get("corpus_rows")
                manifest_chunk_hash = dense_meta.get("chunk_ids_sha256")

                if expected_dense_model and model_id != expected_dense_model:
                    errors.append(f"Dense model mismatch! Expected '{expected_dense_model}', found '{model_id}' in index manifest.")
                if dtype != "float16":
                    errors.append(f"Dense dtype must be 'float16', found '{dtype}' in index manifest.")
                if dim != 768 and "bge-m3" not in str(model_id).lower():
                    errors.append(f"Dense dimension mismatch! Expected 768, found {dim} in index manifest.")
                if corpus_row_count is not None and manifest_rows is not None and manifest_rows != corpus_row_count:
                    errors.append(f"Dense index rows ({manifest_rows}) != legal_chunks rows ({corpus_row_count}).")
                if corpus_chunk_ids_hash is not None and manifest_chunk_hash and manifest_chunk_hash != corpus_chunk_ids_hash:
                    errors.append("Dense index chunk_ids_sha256 does not match corpus chunk IDs hash.")

                # Fast check shape without reading whole array into RAM
                import numpy as np
                mmap_emb = np.load(emb_file, mmap_mode="r")
                if str(mmap_emb.dtype) != "float16":
                    errors.append(f"Dense embeddings.npy file has actual dtype '{mmap_emb.dtype}', expected 'float16'.")
                if manifest_rows and mmap_emb.shape[0] != manifest_rows:
                    errors.append(f"Dense embeddings.npy shape {mmap_emb.shape[0]} != manifest rows {manifest_rows}.")

                if verify_dense_hash:
                    print("Verifying full SHA256 checksum of embeddings.npy (this takes a few seconds)...")
                    with open(emb_file, "rb") as f:
                        file_sha = hashlib.sha256(f.read()).hexdigest()
                    expected_sha = dense_meta.get("embeddings_sha256")
                    if expected_sha and file_sha != expected_sha:
                        errors.append(f"Dense embeddings.npy SHA256 mismatch! Expected {expected_sha[:12]}, got {file_sha[:12]}")
                    else:
                        print(f" - Dense Embeddings SHA256 verified: {file_sha[:12]}...")

                print(f" - Dense DEk21 Index: verified ({mmap_emb.shape}, dtype={mmap_emb.dtype})")
            except Exception as e:
                errors.append(f"Failed to validate Dense DEk21 index: {e}")

    # 7. Public Test Set Schema Check
    if public_path and os.path.exists(public_path):
        try:
            with open(public_path, "r", encoding="utf-8") as f:
                pub_data = json.load(f)
            pub_len = len(pub_data)
            details["public_count"] = pub_len
            if pub_len != 1000:
                errors.append(f"Public test set has {pub_len} queries (expected exactly 1000).")
            else:
                print(f" - Public dataset: 1000 queries verified from {public_path}")
        except Exception as e:
            errors.append(f"Failed to parse public dataset at {public_path}: {e}")

    passed = len(errors) == 0
    print(f"\nPreflight Result: {'PASSED' if passed else 'FAILED'}")
    if errors:
        print("Errors:")
        for err in errors:
            print(f" [ERROR] {err}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f" [WARN] {w}")

    return {
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="LegalQA Kaggle Preflight Diagnostics")
    parser.add_argument("--pipeline_config", default="configs/pipeline.yaml")
    parser.add_argument("--models_config", default="configs/models.yaml")
    parser.add_argument("--production_config", default="configs/production_selection.yaml")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--expected_gpus", type=int, default=2)
    parser.add_argument("--allow_single_gpu", action="store_true")
    parser.add_argument("--check_data", action="store_true")
    parser.add_argument("--data_dir", default="artifacts/task2/data")
    parser.add_argument("--bm25_dir", default="artifacts/task2/indexes/bm25")
    parser.add_argument("--dek21_dir", default="artifacts/task2/indexes/dek21")
    parser.add_argument("--public_path", default="artifacts/raw/public-official.json")
    parser.add_argument("--verify_dense_hash", action="store_true")
    parser.add_argument("--stack", default="stack_a", choices=["stack_a", "stack_b"])
    parser.add_argument("--require_training", action="store_true")
    args = parser.parse_args()

    res = run_preflight_checks(
        pipeline_config_path=args.pipeline_config,
        models_config_path=args.models_config,
        production_config_path=args.production_config,
        require_cuda=args.require_cuda,
        expected_gpu_count=args.expected_gpus,
        allow_single_gpu=args.allow_single_gpu,
        check_dataset_files=args.check_data,
        data_dir=args.data_dir,
        bm25_dir=args.bm25_dir,
        dek21_dir=args.dek21_dir,
        public_path=args.public_path,
        verify_dense_hash=args.verify_dense_hash,
        stack=args.stack,
        require_training_files=args.require_training,
    )
    if not res["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
