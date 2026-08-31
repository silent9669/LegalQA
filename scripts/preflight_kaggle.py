"""Canonical preflight validation script for Kaggle Dual-T4 LegalQA execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.security import assert_no_secrets_in_workspace
from scripts.audit_parameters import audit_parameter_budget, verify_config_consistency, load_config_file


def run_preflight_checks(
    pipeline_config_path: str = "configs/pipeline.yaml",
    models_config_path: str = "configs/models.yaml",
    require_cuda: bool = False,
    expected_gpu_count: int = 2,
    check_dataset_files: bool = False,
    data_dir: str = "artifacts/task2/data",
    index_dir: Optional[str] = None,
    bm25_dir: str = "artifacts/task2/indexes/bm25",
    dek21_dir: str = "artifacts/task2/indexes/dek21",
    public_path: Optional[str] = "artifacts/raw/public-official.json",
    stack: str = "stack_a",
    require_training_files: bool = False,
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

    # 3. Hardware & CUDA Checks
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_cnt = torch.cuda.device_count() if cuda_avail else 0
        details["cuda"] = {"available": cuda_avail, "gpu_count": gpu_cnt}
        if require_cuda:
            if not cuda_avail:
                errors.append("CUDA is required but not available.")
            elif gpu_cnt < expected_gpu_count:
                errors.append(f"Expected {expected_gpu_count} CUDA GPUs, found {gpu_cnt}.")
        if cuda_avail:
            for i in range(gpu_cnt):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                print(f" - GPU {i}: {name} (Compute: {cap[0]}.{cap[1]})")
    except ImportError:
        if require_cuda:
            errors.append("PyTorch is not installed.")

    # 4. Dataset Files Check
    if check_dataset_files:
        required_data_files = [
            os.path.join(data_dir, "legal_chunks.parquet"),
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

    # 5. Public Test Set Schema Check
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
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--expected_gpus", type=int, default=2)
    parser.add_argument("--check_data", action="store_true")
    parser.add_argument("--data_dir", default="artifacts/task2/data")
    parser.add_argument("--stack", default="stack_a", choices=["stack_a", "stack_b"])
    parser.add_argument("--require_training", action="store_true")
    args = parser.parse_args()

    res = run_preflight_checks(
        pipeline_config_path=args.pipeline_config,
        models_config_path=args.models_config,
        require_cuda=args.require_cuda,
        expected_gpu_count=args.expected_gpus,
        check_dataset_files=args.check_data,
        data_dir=args.data_dir,
        stack=args.stack,
        require_training_files=args.require_training,
    )
    if not res["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
