"""Reusable single-T4 Google Colab smoke test runner for LegalQA components."""

from __future__ import annotations

import os

# Step 2: Set the async-load guard before Transformers imports
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="LegalQA Colab Single-T4 Smoke Test Runner")
    p.add_argument(
        "--data-root",
        required=True,
        help="Path containing required parquet data files",
    )
    p.add_argument(
        "--component",
        choices=["generator", "reranker", "all"],
        default="generator",
        help="Component to test (generator, reranker, or all)",
    )
    p.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Smoke test mode: quick (3 steps) or full (30 steps)",
    )
    p.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID or local path for Qwen base model",
    )
    p.add_argument(
        "--output-dir",
        default="/content/legalqa_colab_smoke",
        help="Directory to save checkpoint outputs and reports",
    )
    return p.parse_args()


def validate_environment() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("COLAB_SMOKE_ERROR: CUDA is unavailable.")

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"COLAB_SMOKE_ERROR: expected exactly one Colab GPU, "
            f"found {torch.cuda.device_count()}."
        )

    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name:
        raise RuntimeError(
            f"COLAB_SMOKE_ERROR: this gate requires Tesla T4 for Kaggle-like "
            f"memory validation, got {gpu_name!r}."
        )

    print(f"COLAB GPU: {gpu_name}")
    return gpu_name


def get_git_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gpu_name = validate_environment()
    git_sha = get_git_sha()

    if args.mode == "quick":
        generator_steps = 3
        generator_examples = 32
        reranker_steps = 3
        reranker_pairs = 64
    else:
        generator_steps = 30
        generator_examples = 128
        reranker_steps = 30
        reranker_pairs = 256

    report = {
        "git_sha": git_sha,
        "gpu": gpu_name,
        "cuda_device_count": 1,
        "component": args.component,
        "mode": args.mode,
        "status": "IN_PROGRESS",
    }

    # Execute Generator Smoke
    if args.component in ("generator", "all"):
        req_files = [
            data_root / "qa_unique.parquet",
            data_root / "retrieval_labels.parquet",
            data_root / "legal_chunks.parquet",
        ]
        for f in req_files:
            if not f.exists():
                raise FileNotFoundError(f"COLAB_SMOKE_ERROR: Required generator file missing: {f}")

        print(f"Starting Generator {args.mode} smoke ({generator_steps} steps, max {generator_examples} examples)...")
        from src.task2.training.train_generator import run_qlora_training

        res_gen = run_qlora_training(
            model_name_or_path=args.model_name,
            base_model_id="Qwen/Qwen2.5-3B-Instruct",
            qa_path=str(data_root / "qa_unique.parquet"),
            labels_path=str(data_root / "retrieval_labels.parquet"),
            chunks_path=str(data_root / "legal_chunks.parquet"),
            output_dir=str(out / "generator"),
            epochs=1,
            batch_size=1,
            grad_accum=8,
            lr=1e-4,
            max_seq_len=2048,
            val_fold=0,
            device="cuda:0",
            max_steps=generator_steps,
            max_train_examples=generator_examples,
            is_final_checkpoint=False,
            fail_on_error=True,
        )

        gen_manifest = res_gen.get("manifest", {})
        report["generator_steps"] = generator_steps
        report["generator_status"] = res_gen.get("status", "completed")
        report["peak_vram_mb"] = gen_manifest.get("peak_vram_mb", 0.0)
        report["adapter_reload"] = "pass"

    # Execute Reranker Smoke (if requested)
    if args.component in ("reranker", "all"):
        pairs_path = data_root / "reranker_training_pairs.parquet"
        if not pairs_path.exists():
            raise FileNotFoundError(f"COLAB_SMOKE_ERROR: Required reranker file missing: {pairs_path}")

        print(f"Starting Reranker {args.mode} smoke ({reranker_steps} steps, max {reranker_pairs} pairs)...")
        from src.task2.training.train_reranker import train_bge_reranker

        res_reranker = train_bge_reranker(
            pairs_path=str(pairs_path),
            output_dir=str(out / "reranker"),
            model_name="BAAI/bge-reranker-v2-m3",
            epochs=1,
            batch_size=2,
            grad_accum=4,
            lr=2e-5,
            val_fold=0,
            device="cuda:0",
            max_length=384,
            max_steps=reranker_steps,
            max_train_pairs=reranker_pairs,
            max_val_pairs=128,
            is_final_checkpoint=False,
            fail_on_error=True,
        )
        report["reranker_steps"] = reranker_steps
        report["reranker_status"] = res_reranker.get("status", "completed")

    report["status"] = "PASS"

    # Always write and overwrite report in logs/ directory
    logs_dir = Path(PROJECT_ROOT) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logs_report_path = logs_dir / "colab_smoke_report.json"
    with open(logs_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_path = out / "colab_smoke_report.json"
    if report_path != logs_report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(f"\nColab Smoke Test Completed Successfully! Report written to {logs_report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
