#!/usr/bin/env python3
"""
Canonical unified pipeline runner for LegalQA Task 2.
Supports both Kaggle Dual-T4 smoke testing and Google Colab A100 production training.
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path

# Safe CUDA allocation and model loading defaults
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.env_loader import load_environment
from src.task2.hf_uploader import upload_directory_to_hf
from src.task2.pipeline.profiles import load_profile_from_yaml, resolve_execution_profile
from src.task2.pipeline.runner import run_pipeline
from src.task2.provenance.freeze_tuple import verify_smoke_pass, build_run_manifest
from src.task2.path_resolver import resolve_runtime_paths

def main():
    parser = argparse.ArgumentParser(description="LegalQA Task 2 Pipeline Execution Entrypoint")
    parser.add_argument("--config", default="configs/kaggle_smoke_t4.yaml", help="Path to profile configuration YAML")
    parser.add_argument("--data-dir", default=None, help="Explicit dataset directory path")
    parser.add_argument("--output-dir", default=None, help="Output directory for checkpoints and logs")
    parser.add_argument("--require-smoke-pass", default=None, help="Path to kaggle_smoke_report.json (required for A100)")
    parser.add_argument("--allow-single-gpu", action="store_true", help="Allow running on single GPU")
    parser.add_argument("--env-file", default=None, help="Path to .env credential file")
    parser.add_argument("--no-upload-to-hf", action="store_true", help="Skip automatic Hugging Face upload")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Automatically load environment variables and credentials
    env_info = load_environment(args.env_file)

    print(f"=== LegalQA Task 2 Pipeline Runner ===")
    print(f"Config: {args.config}")
    if env_info.get("loaded_from_file"):
        print(f"Loaded credentials from: {env_info['loaded_from_file']}")
    print(f"Hugging Face Auth: {'CONFIGURED (' + env_info['hf_token_masked'] + ')' if env_info['hf_token_configured'] else 'NOT CONFIGURED'}")
    print(f"Kaggle Auth: {'CONFIGURED (' + env_info['kaggle_user'] + ')' if env_info['kaggle_configured'] else 'NOT CONFIGURED'}")

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    profile = load_profile_from_yaml(args.config)
    print(f"Resolved Profile: {profile.name}")

    # Enforce smoke pass gate before production training
    if args.require_smoke_pass:
        print(f"Checking smoke gate verification: {args.require_smoke_pass}")
        if not verify_smoke_pass(args.require_smoke_pass):
            print(f"Error: Smoke gate validation FAILED. A100 production training refused.")
            sys.exit(1)
        print("Smoke gate PASS verified.")

    # Hardware devices
    import torch
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"CUDA GPUs Available: {gpu_count}")

    dev_cfg = cfg.get("devices", {})
    gen_device = dev_cfg.get("generator", "cuda:0" if gpu_count > 0 else "cpu")
    retrieval_device = dev_cfg.get("retrieval", "cuda:1" if gpu_count > 1 else ("cuda:0" if gpu_count > 0 else "cpu"))

    # If running on single GPU or explicit single-GPU mode requested, clamp to available devices
    if gpu_count <= 1 or args.allow_single_gpu:
        if gpu_count == 1:
            gen_device = "cuda:0"
            retrieval_device = "cuda:0"
        elif gpu_count == 0:
            gen_device = "cpu"
            retrieval_device = "cpu"
    print(f"Allocated Devices -> Generator: {gen_device} | Retrieval: {retrieval_device}")

    # Resolve paths
    base_data = args.data_dir or cfg.get("data", {}).get("runtime_root", "/kaggle/input")
    paths = resolve_runtime_paths(base_data, strict=False)

    if "public_test_path" not in paths:
        candidate_test = os.path.join(paths.get("data_dir", ""), "public-official.json")
        if not os.path.exists(candidate_test):
            candidate_test = os.path.join(paths.get("runtime_root", ""), "public-official.json")
        paths["public_test_path"] = candidate_test

    out_dir = args.output_dir or cfg.get("outputs", {}).get("report_path", "runs/current")
    if out_dir.endswith(".json"):
        out_dir = os.path.dirname(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Output Directory: {out_dir}")
    print(f"Executing profile '{profile.name}'...")

    outputs = run_pipeline(
        profile=profile,
        paths=paths,
        gen_device=gen_device,
        retrieval_device=retrieval_device,
        output_dir=out_dir,
        seed=args.seed,
        code_root=str(Path(__file__).resolve().parent.parent),
        allow_single_gpu=args.allow_single_gpu or (gpu_count < 2),
    )

    # Automatic Hugging Face upload if configured
    hf_cfg = cfg.get("huggingface")
    if hf_cfg and hf_cfg.get("repo_id") and not args.no_upload_to_hf:
        repo_id = hf_cfg["repo_id"]
        private = hf_cfg.get("private", True)
        print(f"\n=== Auto-Uploading Run Bundle to Hugging Face ===")
        print(f"Target Repo: {repo_id} (private={private})")
        try:
            from src.task2.hf_uploader import upload_run_bundle_to_hf, upload_directory_to_hf
            if os.path.exists(os.path.join(out_dir, "production_run_manifest.json")):
                upload_res = upload_run_bundle_to_hf(
                    bundle_dir=out_dir,
                    repo_id=repo_id,
                    private=private,
                )
            else:
                upload_res = upload_directory_to_hf(
                    repo_id=repo_id,
                    folder_path=out_dir,
                    path_in_repo=f"runs/{profile.name}",
                    private=private,
                    commit_message=f"feat(release): trained {profile.name} artifacts",
                )
            print(f"Hugging Face Upload: {upload_res.get('status')} -> {upload_res.get('repo_url')}")
        except Exception as e:
            print(f"Warning: Hugging Face upload failed: {e}", file=sys.stderr)

    print(f"\nExecution finished successfully for profile: {profile.name}")

if __name__ == "__main__":
    main()
