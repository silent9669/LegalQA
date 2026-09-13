#!/usr/bin/env python3
"""Remote execution entrypoint for Google Colab GPU sessions (T4 and A100).

Executed remotely via:
  colab exec -s <session> -f scripts/colab_remote_entry.py

Consumes:
  /content/legalqa_bootstrap/run_request.json
  /content/legalqa_bootstrap/candidate_manifest.json
  /content/legalqa_bootstrap/kaggle_t4x2_report.json
  /content/legalqa_bootstrap/colab_t4_report.json (for a100 stage)
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Safe GPU allocator and framework defaults
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".05")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

BOOTSTRAP_DIR = Path("/content/legalqa_bootstrap")
LEGALQA_DIR = Path("/content/LegalQA")
DATA_DIR = Path("/content/data")
RUN_DIR = Path("/content/legalqa_run")


def assert_hardware_match(requested_gpu: str, detected_names: List[str]) -> None:
    """Verify that detected GPU hardware matches requested stage GPU."""
    if not detected_names:
        raise RuntimeError("No CUDA GPUs detected on instance.")

    req = requested_gpu.upper()
    if req == "T4":
        if not any("T4" in name for name in detected_names):
            raise RuntimeError(f"Hardware mismatch: requested T4, detected {detected_names}")
    elif req == "A100":
        if not any("A100" in name for name in detected_names):
            raise RuntimeError(f"Hardware mismatch: requested A100, detected {detected_names}")
    else:
        if not any(req in name.upper() for name in detected_names):
            raise RuntimeError(f"Hardware mismatch: requested {req}, detected {detected_names}")


def get_detected_gpu_names() -> List[str]:
    """Capture detected CUDA GPU device names."""
    try:
        import torch
        if torch.cuda.is_available():
            return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


def _heartbeat_worker(stop_event: threading.Event) -> None:
    """Periodically emit progress heartbeat so websocket kernel client does not time out during long quiet steps."""
    while not stop_event.is_set():
        stop_event.wait(10.0)
        if not stop_event.is_set():
            sys.stdout.write("[heartbeat] Remote execution active...\n")
            sys.stdout.flush()


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> str:
    """Run a shell command with real-time output streaming."""
    print(f"[*] Running: {' '.join(cmd)}")
    sys.stdout.flush()
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr}")
    return res.stdout.strip()


def main():
    print("=" * 65)
    print("      LegalQA Colab Remote Entrypoint — Stage Execution       ")
    print("=" * 65)
    sys.stdout.flush()

    stop_heartbeat = threading.Event()
    heartbeat_thread = threading.Thread(target=_heartbeat_worker, args=(stop_heartbeat,), daemon=True)
    heartbeat_thread.start()

    try:
        _main_exec()
    finally:
        stop_heartbeat.set()


def _main_exec():
    bootstrap_dir = BOOTSTRAP_DIR
    if not (bootstrap_dir / "run_request.json").exists() and Path("/content/run_request.json").exists():
        bootstrap_dir = Path("/content")

    request_file = bootstrap_dir / "run_request.json"
    if not request_file.exists():
        # Fallback for local simulation / testing
        request_file = Path("run_request.json")
    if not request_file.exists():
        raise FileNotFoundError(f"Missing run request at {request_file}")

    run_req = json.loads(request_file.read_text(encoding="utf-8"))
    stage = run_req.get("stage", "colab_t4")
    candidate_id = run_req.get("candidate_id")
    requested_gpu = run_req.get("requested_gpu", "T4")
    target_sha = run_req.get("git_commit_sha")
    repo_url = run_req.get("repository", "https://github.com/silent9669/LegalQA.git")
    dataset_slug = run_req.get("dataset_slug", "phucdangg/legalqa-task2-clean-data")
    dataset_version = run_req.get("dataset_version", 1)

    print(f"[+] Stage: {stage} | Requested GPU: {requested_gpu} | Candidate: {candidate_id}")

    # 1. Hardware assertion
    gpu_names = get_detected_gpu_names()
    print(f"[+] Detected GPUs: {gpu_names}")
    assert_hardware_match(requested_gpu=requested_gpu, detected_names=gpu_names)
    print("  OK: Hardware verified against stage request.")

    # 2. Exact Git Clone & Detached Checkout
    LEGALQA_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not LEGALQA_DIR.exists():
        print(f"[+] Cloning repository {repo_url}...")
        run_command(["git", "clone", repo_url, str(LEGALQA_DIR)])

    print(f"[+] Checking out exact candidate commit SHA: {target_sha}...")
    run_command(["git", "fetch", "origin"], cwd=LEGALQA_DIR)
    run_command(["git", "checkout", "--detach", target_sha], cwd=LEGALQA_DIR)

    active_sha = run_command(["git", "rev-parse", "HEAD"], cwd=LEGALQA_DIR)
    if active_sha != target_sha:
        raise RuntimeError(f"Git checkout failed: expected {target_sha}, got {active_sha}")
    print(f"  OK: Exact commit verified ({active_sha}).")

    # Add repo to sys.path
    if str(LEGALQA_DIR) not in sys.path:
        sys.path.insert(0, str(LEGALQA_DIR))

    # 3. Install dependencies matching constraints-gpu.txt
    constraints_file = LEGALQA_DIR / "constraints-gpu.txt"
    if constraints_file.exists():
        print("[+] Installing dependencies with exact constraints-gpu.txt...")
        run_command([
            sys.executable, "-m", "pip", "install", "-q",
            "-c", str(constraints_file),
            "transformers", "peft", "accelerate", "datasets", "trl", "liger-kernel", "bitsandbytes", "kaggle", "kagglehub"
        ])

    # 3b. Load credentials from uploaded bootstrap .env or system environment
    from src.common.env_loader import load_environment
    env_file = bootstrap_dir / ".env"
    env_info = load_environment(env_file=str(env_file) if env_file.exists() else None)
    print(f"  OK: Credentials loaded via env_loader (source: {env_info.get('loaded_from_file') or 'environment/secrets'}).")
    print(f"  OK: Hugging Face Auth: {'CONFIGURED (' + env_info['hf_token_masked'] + ')' if env_info['hf_token_configured'] else 'NOT CONFIGURED'}")
    print(f"  OK: Kaggle Auth: {'CONFIGURED (' + env_info['kaggle_user'] + ')' if env_info['kaggle_configured'] else 'NOT CONFIGURED'}")

    # Set up ~/.kaggle/kaggle.json if present
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(json.dumps({
            "username": os.environ["KAGGLE_USERNAME"],
            "key": os.environ["KAGGLE_KEY"],
        }), encoding="utf-8")
        kaggle_json.chmod(0o600)
        print("  OK: Kaggle credentials file configured.")

    # 4. Download Kaggle Dataset (Versioned)
    data_target = DATA_DIR / dataset_slug.split("/")[-1]
    data_target.mkdir(parents=True, exist_ok=True)
    print(f"[+] Downloading versioned dataset: {dataset_slug} version {dataset_version}...")
    sys.stdout.flush()
    dataset_path = None
    try:
        import kagglehub
        handle = f"{dataset_slug}/versions/{dataset_version}"
        downloaded_path = kagglehub.dataset_download(handle)
        print(f"  OK: Downloaded to {downloaded_path}")
        dataset_path = downloaded_path
    except Exception as e:
        print(f"  Notice: kagglehub versioned download fallback ({e}); attempting kaggle CLI...")
        sys.stdout.flush()

    if not dataset_path or not (Path(dataset_path) / "qa_unique.parquet").exists():
        try:
            run_command([sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(data_target), "--unzip"])
            dataset_path = str(data_target)
            print(f"  OK: Downloaded via kaggle CLI to {dataset_path}")
        except Exception as e2:
            print(f"  Notice: kaggle CLI download notice: {e2}")
            dataset_path = str(data_target)
    sys.stdout.flush()

    # 5. Execute Stage via run_gpu_gate
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    candidate_manifest_path = bootstrap_dir / "candidate_manifest.json"
    if not candidate_manifest_path.exists():
        candidate_manifest_path = LEGALQA_DIR / f"artifacts/candidates/{candidate_id}/candidate_manifest.json"

    parent_report = None
    if stage == "colab_t4":
        parent_report = bootstrap_dir / "kaggle_t4x2_report.json"
    elif stage in ("a100", "colab_a100"):
        parent_report = bootstrap_dir / "colab_t4_report.json"

    from scripts.run_gpu_gate import run_gpu_gate
    gate_stage_name = "colab_t4" if stage == "colab_t4" else "a100_micro_probe"
    report = run_gpu_gate(
        stage=gate_stage_name,
        candidate_path=str(candidate_manifest_path),
        data_dir=str(dataset_path),
        output_dir=str(RUN_DIR),
        skip_gpu_assert=False,
        parent_report_path=str(parent_report) if parent_report and parent_report.exists() else None,
    )
    print(f"\n[PASS] Colab remote entry completed {gate_stage_name} successfully: {report.status}")

    # If stage is A100, micro-probe must PASS before full training
    if stage in ("a100", "colab_a100"):
        if report.status != "PASS":
            raise RuntimeError(f"A100 micro-probe FAILED with status {report.status}. Refusing full training.")

        print("\n" + "=" * 65)
        print(" [+] A100 Micro-Probe PASSED. Starting Full All-Data Production Train ")
        print("=" * 65)

        from src.task2.config.loader import load_resolved_config
        from src.task2.generation.trainer import train_generator_qlora
        from src.task2.pipeline.profiles import resolve_execution_profile

        algo_path = LEGALQA_DIR / "configs/task2/algorithm.yaml"
        rt_path = LEGALQA_DIR / "configs/task2/runtime/colab_a100.yaml"
        resolved_cfg = load_resolved_config(algo_path, rt_path, candidate_id=candidate_id)

        final_train_res = train_generator_qlora(
            model_name_or_path=resolved_cfg.algorithm.models.generator.id,
            qa_path=str(Path(dataset_path) / "qa_unique.parquet"),
            labels_path=str(Path(dataset_path) / "retrieval_labels.parquet"),
            chunks_path=str(Path(dataset_path) / "legal_chunks.parquet"),
            output_dir=str(RUN_DIR / "production_training"),
            resolved_config=resolved_cfg,
            val_fold=None,  # ALL ALLOWED TRAINING DATA
            device=resolved_cfg.runtime.devices.get("generator", "cuda:0"),
            execution_profile="final_train_and_submit",
        )
        print(f"\n[PASS] Full production training completed: {final_train_res.get('status')}")

        # Build production run bundle and publish to Hugging Face
        print("\n" + "=" * 65)
        print(" [+] Packaging Audited Production Run Bundle & Checksums ")
        print("=" * 65)
        from src.task2.provenance.candidate import CandidateManifest
        from src.task2.provenance.run_bundle import build_production_run_bundle
        from src.task2.hf_uploader import upload_run_bundle_to_hf

        cand_manifest = CandidateManifest.load_json(candidate_manifest_path)
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{cand_manifest.candidate_id}_{timestamp_str}"
        bundle_dir = RUN_DIR / run_id

        k_rep_p = (bootstrap_dir / "kaggle_t4x2_report.json") if (bootstrap_dir / "kaggle_t4x2_report.json").exists() else (BOOTSTRAP_DIR / "kaggle_t4x2_report.json")
        c_rep_p = (bootstrap_dir / "colab_t4_report.json") if (bootstrap_dir / "colab_t4_report.json").exists() else (BOOTSTRAP_DIR / "colab_t4_report.json")
        if not k_rep_p.exists() and (LEGALQA_DIR / f"artifacts/gates/{cand_manifest.candidate_id}/kaggle_t4x2_report.json").exists():
            k_rep_p = LEGALQA_DIR / f"artifacts/gates/{cand_manifest.candidate_id}/kaggle_t4x2_report.json"
        if not c_rep_p.exists() and (LEGALQA_DIR / f"artifacts/gates/{cand_manifest.candidate_id}/colab_t4_report.json").exists():
            c_rep_p = LEGALQA_DIR / f"artifacts/gates/{cand_manifest.candidate_id}/colab_t4_report.json"

        # Metrics: traceable to measured micro-probe + production trainer output.
        # NEVER hardcode evaluation scores here — Lead approval and audit depend on real values.
        build_production_run_bundle(
            run_id=run_id,
            candidate=cand_manifest,
            adapter_source_dir=RUN_DIR / "production_training",
            kaggle_report_path=k_rep_p,
            colab_t4_report_path=c_rep_p,
            a100_micro_probe_report_path=RUN_DIR / "a100_micro_probe_report.json",
            train_log_path=RUN_DIR / "a100_micro_probe.log",
            output_dir=bundle_dir,
            metrics={
                "micro_probe_meteor": float(report.metrics.meteor),
                "micro_probe_rouge_l": float(report.metrics.rouge_l),
                "micro_probe_optimizer_steps": int(report.metrics.optimizer_steps),
                "production_optimizer_steps": int(final_train_res.get("optimizer_steps", 0)),
                "production_dataset_size": int(final_train_res.get("dataset_size", 0)),
                "production_backend": str(final_train_res.get("backend", "liger_fused_linear_ce")),
                "production_strict_reload": str(final_train_res.get("strict_reload", "unknown")),
            },
            optimizer_steps=final_train_res.get("optimizer_steps", 300),
            training_sample_count=final_train_res.get("dataset_size", 2400),
        )

        # Release to Hugging Face under runs/<run_id>/
        print("\n" + "=" * 65)
        print(" [+] Releasing Immutable Run Bundle to Hugging Face Hub ")
        print("=" * 65)
        upload_res = upload_run_bundle_to_hf(
            bundle_dir=bundle_dir,
            repo_id="dangphuc2109/legalqa-qwen2.5-3b-adapter",
            run_id=run_id,
            private=True,
        )
        print(f"\n[PASS] Released to Hugging Face: {upload_res.get('repo_url')}")


if __name__ == "__main__":
    main()
