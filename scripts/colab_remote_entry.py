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
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> str:
    """Run a shell command with real-time output streaming."""
    print(f"[*] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr}")
    return res.stdout.strip()


def main():
    print("=" * 65)
    print("      LegalQA Colab Remote Entrypoint — Stage Execution       ")
    print("=" * 65)

    request_file = BOOTSTRAP_DIR / "run_request.json"
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
            "transformers", "peft", "accelerate", "datasets", "trl", "liger-kernel", "bitsandbytes"
        ])

    # 4. Download Kaggle Dataset (Versioned)
    data_target = DATA_DIR / dataset_slug.split("/")[-1]
    data_target.mkdir(parents=True, exist_ok=True)
    print(f"[+] Downloading versioned dataset: {dataset_slug} version {dataset_version}...")
    try:
        import kagglehub
        downloaded_path = kagglehub.dataset_download(f"{dataset_slug}/{dataset_version}")
        print(f"  OK: Downloaded to {downloaded_path}")
        dataset_path = downloaded_path
    except Exception as e:
        print(f"  Notice: kagglehub versioned download fallback: {e}")
        dataset_path = str(data_target)

    # 5. Execute Stage via run_gpu_gate
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    candidate_manifest_path = BOOTSTRAP_DIR / "candidate_manifest.json"
    if not candidate_manifest_path.exists():
        candidate_manifest_path = LEGALQA_DIR / f"artifacts/candidates/{candidate_id}/candidate_manifest.json"

    parent_report = None
    if stage == "colab_t4":
        parent_report = BOOTSTRAP_DIR / "kaggle_t4x2_report.json"
    elif stage in ("a100", "colab_a100"):
        parent_report = BOOTSTRAP_DIR / "colab_t4_report.json"

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
    print(f"\n[PASS] Colab remote entry completed {stage} successfully: {report.status}")


if __name__ == "__main__":
    main()
