import os
import json
import hashlib
from typing import Dict, Any, Optional

def compute_freeze_tuple_hash(
    dataset_manifest_sha: str,
    git_commit_sha: str,
    config_hash: str,
    base_model_revision: str = "main",
    seed: int = 42,
) -> str:
    """
    Computes deterministic SHA256 binding the execution freeze tuple:
    RUN_TUPLE: dataset_hash + git_sha + config_hash + base_model_rev + seed
    """
    elements = [
        str(dataset_manifest_sha).strip().lower(),
        str(git_commit_sha).strip().lower(),
        str(config_hash).strip().lower(),
        str(base_model_revision).strip(),
        str(seed),
    ]
    raw = "::".join(elements)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def verify_smoke_pass(smoke_report_path: str, expected_profile: Optional[str] = None) -> bool:
    """
    Verifies that the Kaggle Dual-T4 smoke report exists and marked status=PASS.
    """
    if not os.path.exists(smoke_report_path):
        return False
    try:
        with open(smoke_report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        status = str(data.get("status", "")).upper()
        if status != "PASS":
            return False
        if expected_profile and data.get("profile") != expected_profile:
            return False
        return True
    except Exception:
        return False

def build_run_manifest(
    run_id: str,
    profile_name: str,
    dataset_manifest_sha: str,
    git_commit_sha: str,
    config_hash: str,
    hardware_info: Dict[str, Any],
    metrics: Dict[str, Any],
    hf_repo: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds the authoritative run_manifest.json for competition audit and Hugging Face export.
    """
    freeze_hash = compute_freeze_tuple_hash(
        dataset_manifest_sha=dataset_manifest_sha,
        git_commit_sha=git_commit_sha,
        config_hash=config_hash,
    )
    return {
        "run_id": run_id,
        "profile": profile_name,
        "freeze_tuple_hash": freeze_hash,
        "git_commit_sha": git_commit_sha,
        "dataset_manifest_sha": dataset_manifest_sha,
        "config_hash": config_hash,
        "hardware": hardware_info,
        "metrics": metrics,
        "huggingface_repo": hf_repo,
    }
