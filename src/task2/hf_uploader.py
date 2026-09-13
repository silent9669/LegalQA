"""Hugging Face Hub artifact and model release uploader for LegalQA Task 2."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.common.env_loader import load_environment
from src.common.security import assert_no_secrets_in_workspace
from src.task2.provenance.checksums import write_checksums_file

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "dangphuc2109/legalqa-qwen2.5-3b-adapter"


def upload_directory_to_hf(
    repo_id: str,
    folder_path: str | Path,
    path_in_repo: Optional[str] = None,
    private: bool = False,  # Repo policy is public; the only secret is HF_TOKEN itself (never committed).
    repo_type: str = "model",
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload local directory to Hugging Face Hub with strict pre-upload secret scanning."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Upload directory does not exist: {folder}")

    # 1. Preflight secret scan - fail-closed on any leaked credentials
    assert_no_secrets_in_workspace(folder, exclude_tests=False)

    # 2. Ensure environment credentials
    if not token:
        load_environment()
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        raise ValueError(
            "HF_TOKEN is required for uploading artifacts to Hugging Face. "
            "Please provide HF_TOKEN in your .env file or environment."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to upload models. Install with: pip install huggingface_hub"
        ) from e

    api = HfApi(token=token)

    # 3. Create repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )
        logger.info(f"Hugging Face repository verified: {repo_id} ({repo_type})")
    except Exception as e:
        logger.warning(f"Notice during create_repo for {repo_id}: {e}")

    # 4. Upload directory
    msg = commit_message or f"Upload LegalQA artifacts from {folder.name}"
    print(f"Uploading artifacts from {folder} to Hugging Face: {repo_id} (path_in_repo={path_in_repo})...")

    commit_info = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=msg,
        ignore_patterns=["*.pyc", "__pycache__", ".git*"],
    )

    commit_sha = getattr(commit_info, "oid", getattr(commit_info, "commit_id", str(commit_info)))
    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"[SUCCESS] Upload complete! Commit SHA: {commit_sha} -> {repo_url}")

    return {
        "status": "SUCCESS",
        "repo_id": repo_id,
        "repo_url": repo_url,
        "repo_type": repo_type,
        "commit_sha": str(commit_sha),
        "commit_info": str(commit_info),
        "source_folder": str(folder),
        "path_in_repo": path_in_repo,
    }


def upload_run_bundle_to_hf(
    bundle_dir: Union[Path, str],
    repo_id: str = DEFAULT_HF_REPO,
    run_id: Optional[str] = None,
    private: bool = False,  # Repo policy is public; the only secret is HF_TOKEN itself (never committed).
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload an immutable audited run bundle under runs/<run_id>/ in Hugging Face Hub."""
    b_path = Path(bundle_dir)
    if not b_path.is_dir():
        raise FileNotFoundError(f"Run bundle directory not found: {b_path}")

    # Infer run_id
    effective_run_id = run_id
    manifest_file = b_path / "production_run_manifest.json"
    if manifest_file.is_file():
        try:
            m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            effective_run_id = effective_run_id or m_data.get("run_id")
        except Exception:
            pass

    effective_run_id = effective_run_id or b_path.name
    path_in_repo = f"runs/{effective_run_id}"

    # Upload to runs/<run_id>/
    upload_res = upload_directory_to_hf(
        repo_id=repo_id,
        folder_path=b_path,
        path_in_repo=path_in_repo,
        private=private,
        commit_message=f"release(task2): publish audited run bundle {effective_run_id}",
        token=token,
    )

    # Record returned HF commit SHA into production_run_manifest.json
    commit_sha = upload_res.get("commit_sha")
    if manifest_file.is_file() and commit_sha:
        try:
            m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            m_data.setdefault("huggingface", {})["commit_sha"] = commit_sha
            m_data["huggingface"]["repository"] = repo_id
            m_data["huggingface"]["path_in_repo"] = path_in_repo
            manifest_file.write_text(json.dumps(m_data, indent=2, ensure_ascii=False), encoding="utf-8")
            # Update checksums
            write_checksums_file(b_path, b_path / "checksums.sha256")
        except Exception as e:
            logger.warning(f"Failed to update manifest with commit SHA: {e}")

    return upload_res
