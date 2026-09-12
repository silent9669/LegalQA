"""Hugging Face Hub artifact and model release uploader for LegalQA Task 2."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.env_loader import load_environment

logger = logging.getLogger(__name__)


def upload_directory_to_hf(
    repo_id: str,
    folder_path: str | Path,
    path_in_repo: Optional[str] = None,
    private: bool = True,
    repo_type: str = "model",
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload local directory (adapter, checkpoints, logs, evidence bundle) to Hugging Face Hub."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Upload directory does not exist: {folder}")

    # Ensure environment is loaded
    if not token:
        load_environment()
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        raise ValueError(
            "HF_TOKEN is required for uploading artifacts to Hugging Face. "
            "Please provide HF_TOKEN in your .env file or Colab environment."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to upload models. Install with: pip install huggingface_hub"
        ) from e

    api = HfApi(token=token)

    # 1. Create repository if it doesn't exist
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

    # 2. Upload directory
    msg = commit_message or f"Upload LegalQA artifacts from {folder.name}"
    print(f"Uploading artifacts from {folder} to Hugging Face: {repo_id}...")

    commit_info = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=msg,
        ignore_patterns=["*.pyc", "__pycache__", ".git*"],
    )

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"[SUCCESS] Upload complete! Artifacts published to: {repo_url}")

    return {
        "status": "SUCCESS",
        "repo_id": repo_id,
        "repo_url": repo_url,
        "repo_type": repo_type,
        "commit_info": str(commit_info),
        "source_folder": str(folder),
    }
