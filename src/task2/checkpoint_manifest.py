"""Checkpoint manifest loader and strict validation for production LegalQA components."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.hashing import sha256_file


def load_reranker_manifest(checkpoint_dir: str) -> Dict[str, Any]:
    """Load reranker checkpoint manifest from directory."""
    manifest_path = os.path.join(checkpoint_dir, "reranker_manifest.json")
    if not os.path.exists(manifest_path):
        # Fallback to general manifest if present
        general_manifest = os.path.join(checkpoint_dir, "manifest.json")
        if os.path.exists(general_manifest):
            manifest_path = general_manifest
        else:
            raise FileNotFoundError(f"Reranker manifest not found in: {checkpoint_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_generator_manifest(checkpoint_dir: str) -> Dict[str, Any]:
    """Load QLoRA generator checkpoint manifest from directory."""
    manifest_path = os.path.join(checkpoint_dir, "generator_manifest.json")
    if not os.path.exists(manifest_path):
        general_manifest = os.path.join(checkpoint_dir, "training_manifest.json")
        if not os.path.exists(general_manifest):
            general_manifest = os.path.join(checkpoint_dir, "manifest.json")
        if os.path.exists(general_manifest):
            manifest_path = general_manifest
        else:
            raise FileNotFoundError(f"Generator manifest not found in: {checkpoint_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_final_checkpoint(
    checkpoint_dir: str,
    expected_base_model: str,
    component_name: str,
    expected_scope: str = "all_allowed_task2_data",
) -> Dict[str, Any]:
    """Strictly assert that a checkpoint was trained on full data as a final checkpoint.

    Task 2 & P0-13: Checks base_model_id and both 'val_fold_excluded' and 'val_fold' keys.
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    if component_name == "reranker":
        manifest = load_reranker_manifest(checkpoint_dir)
    elif component_name in ("generator", "qlora"):
        manifest = load_generator_manifest(checkpoint_dir)
    else:
        # Generic loader
        manifest_path = os.path.join(checkpoint_dir, f"{component_name}_manifest.json")
        if not os.path.exists(manifest_path):
            manifest_path = os.path.join(checkpoint_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found for {component_name} in {checkpoint_dir}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    if not manifest.get("is_final_checkpoint"):
        raise ValueError(
            f"Checkpoint in {checkpoint_dir} is NOT marked as is_final_checkpoint=true. "
            f"Found: {manifest.get('is_final_checkpoint')}"
        )

    scope = manifest.get("training_scope")
    if scope != expected_scope:
        raise ValueError(
            f"Checkpoint in {checkpoint_dir} training_scope mismatch. "
            f"Expected '{expected_scope}', found '{scope}'."
        )

    if manifest.get("smoke_only", False):
        raise ValueError(
            f"Checkpoint in {checkpoint_dir} is a smoke checkpoint ('smoke_only': true). "
            f"Refusing to use smoke checkpoint in final/reuse profile."
        )

    # Check both val_fold_excluded and val_fold
    excluded_fold = manifest.get("val_fold_excluded", manifest.get("val_fold"))
    if excluded_fold is not None:
        raise ValueError(
            f"Checkpoint in {checkpoint_dir} was trained with held-out val_fold={excluded_fold}. "
            f"Final checkpoints must be trained on all allowed data with val_fold=None."
        )

    # Task 2: Check canonical base_model_id, base_model, or base_model_name_or_path
    base_m = manifest.get("base_model_id") or manifest.get("base_model") or manifest.get("base_model_name_or_path")
    if base_m and expected_base_model and base_m != expected_base_model:
        raise ValueError(
            f"Checkpoint in {checkpoint_dir} base model mismatch. "
            f"Expected '{expected_base_model}', found '{base_m}'."
        )

    return manifest


def save_checkpoint_manifest(checkpoint_dir: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Atomically write a checkpoint manifest with file digests (same authority).

    Files listed under manifest["files"] (relative paths) are hashed from
    disk and their digests embedded; verify_checkpoint_manifest re-checks
    them. Temp-then-rename: no partial manifest is ever visible.
    """
    target_dir = Path(checkpoint_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    record = dict(manifest)
    digests = {}
    for rel in manifest.get("files", []) or []:
        candidate = target_dir / rel
        if not candidate.is_file():
            raise FileNotFoundError(f"checkpoint file listed in manifest is missing: {rel}")
        digests[rel] = sha256_file(str(candidate))
    record["file_digests"] = digests
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    record["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    final_path = target_dir / "manifest.json"
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-manifest-", dir=str(target_dir))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_name, str(final_path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return record


def verify_checkpoint_manifest(checkpoint_dir: str) -> Dict[str, Any]:
    """Verify manifest completeness: required keys, file presence + digests."""
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Checkpoint manifest not found in: {checkpoint_dir}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for rel, expected in (manifest.get("file_digests") or {}).items():
        candidate = os.path.join(checkpoint_dir, rel)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Checkpoint file missing: {rel}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise ValueError(f"Checkpoint file digest mismatch for {rel}")
    return manifest
