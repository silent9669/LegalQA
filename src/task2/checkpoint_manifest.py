"""Checkpoint manifest loader and strict validation for production LegalQA components."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


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
    """Strictly assert that a checkpoint was trained on full data as a final checkpoint."""
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

    if manifest.get("val_fold") is not None:
        raise ValueError(
            f"Checkpoint in {checkpoint_dir} was trained with held-out val_fold={manifest.get('val_fold')}. "
            f"Final checkpoints must be trained on all allowed data with val_fold=None."
        )

    base_m = manifest.get("base_model") or manifest.get("base_model_name_or_path")
    if base_m and expected_base_model and base_m != expected_base_model:
        raise ValueError(
            f"Checkpoint in {checkpoint_dir} base model mismatch. "
            f"Expected '{expected_base_model}', found '{base_m}'."
        )

    return manifest
