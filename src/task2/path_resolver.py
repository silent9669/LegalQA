"""Deterministic runtime and dataset path resolver for Kaggle and local environments (V8)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_runtime_roots(base_path: str = "/kaggle/input") -> List[str]:
    """Find dataset roots that match LegalQA dataset contents."""
    if not os.path.exists(base_path):
        return []

    matching_roots = []
    # Check top-level directory itself
    if _is_legalqa_root(base_path):
        matching_roots.append(base_path)

    # Check direct children subdirectories
    try:
        for entry in os.scandir(base_path):
            if entry.is_dir():
                if _is_legalqa_root(entry.path):
                    matching_roots.append(entry.path)
                else:
                    # Check 1 level deeper (e.g. /kaggle/input/dataset-name/LegalQA)
                    try:
                        for sub in os.scandir(entry.path):
                            if sub.is_dir() and _is_legalqa_root(sub.path):
                                matching_roots.append(sub.path)
                    except (PermissionError, OSError):
                        pass
    except (PermissionError, OSError):
        pass

    return sorted(list(set(matching_roots)))


def _is_legalqa_root(path: str) -> bool:
    """Check if directory contains core LegalQA markers."""
    has_manifest = os.path.exists(os.path.join(path, "dataset_manifest.json")) or os.path.exists(os.path.join(path, "code_manifest.json"))
    has_parquet = os.path.exists(os.path.join(path, "legal_chunks.parquet")) or os.path.exists(os.path.join(path, "data", "legal_chunks.parquet"))
    has_bm25 = os.path.exists(os.path.join(path, "indexes", "bm25")) or os.path.exists(os.path.join(path, "bm25"))
    return (has_manifest and has_parquet) or (has_parquet and has_bm25)


def find_qwen_model_dir(base_path: str = "/kaggle/input", expected_arch: str = "Qwen2ForCausalLM") -> Optional[str]:
    """Deterministically find and validate Qwen model directory by inspecting config.json.

    Hard-fails if multiple ambiguous Qwen model directories are found. Never returns candidate_dirs[0].
    """
    if not os.path.exists(base_path):
        return None

    candidate_dirs: List[str] = []
    for root, dirs, files in os.walk(base_path):
        if "config.json" in files:
            cfg_path = os.path.join(root, "config.json")
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                archs = cfg.get("architectures", [])
                model_type = cfg.get("model_type", "")
                if expected_arch in archs or model_type == "qwen2":
                    candidate_dirs.append(root)
            except Exception:
                continue

    if not candidate_dirs:
        return None

    if len(candidate_dirs) == 1:
        return candidate_dirs[0]

    # Filter by 3B if multiple causal LMs are present (e.g. Qwen2.5-3B-Instruct)
    b3_candidates = [d for d in candidate_dirs if "3b" in d.lower() or "3B" in d]
    if len(b3_candidates) == 1:
        return b3_candidates[0]

    # If still ambiguous, raise explicit RuntimeError instead of returning arbitrary first match
    raise RuntimeError(
        f"Ambiguous Qwen candidate model directories found under '{base_path}': {candidate_dirs}. "
        f"Expected exactly 1 matching Qwen 3B causal LM directory."
    )


def resolve_runtime_paths(
    base_input_dir: str = "/kaggle/input",
    strict: bool = False,
    allow_remote_model_download: bool = True,
) -> Dict[str, str]:
    """Resolve all runtime artifact paths deterministically (Task 3 & 8).

    When strict=True, fails immediately if dataset root or model paths are missing or ambiguous,
    refusing silent fallback to local artifacts directory.
    When allow_remote_model_download=False, fails if mounted Qwen model is not found under base_input_dir.
    """
    roots = find_runtime_roots(base_input_dir)

    if len(roots) == 1:
        runtime_root = roots[0]
    elif len(roots) > 1:
        raise RuntimeError(
            f"Ambiguous runtime dataset roots found under '{base_input_dir}': {roots}. "
            f"Expected exactly 1 matching LegalQA dataset root."
        )
    else:
        if strict:
            raise RuntimeError(
                f"No valid LegalQA dataset root found under '{base_input_dir}' in strict Kaggle mode. "
                f"Ensure the Kaggle dataset is mounted to the notebook session."
            )
        # Fallback to local workspace artifacts in development
        runtime_root = os.path.abspath("artifacts")

    print(f"Resolved primary Runtime Root: {runtime_root}")

    # Sub-paths resolution
    data_dir = os.path.join(runtime_root, "task2", "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(runtime_root, "data")
    if not os.path.exists(data_dir):
        data_dir = runtime_root

    indexes_dir = os.path.join(runtime_root, "task2", "indexes")
    if not os.path.exists(indexes_dir):
        indexes_dir = os.path.join(runtime_root, "indexes")
    if not os.path.exists(indexes_dir):
        indexes_dir = runtime_root

    bm25_dir = os.path.join(indexes_dir, "bm25")
    dek21_dir = os.path.join(indexes_dir, "dek21")

    # Qwen model dir resolution
    qwen_found = find_qwen_model_dir(base_input_dir)
    if qwen_found:
        qwen_dir = qwen_found
    elif allow_remote_model_download:
        qwen_dir = "Qwen/Qwen2.5-3B-Instruct"
    else:
        raise RuntimeError(
            f"Expected mounted Qwen2.5-3B model was not found under '{base_input_dir}'. "
            f"Please attach the Qwen2.5-3B-Instruct Kaggle model to your notebook session."
        )

    return {
        "runtime_root": runtime_root,
        "data_dir": data_dir,
        "bm25_dir": bm25_dir,
        "dek21_dir": dek21_dir,
        "qwen_model_path": qwen_dir,
    }
