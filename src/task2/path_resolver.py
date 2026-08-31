"""Deterministic runtime and dataset path resolver for Kaggle and local environments."""

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
    """Deterministically find and validate Qwen model directory by inspecting config.json."""
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

    # Filter by 3B if specified
    b3_candidates = [d for d in candidate_dirs if "3b" in d.lower() or "3B" in d]
    if len(b3_candidates) == 1:
        return b3_candidates[0]

    return candidate_dirs[0]


def resolve_runtime_paths(base_input_dir: str = "/kaggle/input") -> Dict[str, str]:
    """Resolve all runtime artifact paths deterministically."""
    roots = find_runtime_roots(base_input_dir)

    if len(roots) == 1:
        runtime_root = roots[0]
    elif len(roots) > 1:
        # Prefer the one with dataset_manifest.json
        manifest_roots = [r for r in roots if os.path.exists(os.path.join(r, "dataset_manifest.json"))]
        runtime_root = manifest_roots[0] if manifest_roots else roots[0]
    else:
        # Fallback to local workspace artifacts
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

    # Qwen model dir
    qwen_dir = find_qwen_model_dir(base_input_dir) or "Qwen/Qwen2.5-3B-Instruct"

    return {
        "runtime_root": runtime_root,
        "data_dir": data_dir,
        "bm25_dir": bm25_dir,
        "dek21_dir": dek21_dir,
        "qwen_model_path": qwen_dir,
    }
