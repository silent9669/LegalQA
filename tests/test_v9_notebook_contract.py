"""Tests for LegalQA V9 canonical notebook strictness contract."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


def notebook_source() -> str:
    nb = json.loads(Path("kaggle_kernel/legalqa_gpu_pipeline.ipynb").read_text())
    parts = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        parts.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(parts)


def test_notebook_calls_strict_packaged_code_resolver():
    """Verify notebook resolves packaged code root with strict=True."""
    src = notebook_source()
    assert re.search(
        r"resolve_packaged_code_root\(\s*[\"']/kaggle/input[\"']\s*,\s*strict=True\s*\)",
        src,
        re.DOTALL,
    )


def test_notebook_runtime_paths_are_unconditionally_strict():
    """Verify notebook resolve_runtime_paths is unconditionally strict with no remote download."""
    src = notebook_source()
    assert "strict=True if" not in src
    assert "allow_remote_model_download=False if" not in src
    assert re.search(
        r"resolve_runtime_paths\(\s*[\"']/kaggle/input[\"']\s*,\s*strict=True\s*,\s*allow_remote_model_download=False\s*,?\s*\)",
        src,
        re.DOTALL,
    )


def test_notebook_has_no_code_root_development_fallback():
    """Verify notebook Cell 3 contains no local development fallback for code root."""
    src = notebook_source()
    cell3 = src[src.index("# Cell 3"):src.index("# Cell 4")]
    assert 'resolved_code_root = os.path.abspath(".")' not in cell3
    assert 'resolved_code_root = os.path.abspath(' not in cell3


def test_notebook_manifest_validation_is_not_guarded_by_exists():
    """Verify validate_runtime_manifests is called unconditionally without caller-side exists guards."""
    src = notebook_source()
    cell3 = src[src.index("# Cell 3"):src.index("# Cell 4")]
    assert "validate_runtime_manifests(" in cell3
    assert (
        'if os.path.exists(os.path.join(paths["runtime_root"], "dataset_manifest.json"))'
        not in cell3
    )
