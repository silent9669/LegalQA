"""Tests for LegalQA V9 canonical notebook strictness contract."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


def notebook_cells() -> list[dict]:
    nb = json.loads(Path("kaggle_kernel/legalqa_gpu_pipeline.ipynb").read_text())
    return nb["cells"]


def notebook_source() -> str:
    parts = []
    for cell in notebook_cells():
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        parts.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(parts)


def test_notebook_disables_transformers_v5_async_model_loading():
    """Verify Cell 1 explicitly deactivates Transformers 5 async loading to prevent T4 4-bit OOM spikes."""
    cells = notebook_cells()
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    cell1_src = "".join(code_cells[0].get("source", ""))
    cell2_src = "".join(code_cells[1].get("source", ""))

    expected = 'os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"'
    assert expected in cell1_src
    assert 'os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "0"' not in notebook_source()
    assert 'HF_DEACTIVATE_ASYNC_LOAD"] = "0"' not in notebook_source()

    # Assert committed profile and release binding API 14 remain untouched
    src = notebook_source()
    assert 'EXECUTION_PROFILE = "screen_fold0"' in src
    assert "REQUIRED_RUNTIME_API_VERSION = 14" in src
    assert "REQUIRED_RUNTIME_API_VERSION = 12" not in src
    assert "REQUIRED_RUNTIME_API_VERSION = 13" not in src


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
