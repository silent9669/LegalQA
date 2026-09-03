"""Tests for LegalQA V10/V11/V12/V13 runtime release binding, literal API 13, and pre-resolution validation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from src.task2.path_resolver import (
    find_qwen_model_dir,
    find_runtime_roots,
    resolve_runtime_paths,
)
from src.task2.runtime_integrity import (
    EXPECTED_RUNTIME_API_VERSION,
    resolve_packaged_code_root,
    validate_runtime_manifests,
)

VALID_SHA = "e37e04df3aa78df3170954c64039c1a2c9109317"


def notebook_cell_3_source() -> str:
    nb = json.loads(Path("kaggle_kernel/legalqa_gpu_pipeline.ipynb").read_text())
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", "")) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        if "# Cell 3" in src:
            return src
    raise ValueError("Cell 3 not found in notebook")


def test_notebook_owns_literal_required_runtime_api_version_15():
    """Item 1: Verify notebook owns literal REQUIRED_RUNTIME_API_VERSION = 15 and does not import it."""
    src = notebook_cell_3_source()
    assert re.search(r"REQUIRED_RUNTIME_API_VERSION\s*=\s*15\b", src), "Notebook must define REQUIRED_RUNTIME_API_VERSION = 15"
    assert "EXPECTED_RUNTIME_API_VERSION" not in src, "Notebook must not import or derive EXPECTED_RUNTIME_API_VERSION from packaged code"


def test_notebook_validates_manifests_before_resolve_runtime_paths():
    """Item 2, 3, 4, 5: Verify manifest validation happens before resolve_runtime_paths and cross-checks roots."""
    src = notebook_cell_3_source()
    assert "runtime_root_from_code = str(Path(resolved_code_root).resolve().parents[1])" in src

    val_idx = src.index("validate_runtime_manifests(")
    res_idx = src.index("resolve_runtime_paths(")
    assert val_idx < res_idx, "validate_runtime_manifests must occur BEFORE resolve_runtime_paths in notebook"

    assert "assert os.path.realpath(paths[\"runtime_root\"]) == os.path.realpath(runtime_root_from_code)" in src


@pytest.mark.parametrize("stale_version", [9, 10, 11, 12, 13, 14])
def test_stale_package_rejected_by_v15_contract(tmp_path: Path, stale_version: int):
    """Item 7: Verify stale API 9/10/11/12/13/14 packages with matching SHA are rejected by REQUIRED_RUNTIME_API_VERSION = 15."""
    runtime = tmp_path / "runtime"
    code = runtime / "code" / "LegalQA"
    code.mkdir(parents=True)
    (runtime / "dataset_manifest.json").write_text(
        json.dumps({"runtime_api_version": stale_version, "git_sha": VALID_SHA})
    )
    (code / "code_manifest.json").write_text(
        json.dumps({"runtime_api_version": stale_version, "git_sha": VALID_SHA})
    )

    with pytest.raises(RuntimeError, match=f"runtime_api_version mismatch: found {stale_version}, expected 15"):
        validate_runtime_manifests(str(runtime), str(code), expected_api_version=15)


def test_fresh_v15_package_passes_v15_contract(tmp_path: Path):
    """Item 7: Verify fresh API 15 package passes validation."""
    runtime = tmp_path / "runtime"
    code = runtime / "code" / "LegalQA"
    code.mkdir(parents=True)
    (runtime / "dataset_manifest.json").write_text(
        json.dumps({"runtime_api_version": 15, "git_sha": VALID_SHA})
    )
    (code / "code_manifest.json").write_text(
        json.dumps({"runtime_api_version": 15, "git_sha": VALID_SHA})
    )

    provenance = validate_runtime_manifests(str(runtime), str(code), expected_api_version=15)
    assert provenance["runtime_api_version"] == 15
    assert provenance["git_sha"] == VALID_SHA


def test_nested_kaggle_layout_end_to_end_v15_resolution(tmp_path: Path):
    """Item 8: Verify end-to-end V15 sequence on exact Kaggle nested directory structure."""
    kaggle_input = tmp_path / "kaggle" / "input"
    ds_root = kaggle_input / "datasets" / "phucdangg" / "legalqa-task2-clean-data"
    code_root = ds_root / "code" / "LegalQA"
    (code_root / "src").mkdir(parents=True)
    (code_root / "scripts").mkdir(parents=True)
    (ds_root / "indexes" / "bm25").mkdir(parents=True)
    (ds_root / "indexes" / "dek21").mkdir(parents=True)

    # Populate manifests
    (ds_root / "dataset_manifest.json").write_text(
        json.dumps({"title": "LegalQA", "runtime_api_version": 15, "git_sha": VALID_SHA}),
        encoding="utf-8",
    )
    (ds_root / "code_manifest.json").write_text(
        json.dumps({"runtime_api_version": 15, "git_sha": VALID_SHA}),
        encoding="utf-8",
    )
    (code_root / "code_manifest.json").write_text(
        json.dumps({"runtime_api_version": 15, "git_sha": VALID_SHA}),
        encoding="utf-8",
    )
    (ds_root / "legal_chunks.parquet").write_bytes(b"PAR1_DUMMY")

    # Qwen dummy
    qwen_dir = kaggle_input / "qwen-lm" / "qwen2.5" / "transformers" / "3b-instruct" / "1"
    qwen_dir.mkdir(parents=True)
    (qwen_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}),
        encoding="utf-8",
    )

    # Execute Cell 3 logic step-by-step
    REQUIRED_RUNTIME_API_VERSION = 15
    resolved_code_root = resolve_packaged_code_root(str(kaggle_input), strict=True)
    runtime_root_from_code = str(Path(resolved_code_root).resolve().parents[1])

    assert os.path.realpath(resolved_code_root) == os.path.realpath(str(code_root))
    assert os.path.realpath(runtime_root_from_code) == os.path.realpath(str(ds_root))

    # Validate manifests
    provenance = validate_runtime_manifests(
        runtime_root=runtime_root_from_code,
        code_root=resolved_code_root,
        expected_api_version=REQUIRED_RUNTIME_API_VERSION,
    )
    assert provenance["runtime_api_version"] == 15

    # Resolve paths
    paths = resolve_runtime_paths(
        str(kaggle_input),
        strict=True,
        allow_remote_model_download=False,
    )

    # Cross check
    assert os.path.realpath(paths["runtime_root"]) == os.path.realpath(runtime_root_from_code)
    assert paths["bm25_dir"] == str(ds_root / "indexes" / "bm25")
    assert paths["dek21_dir"] == str(ds_root / "indexes" / "dek21")
    assert paths["qwen_model_path"] == str(qwen_dir)
