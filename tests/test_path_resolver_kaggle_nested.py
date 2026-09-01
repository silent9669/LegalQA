"""Tests for robust recursive LegalQA runtime root resolution on Kaggle nested layouts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.task2.path_resolver import (
    find_qwen_model_dir,
    find_runtime_roots,
    resolve_runtime_paths,
)


def _populate_legalqa_dataset_root(root: Path, with_data_subfolder: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_manifest.json").write_text(
        json.dumps({"title": "LegalQA", "runtime_api_version": 13}),
        encoding="utf-8",
    )
    if with_data_subfolder:
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "legal_chunks.parquet").write_bytes(b"PAR1_DUMMY")
    else:
        (root / "legal_chunks.parquet").write_bytes(b"PAR1_DUMMY")

    bm25_dir = root / "indexes" / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    (bm25_dir / "bm25_manifest.json").write_text("{}", encoding="utf-8")


def _populate_qwen_model_dir(root: Path) -> Path:
    qwen_dir = root / "qwen-lm" / "qwen2.5" / "transformers" / "3b-instruct" / "1"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    (qwen_dir / "config.json").write_text(
        json.dumps({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "torch_dtype": "bfloat16",
        }),
        encoding="utf-8",
    )
    return qwen_dir


def test_kaggle_nested_dataset_root_discovery(tmp_path: Path):
    """Reproduce exact Kaggle nested path: /kaggle/input/datasets/<owner>/<slug>."""
    kaggle_input = tmp_path / "kaggle" / "input"
    ds_root = kaggle_input / "datasets" / "phucdangg" / "legalqa-task2-clean-data"
    _populate_legalqa_dataset_root(ds_root)
    qwen_dir = _populate_qwen_model_dir(kaggle_input)

    roots = find_runtime_roots(str(kaggle_input))
    assert len(roots) == 1
    assert roots[0] == str(ds_root)

    paths = resolve_runtime_paths(
        str(kaggle_input),
        strict=True,
        allow_remote_model_download=False,
    )
    assert paths["runtime_root"] == str(ds_root)
    assert paths["bm25_dir"] == str(ds_root / "indexes" / "bm25")
    assert paths["qwen_model_path"] == str(qwen_dir)


def test_kaggle_custom_owner_slug_nested_discovery(tmp_path: Path):
    """Verify that arbitrary owner/slug nested layouts are discovered without hardcoding."""
    kaggle_input = tmp_path / "kaggle" / "input"
    ds_root = kaggle_input / "datasets" / "arbitrary_org" / "custom-law-qa-2026"
    _populate_legalqa_dataset_root(ds_root, with_data_subfolder=True)
    _populate_qwen_model_dir(kaggle_input)

    roots = find_runtime_roots(str(kaggle_input))
    assert len(roots) == 1
    assert roots[0] == str(ds_root)

    paths = resolve_runtime_paths(
        str(kaggle_input),
        strict=True,
        allow_remote_model_download=False,
    )
    assert paths["runtime_root"] == str(ds_root)
    assert paths["data_dir"] == str(ds_root / "data")


def test_kaggle_legacy_flat_dataset_root_discovery(tmp_path: Path):
    """Verify that legacy flat /kaggle/input/<slug> layout continues to work."""
    kaggle_input = tmp_path / "kaggle" / "input"
    ds_root = kaggle_input / "legalqa-task2-clean-data"
    _populate_legalqa_dataset_root(ds_root)

    roots = find_runtime_roots(str(kaggle_input))
    assert len(roots) == 1
    assert roots[0] == str(ds_root)


def test_ambiguous_dataset_roots_hard_fails(tmp_path: Path):
    """Verify that multiple competing dataset roots trigger an explicit RuntimeError."""
    kaggle_input = tmp_path / "kaggle" / "input"
    ds_root1 = kaggle_input / "datasets" / "user1" / "legalqa-v1"
    ds_root2 = kaggle_input / "datasets" / "user2" / "legalqa-v2"
    _populate_legalqa_dataset_root(ds_root1)
    _populate_legalqa_dataset_root(ds_root2)

    roots = find_runtime_roots(str(kaggle_input))
    assert len(roots) == 2

    with pytest.raises(RuntimeError, match="Ambiguous runtime dataset roots"):
        resolve_runtime_paths(str(kaggle_input), strict=True)


def test_no_dataset_roots_strict_hard_fails(tmp_path: Path):
    """Verify that 0 matching dataset roots triggers an explicit RuntimeError in strict mode."""
    kaggle_input = tmp_path / "kaggle" / "input"
    kaggle_input.mkdir(parents=True)
    # Populate unrelated directory
    unrelated = kaggle_input / "datasets" / "user" / "some-image-dataset"
    unrelated.mkdir(parents=True)
    (unrelated / "images.txt").write_text("img1.png", encoding="utf-8")

    roots = find_runtime_roots(str(kaggle_input))
    assert len(roots) == 0

    with pytest.raises(RuntimeError, match="No valid LegalQA dataset root found"):
        resolve_runtime_paths(str(kaggle_input), strict=True)
