"""Tests for LegalQA V9 full-OOF real-only execution contract and fold-map validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.run_oof_validation import (
    assert_fold_generator_checkpoint,
    assert_fold_reranker_checkpoint,
    validate_fold_checkpoint_map,
    validate_full_mode_contract,
)


def test_full_mode_missing_dense_index_fails(tmp_path):
    """Task 4: Verify validate_full_mode_contract raises if DEk21 embeddings.npy is missing."""
    bm25_dir = tmp_path / "bm25"
    bm25_dir.mkdir()
    dek21_dir = tmp_path / "dek21"
    dek21_dir.mkdir()
    # Missing embeddings.npy

    with pytest.raises(RuntimeError, match="DEk21"):
        validate_full_mode_contract(
            bm25_dir=str(bm25_dir),
            dek21_dir=str(dek21_dir),
            held_out_fold=0,
            fold_checkpoint_map=None,
            n_splits=5,
            reranker_checkpoint="BAAI/bge-reranker-v2-m3",
            adapter_path=None,
            retrieval_device="cuda:1",
            gen_device="cuda:0",
        )


def test_full_mode_cpu_device_fails(tmp_path):
    """Task 4: Verify validate_full_mode_contract raises if devices are CPU in full mode."""
    bm25_dir = tmp_path / "bm25"
    bm25_dir.mkdir()
    dek21_dir = tmp_path / "dek21"
    dek21_dir.mkdir()
    (dek21_dir / "embeddings.npy").write_bytes(b"dummy")
    (dek21_dir / "dense_manifest.json").write_text("{}")

    with pytest.raises(RuntimeError, match="CUDA device"):
        validate_full_mode_contract(
            bm25_dir=str(bm25_dir),
            dek21_dir=str(dek21_dir),
            held_out_fold=0,
            fold_checkpoint_map=None,
            n_splits=5,
            reranker_checkpoint="BAAI/bge-reranker-v2-m3",
            adapter_path=None,
            retrieval_device="cpu",  # Invalid in full mode!
            gen_device="cuda:0",
        )


def test_full_oof_incomplete_checkpoint_map_fails():
    """Task 5: Verify validate_fold_checkpoint_map raises if any target fold is missing."""
    fold_map = {
        0: {"reranker": "/r0", "adapter": "/g0"},
        1: {"reranker": "/r1", "adapter": "/g1"},
    }
    with pytest.raises(RuntimeError, match="missing folds"):
        validate_fold_checkpoint_map(
            fold_map,
            target_folds=[0, 1, 2, 3, 4],
            require_reranker=True,
            require_adapter=True,
        )
