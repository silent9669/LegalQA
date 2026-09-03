"""Tests for LegalQA V7/V9 runtime integrity, bootstrap protection, and preflight."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from scripts.bootstrap_kaggle_env import (
    assert_protected_versions_unchanged,
    get_installed_distribution_version,
    satisfies_spec,
    snapshot_protected_versions,
    write_protected_constraints,
)
from src.task2.checkpoint_manifest import assert_final_checkpoint
from src.task2.path_resolver import find_qwen_model_dir, find_runtime_roots, resolve_runtime_paths
from src.task2.training.train_generator import (
    build_grounded_training_examples,
    build_sft_config,
    run_seq_len_diagnostic,
)
from scripts.preflight_kaggle import run_preflight_checks


def test_bootstrap_protected_snapshot_detects_version_drift():
    """Step 1.1: Verify assert_protected_versions_unchanged raises when a protected package differs."""
    before = {"torch": "2.6.0", "triton": "3.2.0"}
    after = {"torch": "2.7.0", "triton": "3.2.0"}

    with pytest.raises(RuntimeError, match="Protected runtime package changed"):
        assert_protected_versions_unchanged(before, after)


def test_bootstrap_protected_snapshot_detects_removed_package():
    """Verify assert_protected_versions_unchanged raises when a protected package was removed."""
    before = {"torch": "2.6.0", "triton": "3.2.0"}
    after = {"triton": "3.2.0"}

    with pytest.raises(RuntimeError, match="Protected runtime package removed"):
        assert_protected_versions_unchanged(before, after)


def test_bootstrap_protected_snapshot_detects_new_protected_package():
    """Verify assert_protected_versions_unchanged raises when a new protected package appears."""
    before = {"torch": "2.6.0"}
    after = {"torch": "2.6.0", "triton": "3.2.0"}

    with pytest.raises(RuntimeError, match="New protected runtime package introduced"):
        assert_protected_versions_unchanged(before, after)


def test_satisfies_spec():
    """Step 1.3: Verify version specification matching."""
    assert satisfies_spec("0.12.0", ">=0.11.0") is True
    assert satisfies_spec("0.10.0", ">=0.11.0") is False
    assert satisfies_spec("2.2.0", ">=2.0.0,<3.0.0") is True
    assert satisfies_spec(None, ">=1.0.0") is False


def test_write_protected_constraints(tmp_path):
    """Step 1.4: Verify write_protected_constraints creates valid constraint lines."""
    snapshot = {
        "torch": "2.6.0",
        "nvidia-cuda-runtime-cu12": "12.4.127",
    }
    constraints_path = tmp_path / "constraints.txt"
    written = write_protected_constraints(snapshot, str(constraints_path))

    assert os.path.exists(written)
    content = constraints_path.read_text()
    assert "torch==2.6.0" in content
    assert "nvidia-cuda-runtime-cu12==12.4.127" in content


def test_generator_manifest_uses_canonical_base_model_id(tmp_path):
    """Task 2: Verify assert_final_checkpoint validates base_model_id regardless of resolved_model_path."""
    ckpt_dir = tmp_path / "qlora_ckpt"
    ckpt_dir.mkdir()

    manifest = {
        "base_model_id": "Qwen/Qwen2.5-3B-Instruct",
        "resolved_model_path": "/kaggle/input/some-dataset/qwen2.5-3b-instruct/1",
        "training_scope": "all_allowed_task2_data",
        "is_final_checkpoint": True,
        "smoke_only": False,
        "val_fold_excluded": None,
    }
    (ckpt_dir / "generator_manifest.json").write_text(json.dumps(manifest))

    verified = assert_final_checkpoint(
        str(ckpt_dir),
        expected_base_model="Qwen/Qwen2.5-3B-Instruct",
        component_name="generator",
    )
    assert verified["base_model_id"] == "Qwen/Qwen2.5-3B-Instruct"

    # Mismatched expected base model should raise
    with pytest.raises(ValueError, match="base model mismatch"):
        assert_final_checkpoint(
            str(ckpt_dir),
            expected_base_model="Qwen/Qwen2.5-7B-Instruct",
            component_name="generator",
        )


def test_preflight_bm25s_index_files_strict(tmp_path):
    """Task 6: Verify preflight checks actual BM25S index files when check_indexes=True."""
    bm25_dir = tmp_path / "bm25"
    bm25_dir.mkdir()

    # Case 1: manifest exists, but bm25s_index/params.index.json is missing
    (bm25_dir / "bm25_manifest.json").write_text(json.dumps({"corpus_size": 100}))

    res = run_preflight_checks(
        pipeline_config_path="configs/pipeline.yaml",
        models_config_path="configs/models.yaml",
        production_config_path="configs/production_selection.yaml",
        require_cuda=False,
        check_dataset_files=False,
        check_indexes=True,
        bm25_dir=str(bm25_dir),
        dek21_dir=None,
    )
    assert not res["passed"]
    assert any("BM25 index parameters file missing" in err for err in res["errors"])

    # Case 2: Create bm25s_index/params.index.json -> passes
    idx_dir = bm25_dir / "bm25s_index"
    idx_dir.mkdir()
    (idx_dir / "params.index.json").write_text("{}")

    res2 = run_preflight_checks(
        pipeline_config_path="configs/pipeline.yaml",
        models_config_path="configs/models.yaml",
        production_config_path="configs/production_selection.yaml",
        require_cuda=False,
        check_dataset_files=False,
        check_indexes=True,
        bm25_dir=str(bm25_dir),
        dek21_dir=None,
    )
    assert res2["passed"] is True


def test_preflight_require_public_gate(tmp_path):
    """Task 6: Verify require_public flag in preflight."""
    res_missing_public = run_preflight_checks(
        pipeline_config_path="configs/pipeline.yaml",
        models_config_path="configs/models.yaml",
        production_config_path="configs/production_selection.yaml",
        require_cuda=False,
        check_dataset_files=False,
        check_indexes=False,
        public_path=str(tmp_path / "nonexistent-public.json"),
        require_public=True,
    )
    assert not res_missing_public["passed"]
    assert any("Required public-official.json is missing" in err for err in res_missing_public["errors"])


def test_strict_kaggle_path_resolution_no_local_fallback(tmp_path):
    """Task 8: Verify strict path resolution raises if no dataset roots found instead of falling back to local artifacts."""
    empty_input = tmp_path / "empty_kaggle_input"
    empty_input.mkdir()

    with pytest.raises(RuntimeError, match="No valid LegalQA dataset root found"):
        resolve_runtime_paths(str(empty_input), strict=True)


def test_stale_runtime_api_manifest_detection():
    """Task 7: Verify that runtime API version is defined and validated."""
    import yaml
    from src.task2.runtime_integrity import EXPECTED_RUNTIME_API_VERSION
    with open("configs/runtime_api.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data.get("runtime_api_version") == EXPECTED_RUNTIME_API_VERSION
    assert EXPECTED_RUNTIME_API_VERSION == 15


def test_sequence_diagnostics_use_tokenizer_counts(tmp_path):
    """Task 9: Verify sequence diagnostics use tokenizer counts and return diagnostics."""
    class MockTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            user_content = messages[1]["content"]
            return f"<|im_start|>system\nPrompt<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids)

    tok = MockTokenizer()
    import pandas as pd
    df_qa = pd.DataFrame([
        {"qa_id": "q1", "question_raw": "Q1", "answer_raw": "A1 " * 10}
    ])
    examples, diag = build_grounded_training_examples(
        df_qa=df_qa,
        tokenizer=tok,
        max_seq_len=50,
        return_diagnostics=True,
    )
    assert len(examples) == 1
    assert "p50_tokens" in diag
    assert "p95_tokens" in diag
    assert "drop_rate" in diag
    assert diag["dropped_count"] == 0
