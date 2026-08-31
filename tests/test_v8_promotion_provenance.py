"""Tests for LegalQA V8/V9 promotion provenance, byte-identical report serialization, and OOF checkpoint checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from src.common.hashing import sha256_file
from src.task2.evaluation import write_promotion_report
from src.task2.production_config import (
    load_production_selection,
    validate_production_selection_for_profile,
)
from scripts.promote_production_selection import promote_production_selection
from scripts.run_oof_validation import (
    assert_fold_generator_checkpoint,
    assert_fold_reranker_checkpoint,
)


def _build_valid_report_dict() -> Dict[str, Any]:
    sys_template = {
        "sample_ids_sha256": "hash_123",
        "sample_size": 250,
        "candidate_family_meteors": {"stitched_extract": 0.315},
        "retrieval_metrics": {"chunk_mrr": 0.45},
        "reranker_checkpoint": "checkpoints/reranker/best",
        "generator_model": "Qwen/Qwen2.5-3B-Instruct",
        "adapter_path": None,
        "dense_model": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        "no_mocks": True,
        "no_fallbacks": True,
    }
    return {
        "screen_protocol_version": 8,
        "held_out_fold": 0,
        "sample_ids_sha256": "hash_123",
        "sample_size": 250,
        "evaluated_systems": {
            "R0G0": dict(sys_template, reranker_checkpoint="BAAI/bge-reranker-v2-m3"),
            "R1G0": dict(sys_template, reranker_checkpoint="checkpoints/reranker/best"),
            "R_SELECTED_G1": dict(sys_template, reranker_checkpoint="checkpoints/reranker/best"),
        },
        "selected_reranker": {"use_task_tuned": True, "checkpoint": "checkpoints/reranker/best"},
        "selected_generator": {"use_qlora": False, "adapter": None},
        "final_measured_system_key": "R1G0",
        "candidate_policy": {"type": "fixed_baseline", "best_fixed_candidate": "stitched_extract"},
        "overall_deployable_winner": "stitched_extract",
        "overall_deployable_meteor": 0.315,
    }


def test_promotion_report_mirror_is_byte_identical(tmp_path):
    """Task 7: Verify write_promotion_report writes byte-identical JSON files and returns valid sha256."""
    report = {
        "screen_protocol_version": 8,
        "held_out_fold": 0,
        "sample_size": 250,
        "metrics": {"meteor": 0.320},
    }
    p1 = tmp_path / "report_primary.json"
    p2 = tmp_path / "report_mirror.json"

    res = write_promotion_report(report, str(p1), str(p2))

    assert p1.read_bytes() == p2.read_bytes()
    expected_sha = sha256_file(p1)
    assert res["sha256"] == expected_sha
    assert res["primary_path"] == str(p1)
    assert res["mirror_path"] == str(p2)


def test_promoter_valid_protocol_8_report(tmp_path):
    """Verify promoter succeeds on complete valid Protocol 8 report."""
    rep = _build_valid_report_dict()
    rep_file = tmp_path / "valid_report.json"
    rep_file.write_text(json.dumps(rep))

    out_cfg = tmp_path / "promoted_config.yaml"
    promoted = promote_production_selection(
        report_path=str(rep_file),
        config_path="configs/production_selection.yaml",
        output_path=str(out_cfg),
    )
    assert promoted["status"] == "PROMOTED"
    assert promoted["screen_protocol_version"] == 8
    assert promoted["candidate_policy"]["best_fixed_candidate"] == "stitched_extract"


def test_promoter_rejects_missing_system_field(tmp_path):
    """Task 3: Verify promoter rejects report where evaluated system is missing required fields."""
    rep = _build_valid_report_dict()
    del rep["evaluated_systems"]["R1G0"]["sample_ids_sha256"]  # Omit field
    rep_file = tmp_path / "missing_field.json"
    rep_file.write_text(json.dumps(rep))

    with pytest.raises(ValueError, match="missing required fields"):
        promote_production_selection(
            report_path=str(rep_file),
            config_path="configs/production_selection.yaml",
            output_path=str(tmp_path / "out.yaml"),
        )


def test_promoter_rejects_missing_required_system(tmp_path):
    """Task 3: Verify promoter rejects report missing R1G0 or R_SELECTED_G1."""
    rep = _build_valid_report_dict()
    del rep["evaluated_systems"]["R1G0"]
    rep_file = tmp_path / "missing_system.json"
    rep_file.write_text(json.dumps(rep))

    with pytest.raises(ValueError, match="Missing required evaluated system"):
        promote_production_selection(
            report_path=str(rep_file),
            config_path="configs/production_selection.yaml",
            output_path=str(tmp_path / "out.yaml"),
        )


def test_promoter_rejects_sample_id_mismatch(tmp_path):
    """Task 8: Verify promoter rejects report where evaluated systems used different evaluation subsets."""
    rep = _build_valid_report_dict()
    rep["evaluated_systems"]["R1G0"]["sample_ids_sha256"] = "DIFFERENT_HASH_456"
    rep_file = tmp_path / "sample_mismatch.json"
    rep_file.write_text(json.dumps(rep))

    with pytest.raises(ValueError, match="Sample IDs hash mismatch"):
        promote_production_selection(
            report_path=str(rep_file),
            config_path="configs/production_selection.yaml",
            output_path=str(tmp_path / "out.yaml"),
        )


def test_promoter_rejects_reranker_checkpoint_mismatch(tmp_path):
    """Task 8: Verify promoter rejects report where final measured system used a different reranker checkpoint."""
    rep = _build_valid_report_dict()
    rep["evaluated_systems"]["R1G0"]["reranker_checkpoint"] = "checkpoints/other/reranker"
    rep_file = tmp_path / "reranker_mismatch.json"
    rep_file.write_text(json.dumps(rep))

    with pytest.raises(ValueError, match="Reranker checkpoint mismatch"):
        promote_production_selection(
            report_path=str(rep_file),
            config_path="configs/production_selection.yaml",
            output_path=str(tmp_path / "out.yaml"),
        )


def test_promoter_rejects_candidate_score_mismatch(tmp_path):
    """Task 8: Verify promoter rejects report where overall_deployable_meteor differs from measured candidate score."""
    rep = _build_valid_report_dict()
    rep["overall_deployable_meteor"] = 0.999  # Fake score mismatch
    rep_file = tmp_path / "score_mismatch.json"
    rep_file.write_text(json.dumps(rep))

    with pytest.raises(ValueError, match="Candidate score mismatch"):
        promote_production_selection(
            report_path=str(rep_file),
            config_path="configs/production_selection.yaml",
            output_path=str(tmp_path / "out.yaml"),
        )


def test_assert_fold_checkpoints_provenance(tmp_path):
    """Task 9: Verify assert_fold_reranker_checkpoint and assert_fold_generator_checkpoint."""
    # 1. Reranker wrong excluded fold fails
    r_dir = tmp_path / "reranker_fold0"
    r_dir.mkdir()
    (r_dir / "reranker_manifest.json").write_text(json.dumps({
        "base_model": "BAAI/bge-reranker-v2-m3",
        "val_fold_excluded": 1,  # Trained excluding fold 1
        "training_scope": "folds_excluding_1",
        "smoke_only": False,
    }))

    with pytest.raises(ValueError, match="val_fold_excluded="):
        assert_fold_reranker_checkpoint(str(r_dir), fold_id=0, expected_base_model="BAAI/bge-reranker-v2-m3")

    # Correct fold passes
    verified = assert_fold_reranker_checkpoint(str(r_dir), fold_id=1, expected_base_model="BAAI/bge-reranker-v2-m3")
    assert verified["val_fold_excluded"] == 1

    # 2. Generator smoke checkpoint fails
    g_dir = tmp_path / "gen_fold0"
    g_dir.mkdir()
    (g_dir / "generator_manifest.json").write_text(json.dumps({
        "base_model_id": "Qwen/Qwen2.5-3B-Instruct",
        "val_fold_excluded": 0,
        "training_scope": "folds_excluding_0",
        "smoke_only": True,  # Smoke!
    }))

    with pytest.raises(ValueError, match="smoke_only"):
        assert_fold_generator_checkpoint(str(g_dir), fold_id=0, expected_base_model="Qwen/Qwen2.5-3B-Instruct")
