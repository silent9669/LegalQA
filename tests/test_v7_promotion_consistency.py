"""Tests for LegalQA V7 component-consistent screening and promotion logic."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from src.task2.evaluation import (
    best_deployable_candidate,
    decide_generator_promotion,
    decide_reranker_promotion,
    evaluate_checkpoint,
    run_screen_matrix,
)
from src.task2.production_config import (
    load_production_selection,
    validate_production_selection_for_profile,
)
from scripts.promote_production_selection import promote_production_selection


def test_best_deployable_candidate_excludes_meta_keys():
    """Verify best_deployable_candidate selects highest scoring candidate excluding meta keys."""
    summary = {
        "candidate_family_meteors": {
            "selected": 0.350,
            "oracle_best": 0.400,
            "stitched_extract": 0.310,
            "focused_extract": 0.280,
            "strategy_f_1000": 0.325,
        }
    }
    cand_name, cand_score = best_deployable_candidate(summary)
    assert cand_name == "strategy_f_1000"
    assert cand_score == 0.325


def test_reranker_promotion_decision_improves_retrieval():
    """Task 3: Test decide_reranker_promotion logic."""
    r0g0 = {
        "retrieval_metrics": {"chunk_mrr": 0.450, "chunk_recall_at_8": 0.700},
        "candidate_family_meteors": {"stitched_extract": 0.300},
    }
    # R1 improves retrieval without downstream regression -> Promote
    r1g0_good = {
        "retrieval_metrics": {"chunk_mrr": 0.480, "chunk_recall_at_8": 0.750},
        "candidate_family_meteors": {"stitched_extract": 0.302},
    }
    decision = decide_reranker_promotion(r0g0, r1g0_good, retrieval_tolerance=0.001, meteor_tolerance=0.005)
    assert decision["promote"] is True

    # R1 regresses retrieval -> Reject
    r1g0_bad = {
        "retrieval_metrics": {"chunk_mrr": 0.440, "chunk_recall_at_8": 0.690},
        "candidate_family_meteors": {"stitched_extract": 0.300},
    }
    decision_bad = decide_reranker_promotion(r0g0, r1g0_bad, retrieval_tolerance=0.001, meteor_tolerance=0.005)
    assert decision_bad["promote"] is False


def test_generator_promotion_decision_requires_deployable_improvement():
    """Task 3: Test decide_generator_promotion logic."""
    selected_base = {
        "candidate_family_meteors": {
            "stitched_extract": 0.310,
            "focused_extract": 0.280,
            "generated": 0.250,
            "strategy_f_1000": 0.300,
        }
    }

    # Case 1: QLoRA generated candidate beats all base candidates -> Promote
    qlora_good = {
        "candidate_family_meteors": {
            "stitched_extract": 0.310,
            "focused_extract": 0.280,
            "generated": 0.340,
            "strategy_f_1000": 0.335,
        }
    }
    decision = decide_generator_promotion(selected_base, qlora_good, meteor_tolerance=0.005)
    assert decision["promote"] is True

    # Case 2: QLoRA improves generated (0.290 > 0.250) but still worse than best fixed (0.310) -> Reject
    qlora_insufficient = {
        "candidate_family_meteors": {
            "stitched_extract": 0.310,
            "focused_extract": 0.280,
            "generated": 0.290,
            "strategy_f_1000": 0.305,
        }
    }
    decision_insuf = decide_generator_promotion(selected_base, qlora_insufficient, meteor_tolerance=0.005)
    assert decision_insuf["promote"] is False


def test_promoter_rejects_inconsistent_report(tmp_path):
    """Task 5: Verify promote_production_selection rejects component-inconsistent report."""
    report_file = tmp_path / "inconsistent_report.json"

    # Inconsistent: selected_generator is use_qlora=False, but final_measured_system_key is R_SELECTED_G1
    bad_report = {
        "screen_protocol_version": 8,
        "held_out_fold": 0,
        "sample_ids_sha256": "abc123456",
        "sample_size": 250,
        "evaluated_systems": {
            "R0G0": {"sample_ids_sha256": "abc123456", "sample_size": 250},
            "R1G0": {"sample_ids_sha256": "abc123456", "sample_size": 250},
            "R_SELECTED_G1": {"sample_ids_sha256": "abc123456", "sample_size": 250},
        },
        "selected_reranker": {
            "use_task_tuned": True,
            "checkpoint": "checkpoints/reranker/best",
            "decision_reason": "improved MRR",
        },
        "selected_generator": {
            "use_qlora": False,
            "adapter": None,
            "decision_reason": "did not beat fixed extract",
        },
        "final_measured_system_key": "R_SELECTED_G1",  # Inconsistent with use_qlora=False!
        "candidate_policy": {
            "type": "fixed_baseline",
            "best_fixed_candidate": "strategy_f_1000",
        },
        "overall_deployable_winner": "strategy_f_1000",
        "overall_deployable_meteor": 0.330,
    }
    report_file.write_text(json.dumps(bad_report))

    with pytest.raises(ValueError, match="Component inconsistency"):
        promote_production_selection(
            report_path=str(report_file),
            config_path="configs/production_selection.yaml",
            output_path=str(tmp_path / "out.yaml"),
        )


def test_screen_matrix_missing_checkpoints_fail(tmp_path):
    """Task 3: Verify run_screen_matrix raises FileNotFoundError if tuned reranker or adapter is missing."""
    with pytest.raises(FileNotFoundError, match="Tuned reranker checkpoint directory missing"):
        run_screen_matrix(
            qa_path="artifacts/task2/data/qa_unique.parquet",
            fold_path="artifacts/task2/data/fold_assignments.parquet",
            chunks_path="artifacts/task2/data/legal_chunks.parquet",
            labels_path="artifacts/task2/data/retrieval_labels.parquet",
            tuned_reranker=str(tmp_path / "nonexistent_reranker"),
        )
