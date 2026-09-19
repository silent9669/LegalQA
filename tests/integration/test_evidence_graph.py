"""Task 11: end-to-end evidence rehearsal regressions."""

from __future__ import annotations

import pytest

from src.task2.provenance.evidence_graph import (
    final_run_checklist,
    join_evidence_reports,
    validate_evidence_graph,
)

CAND = "c" * 16


def _gate(stage):
    return {"kind": "gate", "stage": stage, "status": "PASS", "candidate_sha": CAND,
            "report_sha256": "r" * 64, "measured": True}


def _full_reports():
    return [
        _gate("kaggle_t4x2"),
        _gate("colab_t4"),
        _gate("a100_micro_probe"),
        {"kind": "training", "candidate_sha": CAND, "optimizer_steps": 762,
         "training_samples": 6102, "measured": True},
        {"kind": "offline_metrics", "candidate_sha": CAND, "meteor": 0.55, "rouge": 0.53,
         "checkpoint": "ckpt/1", "split_fingerprint": "s" * 64, "measured": True},
        {"kind": "inference", "candidate_sha": CAND, "num_predictions": 1000,
         "submission_sha256": "d" * 64, "measured": True},
        {"kind": "a100", "candidate_sha": CAND, "status": "PASS", "wall_seconds": 16200,
         "gpu": "A100-40GB", "measured": True},
        {"kind": "receipt", "candidate_sha": CAND, "commit_sha": "e" * 40,
         "status": "PUBLISH_VERIFIED", "measured": True},
    ]


def test_graph_rejects_missing_a100_measurement():
    with pytest.raises(ValueError, match="timing"):
        validate_evidence_graph({"candidate_sha": "c", "a100": {"status": "PASS"}})


def test_join_rejects_mixed_candidates_and_fabricated_counts():
    reports = _full_reports()
    tampered = [dict(r, candidate_sha="f" * 16) if r["kind"] == "gate" else r for r in reports]
    with pytest.raises(ValueError, match="one candidate_sha"):
        join_evidence_reports(tampered)
    estimated = [dict(r, estimated=True) if r["kind"] == "training" else r for r in reports]
    with pytest.raises(ValueError, match="fabricated"):
        join_evidence_reports(estimated)
    zero_steps = [dict(r, optimizer_steps=0) if r["kind"] == "training" else r for r in reports]
    with pytest.raises(ValueError, match="optimizer_steps"):
        join_evidence_reports(zero_steps)


def test_full_graph_validates_and_checklist_passes():
    graph = join_evidence_reports(_full_reports())
    validate_evidence_graph(graph)
    checklist = final_run_checklist(graph)
    assert checklist["overall"] == "PASS"
    assert all(c["status"] == "PASS" for c in checklist["checks"])


def test_incomplete_graph_fails_checklist_not_silently():
    graph = join_evidence_reports(_full_reports())
    del graph["receipt"]
    checklist = final_run_checklist(graph)
    assert checklist["overall"] == "FAIL"
    with pytest.raises(ValueError, match="receipt"):
        validate_evidence_graph(graph)


def test_inference_accepts_public_and_private_counts():
    base = [r for r in _full_reports() if r["kind"] != "inference"]
    for count in (1000, 1918):
        link = {"kind": "inference", "candidate_sha": CAND, "num_predictions": count,
                "expected_count": count, "submission_sha256": "d" * 64, "measured": True}
        graph = join_evidence_reports(base + [link])
        assert graph["inference"]["num_predictions"] == count
    bad = {"kind": "inference", "candidate_sha": CAND, "num_predictions": 999,
           "expected_count": 1000, "submission_sha256": "d" * 64, "measured": True}
    with pytest.raises(ValueError, match="count mismatch"):
        join_evidence_reports(base + [bad])
    zero = {"kind": "inference", "candidate_sha": CAND, "num_predictions": 0,
            "submission_sha256": "d" * 64, "measured": True}
    with pytest.raises(ValueError, match="positive prediction count"):
        join_evidence_reports(base + [zero])


@pytest.mark.parametrize("query_count", [1000, 1918, 500])
def test_inference_supports_public_private_and_nonzero_queries(query_count):
    reports = _full_reports()
    for r in reports:
        if r["kind"] == "inference":
            r["num_predictions"] = query_count
    graph = join_evidence_reports(reports)
    validate_evidence_graph(graph)
    checklist = final_run_checklist(graph)
    assert checklist["overall"] == "PASS"
    assert all(c["status"] == "PASS" for c in checklist["checks"])


def test_inference_rejects_zero_or_negative_predictions():
    reports = _full_reports()
    zero_inf = [dict(r, num_predictions=0) if r["kind"] == "inference" else r for r in reports]
    with pytest.raises(ValueError, match="prediction"):
        join_evidence_reports(zero_inf)

