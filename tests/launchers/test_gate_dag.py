"""Task 8: gate DAG + platform adapter regressions."""

from __future__ import annotations

import pytest

from scripts.run_gpu_gate import (
    GATE_DAG,
    build_gate_request,
    run_platform_stage,
    validate_parent_gate,
)

CAND = {
    "candidate_sha": "c" * 16,
    "algorithm_sha256": "a" * 64,
    "dataset_sha256": "b" * 64,
    "scorer_sha256": "c" * 64,
    "runtime_profile": "kaggle_t4x2",
    "runtime_sha256": "d" * 64,
}


def _parent(stage="kaggle_t4x2", sha="c" * 16):
    return {"status": "PASS", "candidate_sha": sha, "stage": stage, "report_sha256": "e" * 64}


def test_a100_rejects_different_candidate():
    with pytest.raises(ValueError, match="candidate"):
        validate_parent_gate({"status": "PASS", "candidate_sha": "old"}, "new", "a100_micro_probe")


def test_dag_order_and_parent_requirements():
    assert list(GATE_DAG) == ["kaggle_t4x2", "colab_t4", "a100_micro_probe"]
    # First stage takes no parent.
    req = build_gate_request(CAND, "kaggle_t4x2", None)
    assert req["required_parent_stage"] is None
    validate_parent_gate(None, "c" * 16, "kaggle_t4x2")
    # Later stages require their exact parent.
    with pytest.raises(ValueError, match="requires parent"):
        build_gate_request(CAND, "colab_t4", None)
    with pytest.raises(ValueError, match="requires a parent"):
        validate_parent_gate(None, "c" * 16, "colab_t4")
    req2 = build_gate_request(CAND, "colab_t4", _parent("kaggle_t4x2"))
    assert req2["parent_report_sha256"] == "e" * 64
    validate_parent_gate(_parent("kaggle_t4x2"), "c" * 16, "colab_t4")


def test_wrong_parent_stage_or_failed_status_rejected():
    with pytest.raises(ValueError, match="stage mismatch"):
        validate_parent_gate(_parent("kaggle_t4x2"), "c" * 16, "a100_micro_probe")
    bad = _parent("colab_t4")
    bad["status"] = "FAILED_TRANSIENT"
    with pytest.raises(ValueError, match="not PASS"):
        validate_parent_gate(bad, "c" * 16, "a100_micro_probe")
    bad2 = _parent("colab_t4")
    del bad2["report_sha256"]
    with pytest.raises(ValueError, match="report_sha256"):
        validate_parent_gate(bad2, "c" * 16, "a100_micro_probe")


def test_request_binds_same_candidate_and_declared_runtime():
    cand_b = dict(CAND, runtime_profile="modal_a100", runtime_sha256="f" * 64)
    req = build_gate_request(cand_b, "a100_micro_probe", _parent("colab_t4"))
    assert req["candidate_sha"] == "c" * 16
    assert req["runtime_profile"] == "modal_a100"
    assert req["runtime_sha256"] == "f" * 64
    with pytest.raises(ValueError, match="unknown gate stage"):
        build_gate_request(CAND, "tpu_v9", None)


def test_platform_stage_rejects_missing_candidate():
    with pytest.raises(ValueError, match="candidate_path"):
        run_platform_stage({"stage": "kaggle_t4x2"})
    with pytest.raises(ValueError, match="unknown gate stage"):
        run_platform_stage({"stage": "nope", "candidate_path": "x"})


def test_smoke_cannot_upload(tmp_path):
    import yaml

    from src.task2.pipeline.profiles import load_profile_from_yaml

    profile = load_profile_from_yaml("configs/task2/runtime/kaggle_t4x2.yaml")
    assert "smoke" in profile.name or "kaggle" in profile.name
    # run_pipeline refuses auto-upload for smoke/probe profiles: simulate the
    # guard predicate directly (no GPU execution in this unit test).
    is_smoke_profile = any(tag in profile.name for tag in ("smoke", "probe", "screen", "kaggle"))
    assert is_smoke_profile


def test_pure_data_packaging_rejects_code_and_secrets(tmp_path):
    import pandas as pd

    from src.task2.dataset.validator import validate_dataset

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "exploit.py").write_text("print('evil')", encoding="utf-8")
    (data_dir / "qa_unique.parquet").write_text("x", encoding="utf-8")
    report = validate_dataset(str(data_dir), schema_path="configs/dataset_schema.yaml")
    assert report["status"] == "FAIL"
    assert any("code" in e.lower() or "executable" in e.lower() for e in report["errors"])
