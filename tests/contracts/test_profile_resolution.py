import os
import pytest
from src.task2.pipeline.profiles import load_profile_from_yaml
from src.task2.provenance.freeze_tuple import compute_freeze_tuple_hash, verify_smoke_pass

def test_resolve_colab_a100_profile():
    profile = load_profile_from_yaml("configs/colab_train_a100.yaml")
    assert profile.name == "colab_train_a100"
    assert profile.requires_generator is True
    assert profile.run_generator_training is True

def test_freeze_tuple_computation():
    freeze_hash = compute_freeze_tuple_hash(
        dataset_manifest_sha="abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abc",
        git_commit_sha="20d0e517ab06411082b2ad5a8dfe872796d14756",
        config_hash="fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
        base_model_revision="main",
    )
    assert isinstance(freeze_hash, str)
    assert len(freeze_hash) == 64

def test_verify_smoke_pass(tmp_path):
    import json
    report_file = tmp_path / "kaggle_smoke_report.json"

    # Missing file
    assert verify_smoke_pass(str(report_file)) is False

    # PASS status
    report_file.write_text(json.dumps({"status": "PASS", "profile": "kaggle_smoke_t4"}))
    assert verify_smoke_pass(str(report_file)) is True

    # FAIL status
    report_file.write_text(json.dumps({"status": "FAIL", "errors": ["OOM"]}))
    assert verify_smoke_pass(str(report_file)) is False
