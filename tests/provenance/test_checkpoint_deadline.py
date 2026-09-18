"""Task 7: checkpoint + deadline evidence regressions."""

from __future__ import annotations

import json

import pytest

from src.task2.checkpoint_manifest import save_checkpoint_manifest, verify_checkpoint_manifest
from src.task2.provenance.deadline import (
    allow_stage,
    bump_retry,
    load_complete_checkpoint,
    save_complete_checkpoint,
)


def _manifest(**over):
    base = {
        "candidate_id": "c" * 16,
        "code_commit_sha": "a" * 40,
        "model_revision": "Qwen/Qwen2.5-3B-Instruct@d8a1c8901eb4284d720235adcf8849767f40d7e4",
        "data_hash": "b" * 64,
        "split_fingerprint": "c" * 64,
        "global_step": 10,
        "elapsed_seconds": 100,
        "retry_count": 0,
        "optimizer_state": "opt",
        "scheduler_state": "sched",
        "sampler_position": 7,
    }
    base.update(over)
    return base


def test_deadline_denies_over_budget():
    assert allow_stage("train", 10, 95, 100)["status"] == "INCOMPLETE"


def test_deadline_admits_fitting_stage_and_rejects_negative():
    assert allow_stage("train", 10, 80, 100)["status"] == "OK"
    with pytest.raises(ValueError):
        allow_stage("train", -1, 0, 100)


def test_interrupted_checkpoint_is_not_resumable(tmp_path):
    staged = tmp_path / ".tmp-train-xyz"
    staged.mkdir()
    (staged / "state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="not resumable"):
        load_complete_checkpoint(tmp_path, "train")


def test_complete_checkpoint_roundtrip_and_retry_clock(tmp_path):
    saved = save_complete_checkpoint(tmp_path, "train", {"global_step": 10}, _manifest())
    assert saved["stage"] == "train"
    loaded = load_complete_checkpoint(tmp_path, "train")
    assert loaded["state"] == {"global_step": 10}
    assert loaded["manifest"]["retry_count"] == 0
    resumed = bump_retry(loaded["manifest"], elapsed_seconds=150)
    assert resumed["retry_count"] == 1 and resumed["elapsed_seconds"] == 150
    with pytest.raises(ValueError, match="rewind"):
        bump_retry(loaded["manifest"], elapsed_seconds=50)


def test_incomplete_manifest_rejected_before_write(tmp_path):
    bad = _manifest()
    del bad["optimizer_state"]
    with pytest.raises(ValueError, match="incomplete"):
        save_complete_checkpoint(tmp_path, "train", {"global_step": 1}, bad)
    assert not (tmp_path / "train").exists()


def test_checkpoint_manifest_file_digests(tmp_path):
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"fake-weights")
    record = save_checkpoint_manifest(str(tmp_path), {"files": ["adapter_model.safetensors"], "is_final_checkpoint": True})
    assert len(record["manifest_sha256"]) == 64
    assert verify_checkpoint_manifest(str(tmp_path))["is_final_checkpoint"] is True
    weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_checkpoint_manifest(str(tmp_path))
