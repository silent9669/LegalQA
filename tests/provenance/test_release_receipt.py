"""Task 9: strict release staging + immutable receipt regressions."""

from __future__ import annotations

import json

import pytest

from src.task2.provenance.release_receipt import (
    build_release_manifest,
    require_immutable_path,
    stage_release,
    verify_release_manifest,
)


def _run(**over):
    base = {
        "run_id": "run_abc_20260918",
        "candidate_id": "c" * 16,
        "git_commit_sha": "a" * 40,
        "algorithm_sha256": "b" * 64,
        "seed": 42,
        "intended_repository": "org/model",
        "intended_path_in_repo": "runs/run_abc_20260918",
    }
    base.update(over)
    return base


def _artifacts():
    return [
        {"path": "submission.json", "sha256": "0" * 64, "bytes": 12},
        {"path": "final_adapter/adapter_model.safetensors", "sha256": "1" * 64, "bytes": 34},
    ]


def test_receipt_self_hash_boundary_is_explicit():
    with pytest.raises(ValueError, match="receipt"):
        verify_release_manifest({"manifest_sha256": "x", "external_receipt_in_hash": True}, None)


def test_build_rejects_receipt_contamination_and_pointer_paths():
    bad = _run()
    bad["remote_commit_sha"] = "deadbeef"
    with pytest.raises(ValueError, match="receipt"):
        build_release_manifest(bad, _artifacts())
    with pytest.raises(ValueError, match="pointer"):
        require_immutable_path("latest")
    with pytest.raises(ValueError, match="immutable"):
        require_immutable_path("")
    assert require_immutable_path("runs/run_abc_20260918") == "runs/run_abc_20260918"


def test_require_immutable_path_rejects_pointer_variants():
    pointer_paths = [
        "latest",
        "best",
        "main",
        "latest/adapter",
        "best/model",
        "main/weights",
        "latest/runs/123",
        "best/final_checkpoint",
    ]
    for pointer in pointer_paths:
        with pytest.raises(ValueError, match="pointer"):
            require_immutable_path(pointer)

    invalid_empty_paths = ["", "   ", None]
    for empty in invalid_empty_paths:
        with pytest.raises(ValueError, match="explicit immutable path_in_repo"):
            require_immutable_path(empty)

    with pytest.raises(ValueError, match="per-run directory"):
        require_immutable_path("standalone_run_id_no_slash")

    valid_paths = [
        "runs/run_20260920_001",
        "experiments/exp_test_run",
        "checkpoints/final_run_42",
    ]
    for valid in valid_paths:
        assert require_immutable_path(valid) == valid



def test_remote_size_digest_mismatch_blocks_verification():
    manifest = build_release_manifest(_run(), _artifacts())
    assert verify_release_manifest(manifest, None)["status"] == "STAGED"
    with pytest.raises(ValueError, match="size mismatch"):
        verify_release_manifest(manifest, {
            "commit_sha": "c" * 40,
            "files": [
                {"path": "submission.json", "size": 999, "sha256": "0" * 64},
                {"path": "final_adapter/adapter_model.safetensors", "size": 34, "sha256": "1" * 64},
            ],
        })
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_release_manifest(manifest, {
            "commit_sha": "c" * 40,
            "files": [
                {"path": "submission.json", "size": 12, "sha256": "0" * 64},
                {"path": "final_adapter/adapter_model.safetensors", "size": 34, "sha256": "2" * 64},
            ],
        })
    ok = verify_release_manifest(manifest, {
        "commit_sha": "c" * 40,
        "files": [
            {"path": "submission.json", "size": 12, "sha256": "0" * 64},
            {"path": "final_adapter/adapter_model.safetensors", "size": 34, "sha256": "1" * 64},
        ],
    })
    assert ok["status"] == "PUBLISH_VERIFIED"


def test_stage_release_keeps_receipt_outside_payload(tmp_path):
    bundle = tmp_path / "run_abc"
    bundle.mkdir()
    (bundle / "submission.json").write_text('{"q1": {"answer": "a"}}', encoding="utf-8")
    manifest = build_release_manifest(_run(), [
        {"path": "submission.json", "sha256": "0" * 64, "bytes": (bundle / "submission.json").stat().st_size},
    ])
    staged = stage_release(bundle, manifest)
    assert staged["status"] == "STAGED"
    assert (bundle / "release_manifest.json").is_file()
    assert (bundle / "checksums.sha256").is_file()
    stored = json.loads((bundle / "release_manifest.json").read_text(encoding="utf-8"))
    assert "external_receipt" not in stored and "remote_commit_sha" not in stored


def test_missing_telemetry_cannot_yield_model_card_claim():
    from src.task2.provenance.run_bundle import build_production_run_bundle, generate_model_card

    with pytest.raises(ValueError, match="measured"):
        generate_model_card(object(), "run_x", {}, optimizer_steps=None, training_samples=2400)
    with pytest.raises(ValueError, match="measured"):
        build_production_run_bundle(
            run_id="run_x", candidate=object(), adapter_source_dir=".",
            kaggle_report_path=".", colab_t4_report_path=".", a100_micro_probe_report_path=".",
            train_log_path=".", output_dir=".", metrics={},
        )
