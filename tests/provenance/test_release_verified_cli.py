"""Tests verifying that scripts/release_verified.py preflight passes on a compliant mock bundle."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from src.task2.provenance.candidate import CandidateManifest, create_candidate_manifest
from src.task2.provenance.gate_report import (
    GateArtifacts,
    GateChecks,
    GateHardware,
    GateIdentity,
    GateMetrics,
    GateParentRef,
    GateReport,
)


def _make_candidate(tmp_path: Path) -> CandidateManifest:
    return create_candidate_manifest(
        git_commit_sha="a" * 40,
        dataset_slug="phucdangg/legalqa-task2-clean-data",
        dataset_version=1,
        dataset_manifest_sha256="d" * 64,
        algorithm_sha256="b" * 64,
        kaggle_t4x2_sha256="c" * 64,
        colab_t4_sha256="d" * 64,
        colab_a100_sha256="e" * 64,
        config_bundle_sha256="f" * 64,
        generator_id="Qwen/Qwen2.5-3B-Instruct",
        generator_revision="1" * 40,
        reranker_id="BAAI/bge-reranker-v2-m3",
        reranker_revision="2" * 40,
        dense_id="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        dense_revision="3" * 40,
        dependency_lock_sha256="4" * 64,
        seed=42,
        created_at_utc="2026-09-20T00:00:00Z",
    )


def _make_gate(stage: str, cid: str, git_sha: str, parent_sha: str | None = None) -> GateReport:
    return GateReport(
        schema_version=1,
        stage=stage,
        status="PASS",
        candidate_id=cid,
        started_at_utc="2026-09-20T00:00:00Z",
        finished_at_utc="2026-09-20T00:01:00Z",
        identity=GateIdentity(
            git_commit_sha=git_sha,
            dataset_slug="phucdangg/legalqa-task2-clean-data",
            dataset_version=1,
            dataset_manifest_sha256="d" * 64,
            algorithm_sha256="b" * 64,
            runtime_profile_sha256="c" * 64 if stage == "kaggle_t4x2" else ("d" * 64 if stage == "colab_t4" else "e" * 64),
            dependency_lock_sha256="4" * 64,
            generator_revision="1" * 40,
            reranker_revision="2" * 40,
            dense_revision="3" * 40,
        ),
        hardware=GateHardware(gpu_count=1, gpu_names=["A100"], torch_version="2.10", cuda_runtime="12.8", driver="580", peak_allocated_mb=100.0, peak_reserved_mb=200.0),
        checks=GateChecks(dataset_verified=True, config_verified=True, model_revisions_verified=True, finite_loss=True, trainable_weight_changed=True, checkpoint_saved=True, checkpoint_reloaded=True, mini_eval_completed=True),
        metrics=GateMetrics(optimizer_steps=166, seconds_per_step=1.0, meteor=0.52, rouge_l=0.55),
        artifacts=GateArtifacts(log_sha256="log", telemetry_sha256="tel", adapter_manifest_sha256="ad"),
        parent_gate=GateParentRef(stage="kaggle_t4x2" if stage == "colab_t4" else "colab_t4", report_sha256=parent_sha) if parent_sha else None,
    )


def test_release_verified_preflight_on_mock_bundle(tmp_path):
    bundle_dir = tmp_path / "mock_run_bundle"
    bundle_dir.mkdir(parents=True)
    gate_dir = bundle_dir / "gate_reports"
    gate_dir.mkdir()
    adapter_dir = bundle_dir / "final_adapter"
    adapter_dir.mkdir()

    # 1. Candidate manifest
    cand = _make_candidate(tmp_path)
    cand.save_json(bundle_dir / "candidate_manifest.json")

    # 2. Gate reports
    k_rep = _make_gate("kaggle_t4x2", cand.candidate_id, cand.git_commit_sha)
    k_rep.save_json(gate_dir / "kaggle_t4x2_report.json")
    c_rep = _make_gate("colab_t4", cand.candidate_id, cand.git_commit_sha, k_rep.compute_sha256())
    c_rep.save_json(gate_dir / "colab_t4_report.json")
    a_rep = _make_gate("a100_micro_probe", cand.candidate_id, cand.git_commit_sha, c_rep.compute_sha256())
    a_rep.save_json(gate_dir / "a100_micro_probe_report.json")

    # 3. Final adapter
    (adapter_dir / "generator_manifest.json").write_text(
        json.dumps({"smoke_only": False, "is_final_checkpoint": True, "dataset_size": 1325}), encoding="utf-8"
    )
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}), encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"dummy_weights")

    # 4. Submission files (10 queries for mock)
    sub = {f"q_{i}": {"answer": f"Answer {i}"} for i in range(10)}
    sub_path = bundle_dir / "submission.json"
    sub_path.write_text(json.dumps(sub, indent=2), encoding="utf-8")
    zip_path = bundle_dir / "submission.json.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_path, arcname="submission.json")

    # 5. Submission provenance
    prov = {
        "sources": {f"q_{i}": "generated" for i in range(10)},
        "counts": {"exact": 0, "fuzzy": 0, "generated": 10, "extractive": 0},
        "num_predictions": 10,
    }
    (bundle_dir / "submission_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    # 6. Production run manifest
    run_manifest = {
        "schema_version": 1,
        "run_id": "mock_run_001",
        "candidate_id": cand.candidate_id,
        "git_commit_sha": cand.git_commit_sha,
        "algorithm_sha256": cand.algorithm_sha256,
        "seed": 42,
        "training_sample_count": 1325,
        "optimizer_steps": 166,
        "num_train_epochs": 1,
        "effective_batch_size": 8,
        "parameter_audit": {"total_learned_parameters": 3_788_891_136},
        "submission": {"public_ids": list(sub.keys())},
        "huggingface": {"repository": "dangphuc2109/legalqa-qwen2.5-3b-adapter"},
    }
    (bundle_dir / "production_run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    # Run release_verified.py --bundle <bundle_dir> (without --publish)
    script_path = Path("scripts/release_verified.py").resolve()
    cmd = [sys.executable, str(script_path), "--bundle", str(bundle_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0, f"release_verified failed: {proc.stderr}\n{proc.stdout}"
    assert "[STAGED]" in proc.stdout
    assert "publish not requested; no upload performed" in proc.stdout

    # Verify release_manifest.json was written inside bundle
    rel_manifest_path = bundle_dir / "release_manifest.json"
    assert rel_manifest_path.is_file()
    rel_man = json.loads(rel_manifest_path.read_text(encoding="utf-8"))
    assert len(rel_man["manifest_sha256"]) == 64

    # Verify receipt is NOT inside the bundle
    receipt_in_bundle = list(bundle_dir.glob("*.receipt.json"))
    assert len(receipt_in_bundle) == 0, f"Receipt must not be inside bundle: {receipt_in_bundle}"


def test_release_verified_fails_on_candidate_mismatch(tmp_path):
    bundle_dir = tmp_path / "mismatch_bundle"
    bundle_dir.mkdir(parents=True)
    gate_dir = bundle_dir / "gate_reports"
    gate_dir.mkdir()
    adapter_dir = bundle_dir / "final_adapter"
    adapter_dir.mkdir()

    cand = _make_candidate(tmp_path)
    cand.save_json(bundle_dir / "candidate_manifest.json")

    # Mismatched gate report candidate_id
    k_rep = _make_gate("kaggle_t4x2", "wrong_candidate_id", cand.git_commit_sha)
    k_rep.save_json(gate_dir / "kaggle_t4x2_report.json")
    c_rep = _make_gate("colab_t4", cand.candidate_id, cand.git_commit_sha, k_rep.compute_sha256())
    c_rep.save_json(gate_dir / "colab_t4_report.json")
    a_rep = _make_gate("a100_micro_probe", cand.candidate_id, cand.git_commit_sha, c_rep.compute_sha256())
    a_rep.save_json(gate_dir / "a100_micro_probe_report.json")

    (adapter_dir / "generator_manifest.json").write_text(json.dumps({"smoke_only": False}), encoding="utf-8")
    sub = {"q_0": {"answer": "A"}}
    (bundle_dir / "submission.json").write_text(json.dumps(sub), encoding="utf-8")
    with zipfile.ZipFile(bundle_dir / "submission.json.zip", "w") as z:
        z.write(bundle_dir / "submission.json", arcname="submission.json")

    run_manifest = {
        "schema_version": 1,
        "run_id": "run_fail",
        "candidate_id": cand.candidate_id,
        "git_commit_sha": cand.git_commit_sha,
        "algorithm_sha256": cand.algorithm_sha256,
        "seed": 42,
        "parameter_audit": {"total_learned_parameters": 100},
        "submission": {"public_ids": ["q_0"]},
        "huggingface": {"repository": "org/repo"},
    }
    (bundle_dir / "production_run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")

    cmd = [sys.executable, str(Path("scripts/release_verified.py").resolve()), "--bundle", str(bundle_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "candidate mismatch" in proc.stderr or "candidate mismatch" in proc.stdout


def test_release_verified_fails_on_param_budget_exceeded(tmp_path):
    bundle_dir = tmp_path / "budget_bundle"
    bundle_dir.mkdir(parents=True)
    gate_dir = bundle_dir / "gate_reports"
    gate_dir.mkdir()
    adapter_dir = bundle_dir / "final_adapter"
    adapter_dir.mkdir()

    cand = _make_candidate(tmp_path)
    cand.save_json(bundle_dir / "candidate_manifest.json")

    k_rep = _make_gate("kaggle_t4x2", cand.candidate_id, cand.git_commit_sha)
    k_rep.save_json(gate_dir / "kaggle_t4x2_report.json")
    c_rep = _make_gate("colab_t4", cand.candidate_id, cand.git_commit_sha, k_rep.compute_sha256())
    c_rep.save_json(gate_dir / "colab_t4_report.json")
    a_rep = _make_gate("a100_micro_probe", cand.candidate_id, cand.git_commit_sha, c_rep.compute_sha256())
    a_rep.save_json(gate_dir / "a100_micro_probe_report.json")

    (adapter_dir / "generator_manifest.json").write_text(json.dumps({"smoke_only": False}), encoding="utf-8")
    sub = {"q_0": {"answer": "A"}}
    (bundle_dir / "submission.json").write_text(json.dumps(sub), encoding="utf-8")
    with zipfile.ZipFile(bundle_dir / "submission.json.zip", "w") as z:
        z.write(bundle_dir / "submission.json", arcname="submission.json")

    # Exceed 4B budget: 4,000,000,001
    run_manifest = {
        "schema_version": 1,
        "run_id": "run_budget_fail",
        "candidate_id": cand.candidate_id,
        "git_commit_sha": cand.git_commit_sha,
        "algorithm_sha256": cand.algorithm_sha256,
        "seed": 42,
        "parameter_audit": {"total_learned_parameters": 4_000_000_001},
        "submission": {"public_ids": ["q_0"]},
        "huggingface": {"repository": "org/repo"},
    }
    (bundle_dir / "production_run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")

    cmd = [sys.executable, str(Path("scripts/release_verified.py").resolve()), "--bundle", str(bundle_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "parameter budget exceeded" in proc.stderr or "parameter budget exceeded" in proc.stdout


def test_release_verified_publish_remote_verification_states(tmp_path, monkeypatch):
    import scripts.release_verified as rv

    bundle_dir = tmp_path / "publish_bundle"
    bundle_dir.mkdir(parents=True)
    gate_dir = bundle_dir / "gate_reports"
    gate_dir.mkdir()
    adapter_dir = bundle_dir / "final_adapter"
    adapter_dir.mkdir()

    cand = _make_candidate(tmp_path)
    cand.save_json(bundle_dir / "candidate_manifest.json")

    k_rep = _make_gate("kaggle_t4x2", cand.candidate_id, cand.git_commit_sha)
    k_rep.save_json(gate_dir / "kaggle_t4x2_report.json")
    c_rep = _make_gate("colab_t4", cand.candidate_id, cand.git_commit_sha, k_rep.compute_sha256())
    c_rep.save_json(gate_dir / "colab_t4_report.json")
    a_rep = _make_gate("a100_micro_probe", cand.candidate_id, cand.git_commit_sha, c_rep.compute_sha256())
    a_rep.save_json(gate_dir / "a100_micro_probe_report.json")

    (adapter_dir / "generator_manifest.json").write_text(json.dumps({"smoke_only": False}), encoding="utf-8")
    sub = {"q_0": {"answer": "A"}}
    (bundle_dir / "submission.json").write_text(json.dumps(sub), encoding="utf-8")
    with zipfile.ZipFile(bundle_dir / "submission.json.zip", "w") as z:
        z.write(bundle_dir / "submission.json", arcname="submission.json")

    run_manifest = {
        "schema_version": 1,
        "run_id": "run_pub_test",
        "candidate_id": cand.candidate_id,
        "git_commit_sha": cand.git_commit_sha,
        "algorithm_sha256": cand.algorithm_sha256,
        "seed": 42,
        "parameter_audit": {"total_learned_parameters": 100},
        "submission": {"public_ids": ["q_0"]},
        "huggingface": {"repository": "mock_org/mock_repo"},
    }
    (bundle_dir / "production_run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")

    receipt_path = tmp_path / "external_receipt.json"

    # Case 1: Remote verification succeeds -> PUBLISH_VERIFIED
    monkeypatch.setattr("src.task2.hf_uploader.upload_run_bundle_to_hf", lambda **kwargs: {"commit_sha": "f" * 40})
    monkeypatch.setattr(rv, "_confirm_remote_bytes", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "release_verified.py",
        "--bundle", str(bundle_dir),
        "--publish",
        "--receipt-out", str(receipt_path),
    ])

    rv.main()
    assert receipt_path.is_file()
    rec_data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert rec_data["status"] == "PUBLISH_VERIFIED"
    assert rec_data["remote_revision"] == "f" * 40

    # Case 2: Remote bytes mismatch -> PUBLISH_UNVERIFIED (never PASS)
    def fail_remote_bytes(*args, **kwargs):
        raise ValueError("simulated remote digest mismatch")

    monkeypatch.setattr(rv, "_confirm_remote_bytes", fail_remote_bytes)
    with pytest.raises(ValueError, match="PUBLISH_UNVERIFIED"):
        rv.main()

    rec_fail = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert rec_fail["status"] == "PUBLISH_UNVERIFIED"
    assert "simulated remote digest mismatch" in rec_fail["verification_error"]


def test_release_verified_two_gate_modal_chain(tmp_path):
    bundle_dir = tmp_path / "modal_chain_bundle"
    bundle_dir.mkdir(parents=True)
    gate_dir = bundle_dir / "gate_reports"
    gate_dir.mkdir()
    adapter_dir = bundle_dir / "final_adapter"
    adapter_dir.mkdir()

    cand = _make_candidate(tmp_path)
    cand.save_json(bundle_dir / "candidate_manifest.json")

    # Direct Modal DAG: kaggle_t4x2 -> a100_micro_probe (no colab_t4)
    k_rep = _make_gate("kaggle_t4x2", cand.candidate_id, cand.git_commit_sha)
    k_rep.save_json(gate_dir / "kaggle_t4x2_report.json")
    a_rep = _make_gate("a100_micro_probe", cand.candidate_id, cand.git_commit_sha, k_rep.compute_sha256())
    a_rep.save_json(gate_dir / "a100_micro_probe_report.json")

    (adapter_dir / "generator_manifest.json").write_text(json.dumps({"smoke_only": False}), encoding="utf-8")
    sub = {"q_0": {"answer": "A"}}
    (bundle_dir / "submission.json").write_text(json.dumps(sub), encoding="utf-8")
    with zipfile.ZipFile(bundle_dir / "submission.json.zip", "w") as z:
        z.write(bundle_dir / "submission.json", arcname="submission.json")

    run_manifest = {
        "schema_version": 1,
        "run_id": "run_modal_chain",
        "candidate_id": cand.candidate_id,
        "git_commit_sha": cand.git_commit_sha,
        "algorithm_sha256": cand.algorithm_sha256,
        "seed": 42,
        "parameter_audit": {"total_learned_parameters": 100},
        "submission": {"public_ids": ["q_0"]},
        "huggingface": {"repository": "mock_org/mock_repo"},
    }
    (bundle_dir / "production_run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")

    cmd = [sys.executable, str(Path("scripts/release_verified.py").resolve()), "--bundle", str(bundle_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"Failed on two-gate Modal chain: {proc.stderr}\n{proc.stdout}"
    assert "[STAGED]" in proc.stdout



