import json
import os
import tempfile
from pathlib import Path
import pytest

from src.task2.provenance.candidate import (
    CandidateManifest,
    DatasetRef,
    ModelRevisionRef,
    ModelsRef,
    RuntimeProfilesRef,
)
from src.task2.provenance.gate_report import (
    GateArtifacts,
    GateChecks,
    GateHardware,
    GateIdentity,
    GateMetrics,
    GateParentRef,
    GateReport,
)
from src.task2.provenance.run_bundle import (
    build_production_run_bundle,
    verify_run_bundle,
    generate_model_card,
)


def sample_candidate() -> CandidateManifest:
    m = CandidateManifest(
        schema_version=1,
        candidate_id="",
        task="task2",
        git_repository="https://github.com/silent9669/LegalQA.git",
        git_commit_sha="c0ffee1234567890abcdef1234567890abcdef12",
        dataset=DatasetRef(
            slug="phucdangg/legalqa-task2-clean-data",
            version=1,
            manifest_sha256="1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
        ),
        algorithm_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        runtime_profile_sha256=RuntimeProfilesRef(
            kaggle_t4x2="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            colab_t4="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            colab_a100="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        ),
        config_bundle_sha256="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        models=ModelsRef(
            generator=ModelRevisionRef(id="Qwen/Qwen2.5-3B-Instruct", revision="rev1"),
            reranker=ModelRevisionRef(id="BAAI/bge-reranker-v2-m3", revision="rev2"),
            dense=ModelRevisionRef(id="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", revision="rev3"),
        ),
        dependency_lock_sha256="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        seed=42,
        created_at_utc="2026-09-12T00:00:00Z",
    )
    return m.with_computed_id()


def sample_gate_report(stage: str, candidate_id: str, git_sha: str, parent_sha: str = None) -> GateReport:
    return GateReport(
        schema_version=1,
        stage=stage,
        status="PASS",
        candidate_id=candidate_id,
        started_at_utc="2026-09-12T01:00:00Z",
        finished_at_utc="2026-09-12T01:30:00Z",
        identity=GateIdentity(
            git_commit_sha=git_sha,
            dataset_slug="phucdangg/legalqa-task2-clean-data",
            dataset_version=1,
            dataset_manifest_sha256="1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
            algorithm_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            runtime_profile_sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            dependency_lock_sha256="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            generator_revision="rev1",
            reranker_revision="rev2",
            dense_revision="rev3",
        ),
        hardware=GateHardware(
            gpu_count=1,
            gpu_names=["NVIDIA A100-SXM4-40GB"],
            torch_version="2.14.0",
            cuda_runtime="12.1",
            driver="535.104.05",
            peak_allocated_mb=18000.0,
            peak_reserved_mb=20000.0,
        ),
        checks=GateChecks(
            dataset_verified=True,
            config_verified=True,
            model_revisions_verified=True,
            finite_loss=True,
            trainable_weight_changed=True,
            checkpoint_saved=True,
            checkpoint_reloaded=True,
            mini_eval_completed=True,
        ),
        metrics=GateMetrics(optimizer_steps=2, seconds_per_step=0.6, meteor=0.49, rouge_l=0.53),
        artifacts=GateArtifacts(log_sha256="log_sha", telemetry_sha256="tel_sha", adapter_manifest_sha256="ad_sha"),
        parent_gate=GateParentRef(stage="colab_t4", report_sha256=parent_sha) if parent_sha else None,
    )


def test_build_production_run_bundle_structure(tmp_path):
    """Verify that build_production_run_bundle generates all mandatory files."""
    candidate = sample_candidate()
    run_id = f"run_{candidate.candidate_id}_20260912"
    output_bundle_dir = tmp_path / "runs" / run_id

    # Create dummy source inputs
    adapter_src = tmp_path / "src_adapter"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text(json.dumps({"lora_r": 16}))
    (adapter_src / "adapter_model.safetensors").write_bytes(b"dummy_weights")

    k_rep = sample_gate_report("kaggle_t4x2", candidate.candidate_id, candidate.git_commit_sha)
    c_rep = sample_gate_report("colab_t4", candidate.candidate_id, candidate.git_commit_sha, k_rep.compute_sha256())
    a_rep = sample_gate_report("a100_micro_probe", candidate.candidate_id, candidate.git_commit_sha, c_rep.compute_sha256())

    k_path = tmp_path / "k.json"
    c_path = tmp_path / "c.json"
    a_path = tmp_path / "a.json"
    k_rep.save_json(k_path)
    c_rep.save_json(c_path)
    a_rep.save_json(a_path)

    log_file = tmp_path / "train.log"
    log_file.write_text("Epoch 1/3\nStep 100/300 loss: 0.85\n")

    manifest = build_production_run_bundle(
        run_id=run_id,
        candidate=candidate,
        adapter_source_dir=adapter_src,
        kaggle_report_path=k_path,
        colab_t4_report_path=c_path,
        a100_micro_probe_report_path=a_path,
        train_log_path=log_file,
        output_dir=output_bundle_dir,
        metrics={"meteor": 0.495, "rouge_l": 0.531},
        optimizer_steps=300,
        training_sample_count=2400,
    )

    assert manifest["run_id"] == run_id
    assert manifest["candidate_id"] == candidate.candidate_id

    # Check that mandatory files exist
    assert (output_bundle_dir / "candidate_manifest.json").exists()
    assert (output_bundle_dir / "production_run_manifest.json").exists()
    assert (output_bundle_dir / "algorithm.resolved.json").exists()
    assert (output_bundle_dir / "runtime.resolved.json").exists()
    assert (output_bundle_dir / "gate_reports" / "kaggle_t4x2_report.json").exists()
    assert (output_bundle_dir / "gate_reports" / "colab_t4_report.json").exists()
    assert (output_bundle_dir / "gate_reports" / "a100_micro_probe_report.json").exists()
    assert (output_bundle_dir / "environment" / "python.txt").exists()
    assert (output_bundle_dir / "environment" / "pip-freeze.txt").exists()
    assert (output_bundle_dir / "logs" / "train.log").exists()
    assert (output_bundle_dir / "metrics.json").exists()
    assert (output_bundle_dir / "final_adapter" / "adapter_config.json").exists()
    assert (output_bundle_dir / "final_adapter" / "adapter_model.safetensors").exists()
    assert (output_bundle_dir / "model_card.md").exists()
    assert (output_bundle_dir / "checksums.sha256").exists()

    # Verify checksums
    assert verify_run_bundle(output_bundle_dir) is True


def test_build_bundle_fails_on_secret_leak(tmp_path):
    """Verify that bundle building fails if a secret token is detected in any bundle file."""
    candidate = sample_candidate()
    run_id = f"run_{candidate.candidate_id}_leaky"
    output_bundle_dir = tmp_path / "runs" / run_id

    adapter_src = tmp_path / "src_adapter"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text(json.dumps({"lora_r": 16}))
    (adapter_src / "adapter_model.safetensors").write_bytes(b"dummy_weights")

    k_rep = sample_gate_report("kaggle_t4x2", candidate.candidate_id, candidate.git_commit_sha)
    c_rep = sample_gate_report("colab_t4", candidate.candidate_id, candidate.git_commit_sha, k_rep.compute_sha256())
    a_rep = sample_gate_report("a100_micro_probe", candidate.candidate_id, candidate.git_commit_sha, c_rep.compute_sha256())

    k_path = tmp_path / "k.json"
    c_path = tmp_path / "c.json"
    a_path = tmp_path / "a.json"
    k_rep.save_json(k_path)
    c_rep.save_json(c_path)
    a_rep.save_json(a_path)

    # Leaky log file with a mock Hugging Face token constructed dynamically
    mock_token = "".join(["hf_", "mocksecrettoken1234567890abcdef"])
    leaky_log = tmp_path / "train.log"
    leaky_log.write_text(f"Training started with HF_TOKEN={mock_token}\n")

    with pytest.raises(RuntimeError, match="Secret scanner detected credentials"):
        build_production_run_bundle(
            run_id=run_id,
            candidate=candidate,
            adapter_source_dir=adapter_src,
            kaggle_report_path=k_path,
            colab_t4_report_path=c_path,
            a100_micro_probe_report_path=a_path,
            train_log_path=leaky_log,
            output_dir=output_bundle_dir,
            metrics={"meteor": 0.495},
            optimizer_steps=300,
            training_sample_count=2400,
        )


def test_build_production_run_bundle_modal_two_gate_chain(tmp_path):
    """Verify that build_production_run_bundle succeeds with direct kaggle_t4x2 -> a100_micro_probe DAG."""
    candidate = sample_candidate()
    run_id = f"run_modal_{candidate.candidate_id}"
    output_bundle_dir = tmp_path / "runs" / run_id

    adapter_src = tmp_path / "src_adapter_modal"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text(json.dumps({"lora_r": 16}))
    (adapter_src / "adapter_model.safetensors").write_bytes(b"weights_data")

    k_rep = sample_gate_report("kaggle_t4x2", candidate.candidate_id, candidate.git_commit_sha)
    # Direct parent is kaggle_t4x2
    a_rep = sample_gate_report("a100_micro_probe", candidate.candidate_id, candidate.git_commit_sha, k_rep.compute_sha256())

    k_path = tmp_path / "k_modal.json"
    a_path = tmp_path / "a_modal.json"
    k_rep.save_json(k_path)
    a_rep.save_json(a_path)

    log_file = tmp_path / "train_modal.log"
    log_file.write_text("Modal Step 166 loss: 0.70\n")

    manifest = build_production_run_bundle(
        run_id=run_id,
        candidate=candidate,
        adapter_source_dir=adapter_src,
        kaggle_report_path=k_path,
        colab_t4_report_path=None,
        a100_micro_probe_report_path=a_path,
        train_log_path=log_file,
        output_dir=output_bundle_dir,
        metrics={"meteor": 0.52, "rouge_l": 0.55},
        optimizer_steps=166,
        training_sample_count=1325,
        num_train_epochs=1,
    )

    assert manifest["run_id"] == run_id
    assert "colab_t4" not in manifest["gate_reports"]
    assert manifest["gate_reports"]["kaggle_t4x2"] == k_rep.compute_sha256()
    assert manifest["gate_reports"]["a100_micro_probe"] == a_rep.compute_sha256()
    assert verify_run_bundle(output_bundle_dir) is True


def test_build_production_run_bundle_modal_runtime_and_submission(tmp_path):
    """Verify that build_production_run_bundle binds modal_a100 runtime and seals submission files."""
    candidate = sample_candidate()
    run_id = f"run_modal_sub_{candidate.candidate_id}"
    output_bundle_dir = tmp_path / "runs" / run_id

    adapter_src = tmp_path / "src_adapter_sub"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text(json.dumps({"lora_r": 16}))
    (adapter_src / "adapter_model.safetensors").write_bytes(b"weights_with_submission")

    k_rep = sample_gate_report("kaggle_t4x2", candidate.candidate_id, candidate.git_commit_sha)
    a_rep = sample_gate_report("a100_micro_probe", candidate.candidate_id, candidate.git_commit_sha, k_rep.compute_sha256())

    k_path = tmp_path / "k_sub.json"
    a_path = tmp_path / "a_sub.json"
    k_rep.save_json(k_path)
    a_rep.save_json(a_path)

    log_file = tmp_path / "train_sub.log"
    log_file.write_text("Modal Step 100 loss: 0.50\n")

    sub_file = tmp_path / "submission.json"
    sub_file.write_text(json.dumps({"1": {"answer": "Answer 1"}}, indent=2))
    prov_file = tmp_path / "submission_provenance.json"
    prov_file.write_text(json.dumps({"1": {"source": "generated"}}, indent=2))

    manifest = build_production_run_bundle(
        run_id=run_id,
        candidate=candidate,
        adapter_source_dir=adapter_src,
        kaggle_report_path=k_path,
        colab_t4_report_path=None,
        a100_micro_probe_report_path=a_path,
        train_log_path=log_file,
        output_dir=output_bundle_dir,
        metrics={"meteor": 0.53, "rouge_l": 0.56},
        optimizer_steps=100,
        training_sample_count=800,
        num_train_epochs=1,
        runtime_profile="modal_a100",
        submission_path=sub_file,
        submission_provenance_path=prov_file,
    )

    assert manifest["run_id"] == run_id
    assert "submission" in manifest
    assert (output_bundle_dir / "submission.json").exists()
    assert (output_bundle_dir / "submission_provenance.json").exists()
    assert verify_run_bundle(output_bundle_dir) is True


def test_reuse_bundle_separates_source_training_metadata(tmp_path):
    """Reuse bundles record 0 new steps; source figures stay namespaced (P1)."""
    candidate = sample_candidate()
    run_id = f"run_reuse_{candidate.candidate_id}"
    output_bundle_dir = tmp_path / "runs" / run_id

    adapter_src = tmp_path / "src_adapter_reuse"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text(json.dumps({"lora_r": 16}))
    (adapter_src / "adapter_model.safetensors").write_bytes(b"reused_adapter_bytes")

    k_rep = sample_gate_report("kaggle_t4x2", candidate.candidate_id, candidate.git_commit_sha)
    a_rep = sample_gate_report("a100_micro_probe", candidate.candidate_id, candidate.git_commit_sha, k_rep.compute_sha256())
    k_path = tmp_path / "k_reuse.json"
    a_path = tmp_path / "a_reuse.json"
    k_rep.save_json(k_path)
    a_rep.save_json(a_path)

    source_adapter = {
        "repo": "dangphuc2109/legalqa-qwen2.5-3b-adapter",
        "revision": "b" * 40,
        "optimizer_steps": 936,
        "dataset_size": 7483,
    }
    manifest = build_production_run_bundle(
        run_id=run_id,
        candidate=candidate,
        adapter_source_dir=adapter_src,
        kaggle_report_path=k_path,
        colab_t4_report_path=None,
        a100_micro_probe_report_path=a_path,
        train_log_path=tmp_path / "missing_train.log",
        output_dir=output_bundle_dir,
        metrics={"meteor": None, "reason": "reuse run carries no new training telemetry"},
        optimizer_steps=0,
        training_sample_count=0,
        training_performed=False,
        source_adapter=source_adapter,
    )
    assert manifest["training_performed"] is False
    assert manifest["optimizer_steps"] == 0
    assert manifest["source_adapter"]["optimizer_steps"] == 936
    train_log = (output_bundle_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "training_performed=false" in train_log  # honest marker, not a fake training log
    assert verify_run_bundle(output_bundle_dir) is True


def test_reuse_bundle_without_source_metadata_refused(tmp_path):
    candidate = sample_candidate()
    with pytest.raises(ValueError, match="source_adapter"):
        build_production_run_bundle(
            run_id=f"run_reuse_bad_{candidate.candidate_id}",
            candidate=candidate,
            adapter_source_dir=tmp_path,
            kaggle_report_path=tmp_path / "missing.json",
            colab_t4_report_path=None,
            a100_micro_probe_report_path=tmp_path / "missing2.json",
            train_log_path=tmp_path / "missing.log",
            output_dir=tmp_path / "bad_bundle",
            metrics={},
            optimizer_steps=0,
            training_sample_count=0,
            training_performed=False,
            source_adapter=None,
        )


