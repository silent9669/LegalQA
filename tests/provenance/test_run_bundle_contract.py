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
