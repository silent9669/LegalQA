import json
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
    verify_gate_report,
)


def create_mock_candidate() -> CandidateManifest:
    manifest = CandidateManifest(
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
    return manifest.with_computed_id()


def test_full_candidate_gate_chain_verification(tmp_path):
    """Test full 3-gate cryptographic chain: Kaggle T4x2 -> Colab T4 -> A100 Micro-Probe."""
    cand = create_mock_candidate()

    # 1. Gate 2: Kaggle Dual-T4 Report
    k_report = GateReport(
        schema_version=1,
        stage="kaggle_t4x2",
        status="PASS",
        candidate_id=cand.candidate_id,
        started_at_utc="2026-09-12T01:00:00Z",
        finished_at_utc="2026-09-12T01:30:00Z",
        identity=GateIdentity(
            git_commit_sha=cand.git_commit_sha,
            dataset_slug=cand.dataset.slug,
            dataset_version=cand.dataset.version,
            dataset_manifest_sha256=cand.dataset.manifest_sha256,
            algorithm_sha256=cand.algorithm_sha256,
            runtime_profile_sha256=cand.runtime_profile_sha256.kaggle_t4x2,
            dependency_lock_sha256=cand.dependency_lock_sha256,
            generator_revision=cand.models.generator.revision,
            reranker_revision=cand.models.reranker.revision,
            dense_revision=cand.models.dense.revision,
        ),
        hardware=GateHardware(
            gpu_count=2,
            gpu_names=["Tesla T4", "Tesla T4"],
            torch_version="2.14.0",
            cuda_runtime="12.1",
            driver="535.104.05",
            peak_allocated_mb=12200.0,
            peak_reserved_mb=13000.0,
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
        metrics=GateMetrics(optimizer_steps=33, seconds_per_step=1.4, meteor=0.48, rouge_l=0.52),
        artifacts=GateArtifacts(log_sha256="k_log", telemetry_sha256="k_tel", adapter_manifest_sha256="k_ad"),
        parent_gate=None,
    )
    k_path = tmp_path / "kaggle_t4x2_report.json"
    k_report.save_json(k_path)
    k_sha = k_report.compute_sha256()

    verified_k = verify_gate_report(k_path, cand, "kaggle_t4x2")
    assert verified_k.status == "PASS"

    # 2. Gate 3: Colab Single-T4 Report (Requires Kaggle Report SHA)
    c_report = GateReport(
        schema_version=1,
        stage="colab_t4",
        status="PASS",
        candidate_id=cand.candidate_id,
        started_at_utc="2026-09-12T02:00:00Z",
        finished_at_utc="2026-09-12T02:10:00Z",
        identity=GateIdentity(
            git_commit_sha=cand.git_commit_sha,
            dataset_slug=cand.dataset.slug,
            dataset_version=cand.dataset.version,
            dataset_manifest_sha256=cand.dataset.manifest_sha256,
            algorithm_sha256=cand.algorithm_sha256,
            runtime_profile_sha256=cand.runtime_profile_sha256.colab_t4,
            dependency_lock_sha256=cand.dependency_lock_sha256,
            generator_revision=cand.models.generator.revision,
            reranker_revision=cand.models.reranker.revision,
            dense_revision=cand.models.dense.revision,
        ),
        hardware=GateHardware(
            gpu_count=1,
            gpu_names=["Tesla T4"],
            torch_version="2.14.0",
            cuda_runtime="12.1",
            driver="535.104.05",
            peak_allocated_mb=9800.0,
            peak_reserved_mb=10500.0,
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
        metrics=GateMetrics(optimizer_steps=5, seconds_per_step=1.2, meteor=0.48, rouge_l=0.51),
        artifacts=GateArtifacts(log_sha256="c_log", telemetry_sha256="c_tel", adapter_manifest_sha256="c_ad"),
        parent_gate=GateParentRef(stage="kaggle_t4x2", report_sha256=k_sha),
    )
    c_path = tmp_path / "colab_t4_report.json"
    c_report.save_json(c_path)
    c_sha = c_report.compute_sha256()

    verified_c = verify_gate_report(c_path, cand, "colab_t4", required_parent_sha256=k_sha)
    assert verified_c.status == "PASS"

    # 3. Gate 4: A100 Micro-Probe Report (Requires Colab T4 Report SHA)
    a_probe_report = GateReport(
        schema_version=1,
        stage="a100_micro_probe",
        status="PASS",
        candidate_id=cand.candidate_id,
        started_at_utc="2026-09-12T03:00:00Z",
        finished_at_utc="2026-09-12T03:05:00Z",
        identity=GateIdentity(
            git_commit_sha=cand.git_commit_sha,
            dataset_slug=cand.dataset.slug,
            dataset_version=cand.dataset.version,
            dataset_manifest_sha256=cand.dataset.manifest_sha256,
            algorithm_sha256=cand.algorithm_sha256,
            runtime_profile_sha256=cand.runtime_profile_sha256.colab_a100,
            dependency_lock_sha256=cand.dependency_lock_sha256,
            generator_revision=cand.models.generator.revision,
            reranker_revision=cand.models.reranker.revision,
            dense_revision=cand.models.dense.revision,
        ),
        hardware=GateHardware(
            gpu_count=1,
            gpu_names=["NVIDIA A100-SXM4-40GB"],
            torch_version="2.14.0",
            cuda_runtime="12.1",
            driver="535.104.05",
            peak_allocated_mb=18500.0,
            peak_reserved_mb=21000.0,
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
        artifacts=GateArtifacts(log_sha256="a_log", telemetry_sha256="a_tel", adapter_manifest_sha256="a_ad"),
        parent_gate=GateParentRef(stage="colab_t4", report_sha256=c_sha),
    )
    a_path = tmp_path / "a100_micro_probe_report.json"
    a_probe_report.save_json(a_path)

    verified_a = verify_gate_report(a_path, cand, "a100_micro_probe", required_parent_sha256=c_sha)
    assert verified_a.status == "PASS"

    # 4. Attempting to bypass Colab T4 gate directly to A100 fails
    with pytest.raises(ValueError, match="Parent gate report SHA256 mismatch"):
        verify_gate_report(a_path, cand, "a100_micro_probe", required_parent_sha256="forged_parent_sha")
