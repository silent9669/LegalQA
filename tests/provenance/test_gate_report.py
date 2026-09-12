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


def sample_candidate() -> CandidateManifest:
    m = CandidateManifest(
        schema_version=1,
        candidate_id="",
        task="task2",
        git_repository="https://github.com/silent9669/LegalQA.git",
        git_commit_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        dataset=DatasetRef(
            slug="phucdangg/legalqa-task2-clean-data",
            version=1,
            manifest_sha256="1111111111111111111111111111111111111111111111111111111111111111",
        ),
        algorithm_sha256="2222222222222222222222222222222222222222222222222222222222222222",
        runtime_profile_sha256=RuntimeProfilesRef(
            kaggle_t4x2="3333333333333333333333333333333333333333333333333333333333333333",
            colab_t4="4444444444444444444444444444444444444444444444444444444444444444",
            colab_a100="5555555555555555555555555555555555555555555555555555555555555555",
        ),
        config_bundle_sha256="6666666666666666666666666666666666666666666666666666666666666666",
        models=ModelsRef(
            generator=ModelRevisionRef(id="Qwen/Qwen2.5-3B-Instruct", revision="rev_gen_123"),
            reranker=ModelRevisionRef(id="BAAI/bge-reranker-v2-m3", revision="rev_rerank_456"),
            dense=ModelRevisionRef(id="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", revision="rev_dense_789"),
        ),
        dependency_lock_sha256="7777777777777777777777777777777777777777777777777777777777777777",
        seed=42,
        created_at_utc="2026-09-12T12:00:00Z",
    )
    return m.with_computed_id()


def sample_kaggle_report(candidate: CandidateManifest) -> GateReport:
    return GateReport(
        schema_version=1,
        stage="kaggle_t4x2",
        status="PASS",
        candidate_id=candidate.candidate_id,
        started_at_utc="2026-09-12T12:10:00Z",
        finished_at_utc="2026-09-12T12:25:00Z",
        identity=GateIdentity(
            git_commit_sha=candidate.git_commit_sha,
            dataset_slug=candidate.dataset.slug,
            dataset_version=candidate.dataset.version,
            dataset_manifest_sha256=candidate.dataset.manifest_sha256,
            algorithm_sha256=candidate.algorithm_sha256,
            runtime_profile_sha256=candidate.runtime_profile_sha256.kaggle_t4x2,
            dependency_lock_sha256=candidate.dependency_lock_sha256,
            generator_revision=candidate.models.generator.revision,
            reranker_revision=candidate.models.reranker.revision,
            dense_revision=candidate.models.dense.revision,
        ),
        hardware=GateHardware(
            gpu_count=2,
            gpu_names=["Tesla T4", "Tesla T4"],
            torch_version="2.14.0",
            cuda_runtime="12.1",
            driver="535.104.05",
            peak_allocated_mb=12500.0,
            peak_reserved_mb=13200.0,
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
        metrics=GateMetrics(
            optimizer_steps=33,
            seconds_per_step=1.45,
            meteor=0.485,
            rouge_l=0.520,
        ),
        artifacts=GateArtifacts(
            log_sha256="aaa111",
            telemetry_sha256="bbb222",
            adapter_manifest_sha256="ccc333",
        ),
        parent_gate=None,
    )


def test_verify_gate_report_success(tmp_path):
    """Test successful verification of Kaggle report matching candidate."""
    cand = sample_candidate()
    rep = sample_kaggle_report(cand)
    rep_path = tmp_path / "kaggle_t4x2_report.json"
    rep.save_json(rep_path)

    verified = verify_gate_report(
        report_path=rep_path,
        candidate=cand,
        expected_stage="kaggle_t4x2",
    )
    assert verified.status == "PASS"
    assert verified.candidate_id == cand.candidate_id


def test_verify_gate_report_fails_on_status_not_pass(tmp_path):
    cand = sample_candidate()
    rep = sample_kaggle_report(cand)
    # Modify status to FAIL
    rep_dict = rep.to_dict()
    rep_dict["status"] = "FAIL"
    rep_path = tmp_path / "rep.json"
    rep_path.write_text(json.dumps(rep_dict))

    with pytest.raises(ValueError, match="Gate status is not PASS: FAIL"):
        verify_gate_report(rep_path, cand, "kaggle_t4x2")


def test_verify_gate_report_fails_on_candidate_mismatch(tmp_path):
    cand = sample_candidate()
    rep = sample_kaggle_report(cand)
    rep_dict = rep.to_dict()
    rep_dict["candidate_id"] = "different_candidate_id"
    rep_path = tmp_path / "rep.json"
    rep_path.write_text(json.dumps(rep_dict))

    with pytest.raises(ValueError, match="Candidate ID mismatch"):
        verify_gate_report(rep_path, cand, "kaggle_t4x2")


def test_verify_gate_report_fails_on_git_sha_mismatch(tmp_path):
    cand = sample_candidate()
    rep = sample_kaggle_report(cand)
    rep_dict = rep.to_dict()
    rep_dict["identity"]["git_commit_sha"] = "wrong_sha"
    rep_path = tmp_path / "rep.json"
    rep_path.write_text(json.dumps(rep_dict))

    with pytest.raises(ValueError, match="Git commit SHA mismatch"):
        verify_gate_report(rep_path, cand, "kaggle_t4x2")


def test_parent_gate_chain_verification(tmp_path):
    """Test Colab T4 gate requiring parent Kaggle report SHA256."""
    cand = sample_candidate()
    k_rep = sample_kaggle_report(cand)
    k_path = tmp_path / "kaggle_t4x2_report.json"
    k_rep.save_json(k_path)
    k_sha = k_rep.compute_sha256()

    colab_rep = GateReport(
        schema_version=1,
        stage="colab_t4",
        status="PASS",
        candidate_id=cand.candidate_id,
        started_at_utc="2026-09-12T13:00:00Z",
        finished_at_utc="2026-09-12T13:10:00Z",
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
        metrics=GateMetrics(
            optimizer_steps=5,
            seconds_per_step=1.2,
            meteor=0.48,
            rouge_l=0.51,
        ),
        artifacts=GateArtifacts(
            log_sha256="log123",
            telemetry_sha256="tel123",
            adapter_manifest_sha256="ad123",
        ),
        parent_gate=GateParentRef(
            stage="kaggle_t4x2",
            report_sha256=k_sha,
        ),
    )
    colab_path = tmp_path / "colab_t4_report.json"
    colab_rep.save_json(colab_path)

    # 1. Matching parent SHA passes
    verified = verify_gate_report(
        report_path=colab_path,
        candidate=cand,
        expected_stage="colab_t4",
        required_parent_sha256=k_sha,
    )
    assert verified.status == "PASS"

    # 2. Mismatched parent SHA fails
    with pytest.raises(ValueError, match="Parent gate report SHA256 mismatch"):
        verify_gate_report(
            report_path=colab_path,
            candidate=cand,
            expected_stage="colab_t4",
            required_parent_sha256="different_parent_sha",
        )
