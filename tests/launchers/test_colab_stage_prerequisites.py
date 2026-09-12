import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from scripts.launch_colab_training import ColabLauncher
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
            gpu_count=2 if stage == "kaggle_t4x2" else 1,
            gpu_names=["Tesla T4"] * (2 if stage == "kaggle_t4x2" else 1),
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
        parent_gate=GateParentRef(stage="kaggle_t4x2", report_sha256=parent_sha) if parent_sha else None,
    )


def test_colab_t4_preflight_fails_on_missing_kaggle_report(tmp_path):
    """colab-t4 must fail preflight if Kaggle Dual-T4 report is absent."""
    cand = sample_candidate()
    cand_path = tmp_path / "candidate_manifest.json"
    cand.save_json(cand_path)

    launcher = ColabLauncher(
        stage="colab-t4",
        candidate_path=str(cand_path),
        kaggle_report_path=str(tmp_path / "non_existent_kaggle_report.json"),
        env_file=str(tmp_path / ".env"),
        skip_ci_check=True,
    )

    with pytest.raises(FileNotFoundError, match="Required Kaggle T4x2 gate report missing"):
        launcher._preflight_checks()


def test_colab_a100_preflight_fails_on_missing_colab_t4_report(tmp_path):
    """a100 stage must fail preflight if Colab T4 report is absent."""
    cand = sample_candidate()
    cand_path = tmp_path / "candidate_manifest.json"
    cand.save_json(cand_path)

    k_rep = sample_gate_report("kaggle_t4x2", cand.candidate_id, cand.git_commit_sha)
    k_path = tmp_path / "kaggle_t4x2_report.json"
    k_rep.save_json(k_path)

    launcher = ColabLauncher(
        stage="a100",
        candidate_path=str(cand_path),
        kaggle_report_path=str(k_path),
        colab_t4_report_path=str(tmp_path / "non_existent_colab_t4_report.json"),
        env_file=str(tmp_path / ".env"),
        skip_ci_check=True,
    )

    with pytest.raises(FileNotFoundError, match="Required Colab T4 gate report missing"):
        launcher._preflight_checks()


def test_colab_a100_preflight_passes_with_valid_chain(tmp_path):
    """a100 stage passes preflight when full gate chain is valid and matches candidate."""
    cand = sample_candidate()
    cand_path = tmp_path / "candidate_manifest.json"
    cand.save_json(cand_path)

    k_rep = sample_gate_report("kaggle_t4x2", cand.candidate_id, cand.git_commit_sha)
    k_path = tmp_path / "kaggle_t4x2_report.json"
    k_rep.save_json(k_path)
    k_sha = k_rep.compute_sha256()

    c_rep = sample_gate_report("colab_t4", cand.candidate_id, cand.git_commit_sha, parent_sha=k_sha)
    c_path = tmp_path / "colab_t4_report.json"
    c_rep.save_json(c_path)

    env_p = tmp_path / ".env"
    env_p.write_text("HF_TOKEN=hf_mocktoken12345678901234567890\nKAGGLE_KEY=mock_kaggle_key\nKAGGLE_USERNAME=mock_user\n")

    launcher = ColabLauncher(
        stage="a100",
        candidate_path=str(cand_path),
        kaggle_report_path=str(k_path),
        colab_t4_report_path=str(c_path),
        env_file=str(env_p),
        skip_ci_check=True,
    )

    # Should execute without raising
    launcher._preflight_checks()
    assert launcher.candidate.candidate_id == cand.candidate_id
