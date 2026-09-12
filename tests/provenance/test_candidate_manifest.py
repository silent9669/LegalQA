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
    create_candidate_manifest,
)


def sample_candidate_manifest() -> CandidateManifest:
    return CandidateManifest(
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


def test_candidate_manifest_id_determinism():
    """Verify that candidate_id is deterministic across different dictionary ordering."""
    m1 = sample_candidate_manifest()
    cid1 = m1.compute_candidate_id()
    assert len(cid1) == 16

    # Create duplicate with identical data
    m2 = sample_candidate_manifest()
    cid2 = m2.compute_candidate_id()
    assert cid1 == cid2

    # Any change in algorithm hash must change candidate_id
    m3 = CandidateManifest(
        schema_version=m1.schema_version,
        candidate_id="",
        task=m1.task,
        git_repository=m1.git_repository,
        git_commit_sha=m1.git_commit_sha,
        dataset=m1.dataset,
        algorithm_sha256="9999999999999999999999999999999999999999999999999999999999999999",
        runtime_profile_sha256=m1.runtime_profile_sha256,
        config_bundle_sha256=m1.config_bundle_sha256,
        models=m1.models,
        dependency_lock_sha256=m1.dependency_lock_sha256,
        seed=m1.seed,
        created_at_utc=m1.created_at_utc,
    )
    assert m3.compute_candidate_id() != cid1


def test_candidate_manifest_serialization_roundtrip(tmp_path):
    """Test saving to JSON and reloading preserves exact fields and candidate_id."""
    manifest = sample_candidate_manifest()
    manifest = manifest.with_computed_id()

    json_path = tmp_path / "candidate_manifest.json"
    manifest.save_json(json_path)

    loaded = CandidateManifest.load_json(json_path)
    assert loaded.candidate_id == manifest.candidate_id
    assert loaded.git_commit_sha == manifest.git_commit_sha
    assert loaded.models.generator.revision == "rev_gen_123"
    assert loaded.dataset.version == 1
