"""Candidate manifest schema and serialization for Task 2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Union

from src.task2.config.loader import canonical_json_dumps, canonical_sha256


@dataclass(frozen=True)
class DatasetRef:
    slug: str
    version: int
    manifest_sha256: str


@dataclass(frozen=True)
class RuntimeProfilesRef:
    kaggle_t4x2: str
    colab_t4: str
    colab_a100: str


@dataclass(frozen=True)
class ModelRevisionRef:
    id: str
    revision: str


@dataclass(frozen=True)
class ModelsRef:
    generator: ModelRevisionRef
    reranker: ModelRevisionRef
    dense: ModelRevisionRef


@dataclass(frozen=True)
class CandidateManifest:
    schema_version: int
    candidate_id: str
    task: str
    git_repository: str
    git_commit_sha: str
    dataset: DatasetRef
    algorithm_sha256: str
    runtime_profile_sha256: RuntimeProfilesRef
    config_bundle_sha256: str
    models: ModelsRef
    dependency_lock_sha256: str
    seed: int
    created_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_candidate_id(self) -> str:
        """Compute deterministic 16-hex character candidate ID over canonical identity payload."""
        payload = {
            "schema_version": self.schema_version,
            "task": self.task,
            "git_repository": self.git_repository,
            "git_commit_sha": self.git_commit_sha,
            "dataset": asdict(self.dataset),
            "algorithm_sha256": self.algorithm_sha256,
            "runtime_profile_sha256": asdict(self.runtime_profile_sha256),
            "config_bundle_sha256": self.config_bundle_sha256,
            "models": asdict(self.models),
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "seed": self.seed,
            "created_at_utc": self.created_at_utc,
        }
        return canonical_sha256(payload)[:16]

    def with_computed_id(self) -> CandidateManifest:
        """Return a copy with computed candidate_id."""
        cid = self.compute_candidate_id()
        return CandidateManifest(
            schema_version=self.schema_version,
            candidate_id=cid,
            task=self.task,
            git_repository=self.git_repository,
            git_commit_sha=self.git_commit_sha,
            dataset=self.dataset,
            algorithm_sha256=self.algorithm_sha256,
            runtime_profile_sha256=self.runtime_profile_sha256,
            config_bundle_sha256=self.config_bundle_sha256,
            models=self.models,
            dependency_lock_sha256=self.dependency_lock_sha256,
            seed=self.seed,
            created_at_utc=self.created_at_utc,
        )

    def save_json(self, path: Union[Path, str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        p.write_text(content, encoding="utf-8")

    @classmethod
    def load_json(cls, path: Union[Path, str]) -> CandidateManifest:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Candidate manifest not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))

        ds_raw = data["dataset"]
        rt_raw = data["runtime_profile_sha256"]
        m_raw = data["models"]

        return cls(
            schema_version=int(data["schema_version"]),
            candidate_id=str(data["candidate_id"]),
            task=str(data["task"]),
            git_repository=str(data["git_repository"]),
            git_commit_sha=str(data["git_commit_sha"]),
            dataset=DatasetRef(
                slug=str(ds_raw["slug"]),
                version=int(ds_raw["version"]),
                manifest_sha256=str(ds_raw["manifest_sha256"]),
            ),
            algorithm_sha256=str(data["algorithm_sha256"]),
            runtime_profile_sha256=RuntimeProfilesRef(
                kaggle_t4x2=str(rt_raw["kaggle_t4x2"]),
                colab_t4=str(rt_raw["colab_t4"]),
                colab_a100=str(rt_raw["colab_a100"]),
            ),
            config_bundle_sha256=str(data["config_bundle_sha256"]),
            models=ModelsRef(
                generator=ModelRevisionRef(**m_raw["generator"]),
                reranker=ModelRevisionRef(**m_raw["reranker"]),
                dense=ModelRevisionRef(**m_raw["dense"]),
            ),
            dependency_lock_sha256=str(data["dependency_lock_sha256"]),
            seed=int(data["seed"]),
            created_at_utc=str(data["created_at_utc"]),
        )


def create_candidate_manifest(
    git_commit_sha: str,
    dataset_slug: str,
    dataset_version: int,
    dataset_manifest_sha256: str,
    algorithm_sha256: str,
    kaggle_t4x2_sha256: str,
    colab_t4_sha256: str,
    colab_a100_sha256: str,
    config_bundle_sha256: str,
    generator_id: str,
    generator_revision: str,
    reranker_id: str,
    reranker_revision: str,
    dense_id: str,
    dense_revision: str,
    dependency_lock_sha256: str,
    seed: int = 42,
    created_at_utc: str = "",
    git_repository: str = "https://github.com/silent9669/LegalQA.git",
) -> CandidateManifest:
    manifest = CandidateManifest(
        schema_version=1,
        candidate_id="",
        task="task2",
        git_repository=git_repository,
        git_commit_sha=git_commit_sha,
        dataset=DatasetRef(slug=dataset_slug, version=dataset_version, manifest_sha256=dataset_manifest_sha256),
        algorithm_sha256=algorithm_sha256,
        runtime_profile_sha256=RuntimeProfilesRef(
            kaggle_t4x2=kaggle_t4x2_sha256,
            colab_t4=colab_t4_sha256,
            colab_a100=colab_a100_sha256,
        ),
        config_bundle_sha256=config_bundle_sha256,
        models=ModelsRef(
            generator=ModelRevisionRef(id=generator_id, revision=generator_revision),
            reranker=ModelRevisionRef(id=reranker_id, revision=reranker_revision),
            dense=ModelRevisionRef(id=dense_id, revision=dense_revision),
        ),
        dependency_lock_sha256=dependency_lock_sha256,
        seed=seed,
        created_at_utc=created_at_utc,
    )
    return manifest.with_computed_id()
