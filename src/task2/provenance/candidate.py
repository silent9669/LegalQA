"""Candidate manifest schema and serialization for Task 2."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Union

from src.task2.config.loader import canonical_json_dumps, canonical_sha256

# Immutable 40-hex git commit SHAs only. Branch names, tags, short SHAs
# and policy strings are floating references and must never pin a release.
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_REQUIRED_MODEL_KEYS = ("generator", "reranker", "dense")


def validate_model_revisions(models: Dict[str, Dict[str, str]]) -> None:
    """Validate that every required model pins an immutable commit revision.

    Raises ValueError mentioning "immutable" for missing entries, missing
    revision fields, or any floating/short/non-hex revision.
    """
    if not isinstance(models, dict):
        raise ValueError("immutable model revisions: expected dict keyed by generator/reranker/dense")
    for key in _REQUIRED_MODEL_KEYS:
        entry = models.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"immutable model revision missing for '{key}'")
        revision = entry.get("revision")
        if not revision or not isinstance(revision, str):
            raise ValueError(f"immutable model revision missing for '{key}': no revision pin")
        if not _COMMIT_SHA_RE.match(revision.strip().lower()):
            raise ValueError(
                f"immutable model revision required for '{key}': got floating/non-commit revision '{revision}'"
            )
        model_id = entry.get("id")
        if not model_id or not isinstance(model_id, str):
            raise ValueError(f"immutable model revision for '{key}' requires a model id")


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
    modal_a100: str = ""


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
    bundle_sha256_by_profile: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_candidate_id(self) -> str:
        """Compute deterministic 16-hex character candidate ID over canonical identity payload."""
        if self.schema_version >= 2:
            payload = {
                "schema_version": self.schema_version,
                "task": self.task,
                "git_repository": self.git_repository,
                "git_commit_sha": self.git_commit_sha,
                "dataset": asdict(self.dataset),
                "algorithm_sha256": self.algorithm_sha256,
                "runtime_profile_sha256": asdict(self.runtime_profile_sha256),
                "config_bundle_sha256": self.config_bundle_sha256,
                "bundle_sha256_by_profile": dict(self.bundle_sha256_by_profile),
                "models": asdict(self.models),
                "dependency_lock_sha256": self.dependency_lock_sha256,
                "seed": self.seed,
                "created_at_utc": self.created_at_utc,
            }
        else:
            payload = {
                "schema_version": self.schema_version,
                "task": self.task,
                "git_repository": self.git_repository,
                "git_commit_sha": self.git_commit_sha,
                "dataset": asdict(self.dataset),
                "algorithm_sha256": self.algorithm_sha256,
                "runtime_profile_sha256": {
                    "kaggle_t4x2": self.runtime_profile_sha256.kaggle_t4x2,
                    "colab_t4": self.runtime_profile_sha256.colab_t4,
                    "colab_a100": self.runtime_profile_sha256.colab_a100,
                },
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
            bundle_sha256_by_profile=dict(self.bundle_sha256_by_profile),
        )

    def declared_runtime_hash(self, profile_name: str) -> str:
        """Return the declared runtime hash for a profile name, or empty string."""
        return getattr(self.runtime_profile_sha256, profile_name, "") or self.bundle_sha256_by_profile.get(
            profile_name + ":runtime", ""
        ) or ""

    def validate_against_config(self, cfg: Any) -> None:
        """Validate this candidate against a resolved algorithm+runtime config.

        Checks algorithm hash, seed, model ids, the declared runtime hash for
        the selected profile, and the recomputed per-run bundle hash binding.
        The per-run bundle_sha256 is algorithm + *selected* runtime, so it is
        expected to differ across platforms; each declared runtime binds to its
        own bundle via bundle_sha256_by_profile when present.
        """
        algo = cfg.algorithm
        runtime = cfg.runtime
        if self.algorithm_sha256 != cfg.algorithm_sha256:
            raise ValueError(
                f"candidate/config algorithm mismatch: candidate {self.algorithm_sha256} vs config {cfg.algorithm_sha256}"
            )
        if self.seed != algo.seed:
            raise ValueError(f"candidate/config seed mismatch: candidate {self.seed} vs config {algo.seed}")
        expected_ids = {
            "generator": algo.models.generator.id,
            "reranker": algo.models.reranker.id,
            "dense": algo.models.dense.id,
        }
        actual_ids = {
            "generator": self.models.generator.id,
            "reranker": self.models.reranker.id,
            "dense": self.models.dense.id,
        }
        if expected_ids != actual_ids:
            raise ValueError(f"candidate/config model id mismatch: {actual_ids} vs {expected_ids}")
        # Model revisions must be immutable pins, never floating refs.
        validate_model_revisions(
            {
                "generator": {"id": self.models.generator.id, "revision": self.models.generator.revision},
                "reranker": {"id": self.models.reranker.id, "revision": self.models.reranker.revision},
                "dense": {"id": self.models.dense.id, "revision": self.models.dense.revision},
            }
        )
        profile_name = runtime.profile_name
        declared = getattr(self.runtime_profile_sha256, profile_name, "")
        if not declared:
            raise ValueError(f"candidate does not declare runtime profile '{profile_name}'")
        if declared != cfg.runtime_sha256:
            raise ValueError(
                f"candidate/config runtime mismatch for '{profile_name}': candidate {declared} vs config {cfg.runtime_sha256}"
            )
        # Recomputed per-run bundle must match the resolved config's own bundle.
        recomputed = canonical_sha256({"algorithm": algo.to_dict(), "runtime": runtime.to_dict()})
        if recomputed != cfg.bundle_sha256:
            raise ValueError("resolved config bundle_sha256 does not match recomputed algorithm+runtime bundle")
        expected_bundle = self.bundle_sha256_by_profile.get(profile_name)
        if expected_bundle is not None and expected_bundle != cfg.bundle_sha256:
            raise ValueError(
                f"candidate/config bundle mismatch for '{profile_name}': candidate {expected_bundle} vs config {cfg.bundle_sha256}"
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
                modal_a100=str(rt_raw.get("modal_a100", "")),
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
            bundle_sha256_by_profile=dict(data.get("bundle_sha256_by_profile", {})),
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
    modal_a100_sha256: str = "",
    bundle_sha256_by_profile: Dict[str, str] | None = None,
    schema_version: int = 2,
) -> CandidateManifest:
    validate_model_revisions(
        {
            "generator": {"id": generator_id, "revision": generator_revision},
            "reranker": {"id": reranker_id, "revision": reranker_revision},
            "dense": {"id": dense_id, "revision": dense_revision},
        }
    )
    manifest = CandidateManifest(
        schema_version=schema_version,
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
            modal_a100=modal_a100_sha256,
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
        bundle_sha256_by_profile=dict(bundle_sha256_by_profile or {}),
    )
    return manifest.with_computed_id()
