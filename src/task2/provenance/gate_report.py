"""Gate report schema, canonical serialization, and strict verifier."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.task2.config.loader import canonical_json_dumps, canonical_sha256
from src.task2.provenance.candidate import CandidateManifest


@dataclass(frozen=True)
class GateIdentity:
    git_commit_sha: str
    dataset_slug: str
    dataset_version: int
    dataset_manifest_sha256: str
    algorithm_sha256: str
    runtime_profile_sha256: str
    dependency_lock_sha256: str
    generator_revision: str
    reranker_revision: str
    dense_revision: str


@dataclass(frozen=True)
class GateHardware:
    gpu_count: int
    gpu_names: List[str]
    torch_version: str
    cuda_runtime: str
    driver: str
    peak_allocated_mb: float
    peak_reserved_mb: float


@dataclass(frozen=True)
class GateChecks:
    dataset_verified: bool
    config_verified: bool
    model_revisions_verified: bool
    finite_loss: bool
    trainable_weight_changed: bool
    checkpoint_saved: bool
    checkpoint_reloaded: bool
    mini_eval_completed: bool


@dataclass(frozen=True)
class GateMetrics:
    optimizer_steps: int
    seconds_per_step: float
    meteor: float
    rouge_l: float


@dataclass(frozen=True)
class GateArtifacts:
    log_sha256: str
    telemetry_sha256: str
    adapter_manifest_sha256: str


@dataclass(frozen=True)
class GateParentRef:
    stage: str
    report_sha256: str


@dataclass(frozen=True)
class GateReport:
    schema_version: int
    stage: str
    status: str
    candidate_id: str
    started_at_utc: str
    finished_at_utc: str
    identity: GateIdentity
    hardware: GateHardware
    checks: GateChecks
    metrics: GateMetrics
    artifacts: GateArtifacts
    parent_gate: Optional[GateParentRef] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_sha256(self) -> str:
        """Compute canonical SHA256 of the gate report content."""
        return canonical_sha256(self.to_dict())

    def save_json(self, path: Union[Path, str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        p.write_text(content, encoding="utf-8")

    @classmethod
    def load_json(cls, path: Union[Path, str]) -> GateReport:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Gate report not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))

        parent_ref = None
        if data.get("parent_gate"):
            parent_ref = GateParentRef(**data["parent_gate"])

        return cls(
            schema_version=int(data["schema_version"]),
            stage=str(data["stage"]),
            status=str(data["status"]),
            candidate_id=str(data["candidate_id"]),
            started_at_utc=str(data["started_at_utc"]),
            finished_at_utc=str(data["finished_at_utc"]),
            identity=GateIdentity(**data["identity"]),
            hardware=GateHardware(**data["hardware"]),
            checks=GateChecks(**data["checks"]),
            metrics=GateMetrics(**data["metrics"]),
            artifacts=GateArtifacts(**data["artifacts"]),
            parent_gate=parent_ref,
        )


def verify_gate_report(
    report_path: Union[Path, str],
    candidate: CandidateManifest,
    expected_stage: Union[str, Tuple[str, ...], List[str], Set[str]],
    required_parent_sha256: Optional[str] = None,
) -> GateReport:
    """Strictly verify a gate report against candidate manifest and expected promotion chain.

    expected_stage accepts one stage or a tuple of accepted stages (the
    migrated DAG lets the A100 microprobe chain under kaggle_t4x2 or
    colab_t4). A parent report is always required by the caller; this only
    checks identity, never authorization.
    """
    report = GateReport.load_json(report_path)

    if isinstance(expected_stage, (tuple, list, set)):
        if report.stage not in expected_stage:
            raise ValueError(f"Gate stage mismatch: expected one of {tuple(expected_stage)}, got {report.stage}")
    elif report.stage != expected_stage:
        raise ValueError(f"Gate stage mismatch: expected {expected_stage}, got {report.stage}")

    if report.status != "PASS":
        raise ValueError(f"Gate status is not PASS: {report.status}")

    if report.candidate_id != candidate.candidate_id:
        raise ValueError(
            f"Candidate ID mismatch in gate report: expected {candidate.candidate_id}, got {report.candidate_id}"
        )

    ident = report.identity
    if ident.git_commit_sha != candidate.git_commit_sha:
        raise ValueError(f"Git commit SHA mismatch: expected {candidate.git_commit_sha}, got {ident.git_commit_sha}")

    if ident.dataset_slug != candidate.dataset.slug:
        raise ValueError(f"Dataset slug mismatch: expected {candidate.dataset.slug}, got {ident.dataset_slug}")

    if ident.dataset_version != candidate.dataset.version:
        raise ValueError(f"Dataset version mismatch: expected {candidate.dataset.version}, got {ident.dataset_version}")

    if ident.dataset_manifest_sha256 != candidate.dataset.manifest_sha256:
        raise ValueError(
            f"Dataset manifest SHA256 mismatch: expected {candidate.dataset.manifest_sha256}, got {ident.dataset_manifest_sha256}"
        )

    if ident.algorithm_sha256 != candidate.algorithm_sha256:
        raise ValueError(f"Algorithm SHA256 mismatch: expected {candidate.algorithm_sha256}, got {ident.algorithm_sha256}")

    if ident.generator_revision != candidate.models.generator.revision:
        raise ValueError(
            f"Generator model revision mismatch: expected {candidate.models.generator.revision}, got {ident.generator_revision}"
        )

    if ident.reranker_revision != candidate.models.reranker.revision:
        raise ValueError(
            f"Reranker model revision mismatch: expected {candidate.models.reranker.revision}, got {ident.reranker_revision}"
        )

    if ident.dense_revision != candidate.models.dense.revision:
        raise ValueError(
            f"Dense model revision mismatch: expected {candidate.models.dense.revision}, got {ident.dense_revision}"
        )

    if ident.dependency_lock_sha256 != candidate.dependency_lock_sha256:
        raise ValueError(
            f"Dependency lock SHA256 mismatch: expected {candidate.dependency_lock_sha256}, got {ident.dependency_lock_sha256}"
        )

    # Verify parent gate report hash chaining
    if required_parent_sha256 is not None:
        if report.parent_gate is None:
            raise ValueError(
                f"Gate report {report.stage} missing required parent_gate record with expected SHA {required_parent_sha256}"
            )
        if report.parent_gate.report_sha256 != required_parent_sha256:
            raise ValueError(
                f"Parent gate report SHA256 mismatch: expected {required_parent_sha256}, got {report.parent_gate.report_sha256}"
            )

    # Fail closed on any check failure
    chk = report.checks
    failed_checks = [k for k, v in asdict(chk).items() if not v]
    if failed_checks:
        raise ValueError(f"Gate report contains failed checks: {failed_checks}")

    return report
