"""Provenance, candidate freezing, and gate verification for LegalQA Task 2."""

from src.task2.provenance.candidate import (
    CandidateManifest,
    DatasetRef,
    ModelRevisionRef,
    ModelsRef,
    RuntimeProfilesRef,
    create_candidate_manifest,
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
from src.task2.provenance.checksums import (
    compute_directory_checksums,
    compute_file_sha256,
    verify_checksums_file,
    write_checksums_file,
)
from src.task2.provenance.freeze_tuple import (
    build_run_manifest,
    compute_freeze_tuple_hash,
    verify_smoke_pass,
)

__all__ = [
    "CandidateManifest",
    "DatasetRef",
    "ModelRevisionRef",
    "ModelsRef",
    "RuntimeProfilesRef",
    "create_candidate_manifest",
    "GateArtifacts",
    "GateChecks",
    "GateHardware",
    "GateIdentity",
    "GateMetrics",
    "GateParentRef",
    "GateReport",
    "verify_gate_report",
    "compute_directory_checksums",
    "compute_file_sha256",
    "verify_checksums_file",
    "write_checksums_file",
    "build_run_manifest",
    "compute_freeze_tuple_hash",
    "verify_smoke_pass",
]
