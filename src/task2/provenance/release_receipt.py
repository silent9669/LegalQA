"""Strict release staging + immutable receipt (no self-hash recursion).

The pre-upload payload manifest records only the INTENDED repository/path.
The returned remote commit SHA and verification results belong to the
EXTERNAL post-upload receipt, stored outside the self-hashed release
directory and never retroactively inserted into the hashed bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from src.common.security import assert_no_secrets_in_workspace
from src.task2.config.loader import canonical_json_dumps, canonical_sha256

REQUIRED_RUN_KEYS = (
    "run_id",
    "candidate_id",
    "git_commit_sha",
    "algorithm_sha256",
    "seed",
)

# Fields that prove a receipt was hashed into its own payload (forbidden).
RECEIPT_CONTAMINANTS = ("external_receipt", "external_receipt_in_hash", "remote_commit_sha")

IMMUTABLE_PREFIX = "runs/"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_immutable_path(path_in_repo: str | None) -> str:
    """Require a per-run immutable HF path; never overwrite latest/best."""
    if not path_in_repo or not str(path_in_repo).strip():
        raise ValueError("release requires an explicit immutable path_in_repo")
    normalized = str(path_in_repo).strip().strip("/")
    if normalized in ("latest", "best", "main") or normalized.startswith(("latest/", "best/", "main/")):
        raise ValueError(f"refusing to overwrite pointer path: {path_in_repo}")
    if "/" not in normalized:
        raise ValueError(f"release path must be a per-run directory, got: {path_in_repo}")
    return normalized


def build_release_manifest(run: Dict[str, Any], artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the canonical pre-upload release manifest.

    run carries run/candidate/config/dataset/model/training/metric identity;
    artifacts is a list of {path, sha256, bytes} for staged files. Any
    receipt/remote-commit material in the inputs raises (receipt boundary).
    Returns the manifest including its self manifest_sha256.
    """
    for key in REQUIRED_RUN_KEYS:
        if not run.get(key):
            raise ValueError(f"release run record missing required key: {key}")
    for contaminant in RECEIPT_CONTAMINANTS:
        if contaminant in run:
            raise ValueError(f"receipt boundary violated: '{contaminant}' must stay outside the hashed release payload")
    staged = []
    for artifact in artifacts:
        for key in ("path", "sha256", "bytes"):
            if artifact.get(key) in (None, ""):
                raise ValueError(f"release artifact missing '{key}': {artifact}")
        for contaminant in RECEIPT_CONTAMINANTS:
            if contaminant in artifact:
                raise ValueError(f"receipt boundary violated in artifact '{artifact.get('path')}'")
        staged.append({"path": str(artifact["path"]), "sha256": str(artifact["sha256"]), "bytes": int(artifact["bytes"])})
    manifest = {
        "schema_version": 1,
        "run_id": str(run["run_id"]),
        "candidate_id": str(run["candidate_id"]),
        "git_commit_sha": str(run["git_commit_sha"]),
        "algorithm_sha256": str(run["algorithm_sha256"]),
        "seed": int(run["seed"]),
        "dataset": run.get("dataset", {}),
        "split_fingerprint": run.get("split_fingerprint", ""),
        "corpus_fingerprint": run.get("corpus_fingerprint", ""),
        "index_fingerprint": run.get("index_fingerprint", ""),
        "models": run.get("models", {}),
        "training": run.get("training", {}),
        "metrics": run.get("metrics", {}),
        "official_public_score": run.get("official_public_score"),
        "submission": run.get("submission", {}),
        "gate_reports": run.get("gate_reports", {}),
        "intended_repository": run.get("intended_repository", ""),
        "intended_path_in_repo": run.get("intended_path_in_repo", ""),
        "artifacts": sorted(staged, key=lambda a: a["path"]),
    }
    if manifest["intended_path_in_repo"]:
        require_immutable_path(manifest["intended_path_in_repo"])
    manifest["manifest_sha256"] = canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    return manifest


def _recomputed_self_hash(manifest: Dict[str, Any]) -> str:
    payload = {k: v for k, v in manifest.items() if k not in ("manifest_sha256", *RECEIPT_CONTAMINANTS)}
    return canonical_sha256(payload)


def verify_release_manifest(manifest: Dict[str, Any], remote: Dict[str, Any] | None) -> Dict[str, Any]:
    """Verify a staged manifest, optionally against remote file metadata.

    - Any receipt material inside the hashed payload raises ValueError.
    - manifest_sha256 must match the recomputed self hash.
    - remote=None -> STAGED (local manifest valid, not yet published).
    - remote={commit_sha, files:[{path,size,sha256}]} -> size/digest
      comparison for every staged artifact; all must match for
      PUBLISH_VERIFIED, else ValueError. Missing remote file info raises.
    """
    if manifest.get("external_receipt_in_hash") or "external_receipt" in manifest or "remote_commit_sha" in manifest:
        raise ValueError("receipt boundary violated: external receipt must stay outside the payload self-hash")
    expected = manifest.get("manifest_sha256")
    if not expected or _recomputed_self_hash(manifest) != expected:
        raise ValueError("release manifest self-hash mismatch")
    staged = manifest.get("artifacts", [])
    if remote is None:
        return {"status": "STAGED", "manifest_sha256": expected, "num_artifacts": len(staged)}
    commit_sha = remote.get("commit_sha")
    if not commit_sha or len(str(commit_sha)) < 7:
        raise ValueError("remote receipt missing commit_sha")
    remote_files = {(f.get("path")): f for f in remote.get("files", [])}
    mismatches = []
    for artifact in staged:
        info = remote_files.get(artifact["path"])
        if info is None:
            mismatches.append(f"missing remote file: {artifact['path']}")
            continue
        if int(info.get("size", -1)) != int(artifact["bytes"]):
            mismatches.append(f"size mismatch: {artifact['path']}")
        if str(info.get("sha256", "")).lower() != str(artifact["sha256"]).lower():
            mismatches.append(f"digest mismatch: {artifact['path']}")
    if mismatches:
        raise ValueError(f"remote verification failed: {mismatches[:5]}")
    return {
        "status": "PUBLISH_VERIFIED",
        "manifest_sha256": expected,
        "commit_sha": str(commit_sha),
        "num_artifacts": len(staged),
    }


def stage_release(bundle_dir: Union[str, Path], manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize, hash, and stage an immutable release inside bundle_dir.

    Runs the secret scan BEFORE hashing, writes release_manifest.json, and
    refreshes checksums.sha256. The external receipt is never written here.
    """
    root = Path(bundle_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"release bundle directory not found: {root}")
    assert_no_secrets_in_workspace(root, exclude_tests=False)
    verify_release_manifest(manifest, None)
    manifest_path = root / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    from src.task2.provenance.checksums import write_checksums_file

    checksums = write_checksums_file(root, root / "checksums.sha256")
    return {
        "status": "STAGED",
        "bundle_dir": str(root),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_path": str(manifest_path),
        "num_files": len(checksums),
    }
