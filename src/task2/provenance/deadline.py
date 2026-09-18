"""Atomic checkpoints + deadline admission for the A100 runtime contract.

Every checkpoint manifest carries candidate id, code commit, model/data/split
fingerprints, RNG/optimizer/scheduler/sampler state, global step, elapsed
attempt time, retry count, and hardware versions. Resume increments the
persisted retry count and retains elapsed time; it never resets the deadline.
A graceful hard stop is INCOMPLETE unless the full eligible run finished.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Union

REQUIRED_MANIFEST_KEYS = (
    "candidate_id",
    "code_commit_sha",
    "model_revision",
    "data_hash",
    "split_fingerprint",
    "global_step",
    "elapsed_seconds",
    "retry_count",
    "optimizer_state",
    "scheduler_state",
    "sampler_position",
)

COMPLETE_MARKER = "COMPLETE"


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def allow_stage(stage: str, predicted_seconds: int, elapsed_seconds: int, budget_seconds: int) -> Dict[str, Any]:
    """Decide whether a stage may start within the remaining budget.

    Refuses with status INCOMPLETE when the pessimistic estimate
    (elapsed + predicted) exceeds the hard-stop budget. Monotonic clocks
    only: negative inputs raise.
    """
    for name, value in (("predicted_seconds", predicted_seconds), ("elapsed_seconds", elapsed_seconds), ("budget_seconds", budget_seconds)):
        if value is None or int(value) < 0:
            raise ValueError(f"allow_stage requires non-negative {name}, got {value}")
    if int(elapsed_seconds) + int(predicted_seconds) > int(budget_seconds):
        return {
            "status": "INCOMPLETE",
            "stage": stage,
            "reason": "deadline_exceeded",
            "elapsed_seconds": int(elapsed_seconds),
            "predicted_seconds": int(predicted_seconds),
            "budget_seconds": int(budget_seconds),
        }
    return {
        "status": "OK",
        "stage": stage,
        "elapsed_seconds": int(elapsed_seconds),
        "predicted_seconds": int(predicted_seconds),
        "budget_seconds": int(budget_seconds),
    }


def save_complete_checkpoint(
    root: Union[str, Path],
    stage: str,
    state: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Atomically persist one optimizer-complete checkpoint for a stage.

    Writes into a unique temp directory, fsyncs files and manifest, then
    atomically renames to <root>/<stage>. Only complete checkpoints (with
    all REQUIRED_MANIFEST_KEYS and a COMPLETE marker) are resumable;
    interrupted temp directories are never promoted.
    """
    if not stage or "/" in stage or stage in (".", ".."):
        raise ValueError(f"invalid stage name: {stage!r}")
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"incomplete checkpoint manifest, missing: {missing}")
    if not isinstance(state, dict) or not state:
        raise ValueError("complete checkpoint requires a nonempty state dict")

    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    manifest_record = dict(manifest)
    manifest_record["stage"] = stage
    state_hash = hashlib.sha256(_canonical(state)).hexdigest()
    manifest_record["state_sha256"] = state_hash
    manifest_json = _canonical(manifest_record)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f".tmp-{stage}-", dir=str(root_p)))
    try:
        state_path = tmp_dir / "state.json"
        manifest_path = tmp_dir / "manifest.json"
        state_path.write_bytes(_canonical(state))
        manifest_path.write_bytes(manifest_json)
        for path in (state_path, manifest_path):
            with open(path, "rb") as f:
                os.fsync(f.fileno())
        (tmp_dir / COMPLETE_MARKER).write_text(state_hash, encoding="utf-8")
        with open(tmp_dir / COMPLETE_MARKER, "rb") as f:
            os.fsync(f.fileno())
        try:
            fd = os.open(str(root_p), os.O_DIRECTORY)
        except OSError:
            fd = None
        target = root_p / stage
        os.rename(str(tmp_dir), str(target))
        if fd is not None:
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    except BaseException:
        import shutil

        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise
    manifest_record["path"] = str(root_p / stage)
    manifest_record["saved_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return manifest_record


def load_complete_checkpoint(root: Union[str, Path], stage: str) -> Dict[str, Any]:
    """Load an optimizer-complete checkpoint; interrupted ones are not resumable."""
    target = Path(root) / stage
    if not target.is_dir():
        # An interrupted temp dir is evidence of failure, never a resume source.
        leftovers = sorted(Path(root).glob(f".tmp-{stage}-*")) if Path(root).is_dir() else []
        if leftovers:
            raise FileNotFoundError(
                f"interrupted checkpoint for stage '{stage}' is not resumable: {leftovers[0]}"
            )
        raise FileNotFoundError(f"no complete checkpoint for stage '{stage}' in {root}")
    marker = target / COMPLETE_MARKER
    manifest_path = target / "manifest.json"
    state_path = target / "state.json"
    if not marker.is_file() or not manifest_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"incomplete checkpoint for stage '{stage}' is not resumable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if marker.read_text(encoding="utf-8").strip() != hashlib.sha256(_canonical(state)).hexdigest():
        raise ValueError(f"checkpoint state hash mismatch for stage '{stage}'")
    if manifest.get("state_sha256") != hashlib.sha256(_canonical(state)).hexdigest():
        raise ValueError(f"checkpoint manifest hash mismatch for stage '{stage}'")
    return {"manifest": manifest, "state": state, "path": str(target)}


def bump_retry(manifest: Dict[str, Any], elapsed_seconds: int) -> Dict[str, Any]:
    """Return a resumed manifest with monotonic retry count and elapsed time."""
    if int(elapsed_seconds) < int(manifest.get("elapsed_seconds", 0)):
        raise ValueError("resume cannot rewind elapsed attempt time")
    resumed = dict(manifest)
    resumed["retry_count"] = int(manifest.get("retry_count", 0)) + 1
    resumed["elapsed_seconds"] = int(elapsed_seconds)
    return resumed
