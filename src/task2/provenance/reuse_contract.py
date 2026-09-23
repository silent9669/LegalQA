"""Reuse-first contract helpers for LegalQA Task 2 (R0 baseline).

Pure, CPU-testable validators shared by ``scripts/modal_app.py`` (request
building + remote strict loading) and ``src/task2/pipeline/runner.py``
(Stage 4 skip-train gate). No GPU, no network, no fallback selection.

R0 baseline (to IMPLEMENT AND MEASURE, not a proven best score):
- candidate ``94aed6911490a9c3`` (config anchor only, not a full recipe id)
- Qwen adapter ``dangphuc2109/legalqa-qwen2.5-3b-adapter``
  subfolder ``runs/run_5433e8b4787137c9_20260920_193355/final_adapter/``
  pinned commit ``b6e86e35e20c403bb82b40b25f85690c987e1d02``
- base Qwen revision ``aa8e72537993ba99e69dfaafa59ed015b17504d1``

The 5433 adapter is the generator behind the ``runs/20260920-215402``
notebook outputs (which reused ``/vol/kaggle_data/lora`` instead of
training) and the ``run_v16_dual_assembled`` release manifest
(``candidate_id 5433e8b4787137c9``). Its manifest records 7.483
examples / 936 optimizer steps, final, full scope, null val fold.
No 0.56 *answer* score was found in the run artifacts — the 0.56x
figures present are retrieval-side (hit@12 0.562, ctx-recall); local
answer METEOR observed there is ~0.52. Do not attach an official or
>0.60 claim to this adapter.

File digests below were MEASURED 2026-09-23 from bytes downloaded at
the pinned revision (never invented); verification still re-hashes the
staged bytes and compares.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.task2.provenance.checksums import compute_file_sha256

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: R0 adapter identity (repo + immutable revision + subfolder). Digests are
#: NOT bundled here: they must be measured from pinned bytes and passed
#: explicitly in the request spec.
R0_ADAPTER_REPO = "dangphuc2109/legalqa-qwen2.5-3b-adapter"
R0_ADAPTER_REVISION = "b6e86e35e20c403bb82b40b25f85690c987e1d02"
R0_ADAPTER_SUBFOLDER = "runs/run_5433e8b4787137c9_20260920_193355/final_adapter"
R0_GENERATOR_BASE_ID = "Qwen/Qwen2.5-3B-Instruct"
R0_GENERATOR_BASE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
R0_ADAPTER_SCOPE = "all_allowed_task2_data"

#: Measured 2026-09-23 from pinned-revision bytes (59.934.640-byte weights;
#: weights SHA equals the git-LFS oid). Reference only: launch requests
#: must still carry explicit digests; verification re-hashes staged bytes.
R0_ADAPTER_FILE_DIGESTS = {
    "adapter_model.safetensors": "dd5af2848f23234e7bd7cb7987b0b199c6832a4f5fde5dbe59a3e21ee379484e",
    "adapter_config.json": "301cac83325dbc5e6a60d5bcf86c0ebb509e46f8f7e7c2a2837031690dc1256b",
    "generator_manifest.json": "1f46415bb5db5633e7bb6b7d4b0da27a3adc7d22b3e22ea775060978725e36a5",
}

#: Files whose digests bind the reused adapter bytes.
REQUIRED_ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "generator_manifest.json")

GENERATOR_MODES = ("reuse", "fresh")


def _norm_commit(value: Any, field: str) -> str:
    text = str(value or "").strip().lower()
    if not _COMMIT_RE.match(text):
        raise ValueError(f"adapter spec requires immutable 40-hex {field}, got {value!r}")
    return text


def _norm_sha256(value: Any, field: str) -> str:
    text = str(value or "").strip().lower()
    if not _SHA256_RE.match(text):
        raise ValueError(f"adapter spec requires 64-hex sha256 for {field}, got {value!r}")
    return text


def default_r0_adapter_identity() -> Dict[str, Any]:
    """Return the R0 adapter identity WITHOUT digests (not usable for reuse).

    Callers must measure file digests at the pinned revision and attach
    them via ``validate_adapter_spec``; this template exists only so the
    repo/revision/subfolder pins stay in one place.
    """
    return {
        "repo": R0_ADAPTER_REPO,
        "revision": R0_ADAPTER_REVISION,
        "subfolder": R0_ADAPTER_SUBFOLDER,
        "base_revision": R0_GENERATOR_BASE_REVISION,
        "file_digests": {},
    }


def r0_adapter_spec() -> Dict[str, Any]:
    """Return the full R0 adapter spec with MEASURED file digests.

    Reference for building explicit ``--adapter-spec`` launch files and
    for tests; the measured digests above were observed at the pinned
    revision on 2026-09-23. Launch code still requires the spec to be
    passed explicitly (no silent default fills it in).
    """
    return {
        "repo": R0_ADAPTER_REPO,
        "revision": R0_ADAPTER_REVISION,
        "subfolder": R0_ADAPTER_SUBFOLDER,
        "base_revision": R0_GENERATOR_BASE_REVISION,
        "training_scope": R0_ADAPTER_SCOPE,
        "file_digests": dict(R0_ADAPTER_FILE_DIGESTS),
    }


def validate_adapter_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate an explicit reuse adapter spec (fail closed, no defaults).

    Requires repo + immutable revision + subfolder + measured 64-hex file
    digests (at least ``adapter_model.safetensors``) + base revision.
    Returns a normalized copy. Raises ValueError on anything missing,
    floating, or malformed. Never fills in an unobserved digest.
    """
    if not isinstance(spec, dict):
        raise ValueError("adapter spec must be a dict with repo/revision/subfolder/file_digests")
    repo = str(spec.get("repo") or "").strip()
    if not repo or "/" not in repo:
        raise ValueError(f"adapter spec requires a repo id like 'org/name', got {spec.get('repo')!r}")
    revision = _norm_commit(spec.get("revision"), "revision")
    subfolder = str(spec.get("subfolder") or "").strip().strip("/")
    if not subfolder:
        raise ValueError("adapter spec requires a non-empty subfolder path")
    base_revision = _norm_commit(spec.get("base_revision"), "base_revision")
    digests = spec.get("file_digests")
    if not isinstance(digests, dict) or not digests:
        raise ValueError("adapter spec requires measured file_digests (no unobserved digest allowed)")
    normalized_digests = {str(k): _norm_sha256(v, str(k)) for k, v in digests.items()}
    if "adapter_model.safetensors" not in normalized_digests:
        raise ValueError("adapter spec file_digests must include adapter_model.safetensors sha256")
    scope = str(spec.get("training_scope") or R0_ADAPTER_SCOPE)
    return {
        "repo": repo,
        "revision": revision,
        "subfolder": subfolder,
        "base_revision": base_revision,
        "file_digests": normalized_digests,
        "training_scope": scope,
    }


def _load_adapter_manifest(adapter_dir: Path) -> Dict[str, Any]:
    manifest_path = adapter_dir / "generator_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"adapter generator_manifest.json missing in {adapter_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def verify_adapter_dir(
    adapter_dir: str | Path,
    spec: Dict[str, Any],
    expected_base_model: str,
    expected_scope: str = R0_ADAPTER_SCOPE,
) -> Dict[str, Any]:
    """Verify a staged adapter dir against the pinned spec + checkpoint contract.

    Checks, in order: directory exists, every ``file_digests`` entry exists
    with matching SHA-256 (1-byte tamper fails), manifest
    ``is_final_checkpoint=true``, ``smoke_only=false``,
    ``training_scope`` match, ``val_fold``/``val_fold_excluded`` null, and
    base-model match. Returns ``{verified, digests, source_metadata}``.
    Raises (no fallback to another source) on any mismatch.
    """
    normalized = validate_adapter_spec(spec)
    target = Path(adapter_dir)
    if not target.is_dir():
        raise FileNotFoundError(f"adapter directory missing: {target}")
    observed: Dict[str, str] = {}
    for rel, expected in normalized["file_digests"].items():
        candidate = target / rel
        if not candidate.is_file():
            raise FileNotFoundError(f"adapter file missing: {rel} in {target}")
        actual = compute_file_sha256(candidate)
        if actual.lower() != expected.lower():
            raise ValueError(f"adapter file digest mismatch for {rel}: expected {expected}, got {actual}")
        observed[rel] = actual.lower()
    manifest = _load_adapter_manifest(target)
    if manifest.get("is_final_checkpoint") is not True:
        raise ValueError(f"adapter {target} is NOT marked is_final_checkpoint=true: {manifest.get('is_final_checkpoint')!r}")
    if manifest.get("smoke_only") is True:
        raise ValueError(f"adapter {target} is a smoke checkpoint; refusing reuse in final path")
    scope = manifest.get("training_scope")
    if scope != expected_scope:
        raise ValueError(f"adapter {target} training_scope mismatch: expected {expected_scope!r}, got {scope!r}")
    excluded = manifest.get("val_fold_excluded", manifest.get("val_fold"))
    if excluded is not None:
        raise ValueError(f"adapter {target} trained with held-out val_fold={excluded!r}; final reuse needs null")
    base_m = manifest.get("base_model_id") or manifest.get("base_model") or manifest.get("base_model_name_or_path")
    if base_m and expected_base_model and str(base_m) != str(expected_base_model):
        raise ValueError(f"adapter {target} base model mismatch: expected {expected_base_model!r}, got {base_m!r}")
    source_metadata = {
        "repo": normalized["repo"],
        "revision": normalized["revision"],
        "subfolder": normalized["subfolder"],
        "base_revision": normalized["base_revision"],
        "training_scope": scope,
        "val_fold": None,
        "is_final_checkpoint": True,
        "optimizer_steps": manifest.get("optimizer_steps", manifest.get("global_step")),
        "dataset_size": manifest.get("dataset_size", manifest.get("training_examples", manifest.get("num_examples"))),
        "num_train_epochs": manifest.get("num_train_epochs"),
    }
    return {"verified": True, "digests": observed, "source_metadata": source_metadata}


def get_executed_git_identity(repo_root: str | Path | None = None) -> Dict[str, Any]:
    """Record the ACTUAL running code revision (never the candidate SHA).

    Returns ``{executed_git_sha, dirty}``; ``unknown`` when git is
    unavailable (honest, not a substitute for the candidate pin).
    """
    cwd = str(repo_root) if repo_root else None
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=cwd).strip().lower()
        if not _COMMIT_RE.match(sha):
            sha = "unknown"
    except Exception:
        return {"executed_git_sha": "unknown", "dirty": "unknown"}
    try:
        porcelain = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=cwd)
        dirty = bool(porcelain.strip())
    except Exception:
        dirty = "unknown"
    return {"executed_git_sha": sha, "dirty": dirty}


def build_source_identity(
    candidate_manifest: Dict[str, Any],
    executed_git_sha: str,
    dirty: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a source record keeping candidate vs executed SHAs separate."""
    candidate_sha = str((candidate_manifest or {}).get("git_commit_sha") or "")
    record: Dict[str, Any] = {
        "candidate_git_sha": candidate_sha,
        "executed_git_sha": str(executed_git_sha or ""),
        "dirty": dirty,
        "git_match": bool(candidate_sha and executed_git_sha and candidate_sha.lower() == str(executed_git_sha).lower()),
        "candidate_id": str((candidate_manifest or {}).get("candidate_id") or ""),
        "algorithm_sha256": str((candidate_manifest or {}).get("algorithm_sha256") or ""),
    }
    if extra:
        record.update(extra)
    return record


def decide_launch_selection(
    *,
    stage: str,
    generator_mode: str,
    candidate_arg: str,
    parent_arg: str,
    skip_parent_check: bool,
    available_candidates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Decide launch file selection without silent auto-pick on the reuse path.

    Reuse ``full`` runs must name an explicit ``--candidate`` manifest and
    either an explicit ``--parent-report`` or an explicit
    ``--skip-parent-check`` bypass (recorded as ``bypass_explicit``, never a
    synthetic PASS). Non-reuse stages keep the legacy newest-mtime auto-pick.
    Raises ValueError when the reuse path lacks an explicit selection.
    """
    mode = str(generator_mode or "").strip()
    if stage == "full" and mode == "reuse":
        if not candidate_arg:
            raise ValueError("reuse full requires an explicit --candidate manifest path (no mtime auto-pick)")
        if not parent_arg and not skip_parent_check:
            raise ValueError("reuse full requires an explicit --parent-report or explicit --skip-parent-check")
        parent_policy = "bypass_explicit" if (skip_parent_check and not parent_arg) else "verify_parent_report"
        return {
            "candidate_path": str(candidate_arg),
            "parent_path": str(parent_arg) if parent_arg else None,
            "parent_policy": parent_policy,
        }
    picks = list(available_candidates or [])
    return {
        "candidate_path": str(candidate_arg) if candidate_arg else (picks[-1] if picks else ""),
        "parent_path": str(parent_arg) if parent_arg else None,
        "parent_policy": "bypass_explicit" if skip_parent_check else "verify_parent_report",
    }


def explicit_bypass_parent_report(candidate_manifest: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """Record an EXPLICIT parent-check bypass (never a synthetic PASS)."""
    cid = str((candidate_manifest or {}).get("candidate_id") or "")
    return {
        "schema_version": 1,
        "stage": stage,
        "status": "BYPASSED_EXPLICIT",
        "candidate_id": cid,
        "candidate_sha": cid,
        "parent_policy": "bypass_explicit",
        "report_sha256": "bypass_explicit",
    }


def write_execution_record(path: str | Path, record: Dict[str, Any]) -> str:
    """Write a reuse execution record (fixture-friendly, no remote needed)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return digest
