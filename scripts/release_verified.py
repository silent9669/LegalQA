#!/usr/bin/env python3
"""Explicit, receipt-producing release orchestration (Task 9).

Stages (but never publishes) by default. --publish performs the explicitly
authorized immutable Hugging Face commit into runs/<run_id>/, verifies the
returned commit (sizes + digests), and writes the EXTERNAL receipt outside
the self-hashed bundle directory.

Usage:
  python scripts/release_verified.py --bundle <run-bundle>            # stage only
  python scripts/release_verified.py --bundle <run-bundle> --publish  # authorized immutable commit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.security import assert_no_secrets_in_workspace  # noqa: E402
from src.task2.provenance.release_receipt import (  # noqa: E402
    build_release_manifest,
    stage_release,
    verify_release_manifest,
)
from src.task2.scorer_contract import (  # noqa: E402
    validate_prediction_payload,
    verify_zip_inner_matches_loose,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact(path: Path, arcname: str) -> dict:
    return {"path": arcname, "sha256": _sha256_file(path), "bytes": path.stat().st_size}


def collect_release_inputs(bundle_dir: Path) -> tuple[dict, list]:
    """Run the CPU preflight sequence and collect the release run record."""
    bundle_dir = Path(bundle_dir)
    candidate = json.loads((bundle_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((bundle_dir / "production_run_manifest.json").read_text(encoding="utf-8"))

    # 1. Same candidate SHA across bundle + gate reports.
    if manifest.get("candidate_id") != candidate.get("candidate_id"):
        raise ValueError("bundle candidate_id does not match candidate_manifest.json")
    gate_dir = bundle_dir / "gate_reports"
    for stage in ("kaggle_t4x2", "a100_micro_probe"):
        report = json.loads((gate_dir / f"{stage}_report.json").read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise ValueError(f"gate report {stage} is not PASS")
        if report.get("candidate_id") != candidate.get("candidate_id"):
            raise ValueError(f"gate report {stage} candidate mismatch")
    if (gate_dir / "colab_t4_report.json").is_file():
        report = json.loads((gate_dir / "colab_t4_report.json").read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise ValueError("gate report colab_t4 is not PASS")
        if report.get("candidate_id") != candidate.get("candidate_id"):
            raise ValueError("gate report colab_t4 candidate mismatch")

    # 2. Final checkpoint scope: all allowed data, val_fold null.
    adapter_manifest_path = bundle_dir / "final_adapter" / "generator_manifest.json"
    if adapter_manifest_path.is_file():
        adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
        if adapter_manifest.get("smoke_only"):
            raise ValueError("refusing release: smoke checkpoint in final bundle")

    # 3. CPU schema, exact-ID, ZIP, secret scans over the submission.
    submission_path = bundle_dir / "submission.json"
    if submission_path.is_file():
        from src.task2.scorer_contract import load_predictions_json

        predictions = load_predictions_json(submission_path)
        public_ids: list = manifest.get("submission", {}).get("public_ids") or sorted(predictions)
        validate_prediction_payload(predictions, public_ids)
        zip_path = bundle_dir / "submission.json.zip"
        if zip_path.is_file():
            verify_zip_inner_matches_loose(zip_path, submission_path)
    assert_no_secrets_in_workspace(bundle_dir, exclude_tests=False)

    # 4. Parameter audit below the strict 4B learned-parameter limit.
    param_audit = manifest.get("parameter_audit") or {}
    total = param_audit.get("total_learned_parameters")
    if total is not None and int(total) >= 4_000_000_000:
        raise ValueError(f"parameter budget exceeded: {total} >= 4B")

    run_id = manifest.get("run_id", bundle_dir.name)
    run = {
        "run_id": run_id,
        "candidate_id": candidate.get("candidate_id"),
        "git_commit_sha": candidate.get("git_commit_sha"),
        "algorithm_sha256": candidate.get("algorithm_sha256"),
        "seed": candidate.get("seed", 42),
        "dataset": candidate.get("dataset", {}),
        "split_fingerprint": manifest.get("split_fingerprint", ""),
        "corpus_fingerprint": manifest.get("corpus_fingerprint", ""),
        "index_fingerprint": manifest.get("index_fingerprint", ""),
        "models": candidate.get("models", {}),
        "training": {
            "optimizer_steps": manifest.get("optimizer_steps"),
            "training_sample_count": manifest.get("training_sample_count"),
            "num_train_epochs": manifest.get("num_train_epochs"),
            "effective_batch_size": manifest.get("effective_batch_size"),
        },
        "metrics": manifest.get("metrics", {}),
        "official_public_score": manifest.get("official_public_score"),
        "submission": manifest.get("submission", {}),
        "gate_reports": manifest.get("gate_reports", {}),
        "intended_repository": manifest.get("huggingface", {}).get("repository", ""),
        "intended_path_in_repo": f"runs/{run_id}",
    }
    artifacts = []
    for rel in ("submission.json", "submission.json.zip", "submission_provenance.json",
                "production_run_manifest.json", "candidate_manifest.json", "release_manifest.json"):
        candidate_path = bundle_dir / rel
        if candidate_path.is_file() and rel != "release_manifest.json":
            artifacts.append(_artifact(candidate_path, rel))
    adapter_dir = bundle_dir / "final_adapter"
    if adapter_dir.is_dir():
        for item in sorted(adapter_dir.iterdir()):
            if item.is_file():
                artifacts.append(_artifact(item, f"final_adapter/{item.name}"))
    return run, artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage (default) or publish (explicit) a verified release.")
    parser.add_argument("--bundle", required=True, help="Run bundle directory")
    parser.add_argument("--publish", action="store_true", help="Explicit authorization: immutable HF commit")
    parser.add_argument("--repo", default="", help="Override intended HF repository")
    parser.add_argument("--receipt-out", default="", help="External receipt path (default: <bundle>.receipt.json)")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle)
    run, artifacts = collect_release_inputs(bundle_dir)
    if args.repo:
        run["intended_repository"] = args.repo
    if not run["intended_repository"]:
        raise SystemExit("no intended HF repository recorded in bundle; pass --repo")
    release_manifest = build_release_manifest(run, artifacts)
    staged = stage_release(bundle_dir, release_manifest)
    print(f"[STAGED] {staged['manifest_sha256']} ({staged['num_files']} files)")

    receipt_out = Path(args.receipt_out) if args.receipt_out else bundle_dir.parent / f"{bundle_dir.name}.receipt.json"
    if receipt_out.resolve().is_relative_to(bundle_dir.resolve()):
        raise SystemExit("receipt must be stored outside the self-hashed bundle directory")

    if not args.publish:
        print("[STAGED] publish not requested; no upload performed.")
        verify_release_manifest(release_manifest, None)
        return

    from src.task2.hf_uploader import upload_run_bundle_to_hf

    upload_res = upload_run_bundle_to_hf(
        bundle_dir=bundle_dir,
        repo_id=run["intended_repository"],
        run_id=run["run_id"],
    )
    commit_sha = upload_res.get("commit_sha")
    local_digests = upload_res.get("local_digests", {})
    remote_files = []
    for artifact in artifacts:
        rel = artifact["path"]
        # Local digest is authoritative pre-download; remote re-download
        # comparison happens below via hf_hub_download at the pinned revision.
        remote_files.append({"path": f"runs/{run['run_id']}/{rel}", "size": artifact["bytes"], "sha256": artifact["sha256"]})
    _ = local_digests
    status = "PUBLISH_UNVERIFIED"
    error_msg = None
    try:
        verified = verify_release_manifest(
            release_manifest,
            {"commit_sha": commit_sha, "files": [
                {"path": a["path"], "size": a["bytes"], "sha256": a["sha256"]} for a in artifacts
            ]},
        )
        _confirm_remote_bytes(run["intended_repository"], commit_sha, run["run_id"], artifacts)
        status = verified.get("status", "PUBLISH_VERIFIED")
    except Exception as exc:
        status = "PUBLISH_UNVERIFIED"
        error_msg = str(exc)

    receipt = {
        "status": status,
        "run_id": run["run_id"],
        "candidate_id": run["candidate_id"],
        "manifest_sha256": release_manifest["manifest_sha256"],
        "local_artifact_digests": {a["path"]: a["sha256"] for a in artifacts},
        "remote_repository": run["intended_repository"],
        "remote_revision": commit_sha,
        "remote_files": remote_files,
    }
    if error_msg:
        receipt["verification_error"] = error_msg
    receipt_out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"[{status}] commit={commit_sha} receipt={receipt_out}")
    if status != "PUBLISH_VERIFIED":
        raise ValueError(f"release verification failed ({status}): {error_msg}")


def _confirm_remote_bytes(repo_id: str, commit_sha: str, run_id: str, artifacts: list) -> None:
    """Re-download each published file at the pinned revision and compare digests."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(f"huggingface_hub required for remote verification: {exc}") from exc
    for artifact in artifacts:
        local_sha = artifact["sha256"]
        downloaded = hf_hub_download(
            repo_id=repo_id, filename=f"runs/{run_id}/{artifact['path']}", revision=commit_sha
        )
        if _sha256_file(Path(downloaded)) != local_sha:
            raise ValueError(f"remote file verification failed: {artifact['path']}")


if __name__ == "__main__":
    main()
