#!/usr/bin/env python3
"""Migration consumer audit: inventory, gate-report comparison, rollback bundle.

Usage:
  python scripts/audit_migration.py --consumers   # print consumer inventory
  python scripts/audit_migration.py --rollback <destination>  # build rollback bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

# Patterns whose consumers must be enumerated before any removal/archive.
TRACKED_PATTERNS = {
    "legacy_flat_config": r"colab_train_a100\.yaml",
    "legacy_smoke_config": r"kaggle_smoke_t4\.yaml",
    "gate_runner": r"run_gpu_gate",
    "modal_t4_gate": r"run_modal_t4_remote",
    "pipeline_runner": r"run_pipeline",
    "modal_adapter": r"modal_app",
    "release_verified": r"release_verified",
    "candidate_freeze": r"freeze_candidate|CandidateManifest",
}

SKIP_DIRS = {".git", ".venv", ".venv311", ".venv-ml", "__pycache__", ".pytest_cache",
             ".playwright-mcp", "node_modules", "dsc2026", ".remember", ".superpowers", ".claude"}
SKIP_SUFFIXES = (".pyc", ".pyo")


def inventory_consumers(paths: List[str]) -> Dict[str, List[str]]:
    """Search repository paths for consumers of migration-sensitive modules.

    Returns {consumer_file: [matched_pattern_names]} with repo-relative paths
    sorted. External symlink targets are never followed.
    """
    consumers: Dict[str, List[str]] = {}
    compiled = {name: re.compile(pattern) for name, pattern in TRACKED_PATTERNS.items()}
    for base in paths:
        root = Path(base)
        if not root.exists():
            continue
        if root.is_symlink():
            continue
        files = [root] if root.is_file() else list(root.rglob("*"))
        for path in files:
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix in SKIP_SUFFIXES:
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            hits = sorted(name for name, rx in compiled.items() if rx.search(text))
            if hits:
                try:
                    rel = str(path.relative_to(REPO_ROOT))
                except ValueError:
                    rel = str(path)
                consumers[rel] = hits
    return dict(sorted(consumers.items()))


def compare_gate_reports(old: Dict, new: Dict) -> Dict:
    """Compare old vs replacement gate request/report identity and stage sets.

    Missing required stages block removal (FAIL); identity mismatches across
    candidate/algorithm/dataset/scorer fail closed; otherwise PASS.
    """
    old_stages = list(old.get("stages", []))
    new_stages = list(new.get("stages", []))
    missing = [s for s in old_stages if s not in new_stages]
    if missing:
        return {"status": "FAIL", "missing_stages": missing, "old_stages": old_stages, "new_stages": new_stages}
    mismatches = []
    for key in ("candidate_sha", "candidate_id", "algorithm_sha256", "dataset_sha256",
                "dataset_manifest_sha256", "scorer_sha256"):
        if key in old and key in new and old[key] != new[key]:
            mismatches.append(key)
    if mismatches:
        return {"status": "FAIL", "identity_mismatches": mismatches}
    old_status = old.get("status", "PASS")
    new_status = new.get("status", "PASS")
    if old_status != "PASS" or new_status != "PASS":
        return {"status": "FAIL", "old_status": old_status, "new_status": new_status}
    return {"status": "PASS", "stages": new_stages or old_stages}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_rollback_bundle(files: List[str], destination: str) -> Dict:
    """Copy files into a rollback bundle with checksums + restore command.

    Symlinks are preserved as links (targets never followed). Returns the
    bundle record including a restore command.
    """
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in files:
        src = Path(name)
        if not src.exists() and not src.is_symlink():
            raise FileNotFoundError(f"rollback source not found: {name}")
        target = dest / Path(name).name
        if src.is_symlink():
            link_target = os.readlink(src)
            if target.is_symlink() or target.exists():
                target.unlink() if target.is_symlink() or target.is_file() else shutil.rmtree(target)
            os.symlink(link_target, target)
            entries.append({"file": name, "type": "symlink", "link_target": link_target})
        elif src.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target, symlinks=True)
            entries.append({"file": name, "type": "dir"})
        else:
            shutil.copy2(src, target)
            entries.append({"file": name, "type": "file", "sha256": _sha256_file(target)})
    record = {
        "files": entries,
        "destination": str(dest),
        "restore_command": f"cp -r {dest}/* <repo-root>/  # then verify checksums",
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    record["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    (dest / "rollback_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Migration consumer audit + rollback bundle.")
    parser.add_argument("--consumers", action="store_true", help="Print consumer inventory")
    parser.add_argument("--rollback", default="", help="Build rollback bundle at destination for tracked legacy files")
    args = parser.parse_args()
    if args.consumers:
        inventory = inventory_consumers(["scripts", "tests", "notebooks", "configs", "src", "docs"])
        print(json.dumps(inventory, indent=2))
    if args.rollback:
        record = build_rollback_bundle(["configs/colab_train_a100.yaml", "configs/kaggle_smoke_t4.yaml"], args.rollback)
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
