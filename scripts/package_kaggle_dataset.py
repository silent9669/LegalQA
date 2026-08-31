"""Package clean, verified LegalQA training dataset artifacts for Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.security import assert_no_secrets_in_workspace

# Whitelisted files to include in clean Kaggle dataset
REQUIRED_FILES = [
    "data/legal_chunks.parquet",
    "data/qa_unique.parquet",
    "data/known_qa.json",
    "data/qa_citations.parquet",
    "data/retrieval_labels.parquet",
    "data/fold_assignments.parquet",
]

OPTIONAL_DIRS = [
    "indexes/bm25",
    "indexes/dek21",
]


def compute_file_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def package_kaggle_dataset(
    source_dir: str = "artifacts/task2",
    staging_dir: str = "kaggle_dataset/staged",
    dataset_title: str = "LegalQA",
    dataset_slug: str = "legalqa-task2-clean-data",
    user_handle: str = "phucdangg",
    dry_run: bool = False,
) -> None:
    print(f"=== Packaging Kaggle Dataset '{dataset_title}' ===")
    src = Path(source_dir)
    stage = Path(staging_dir)

    # Security check
    assert_no_secrets_in_workspace(Path.cwd())

    # Verify required source files exist
    missing = []
    for rel_path in REQUIRED_FILES:
        if not (src / rel_path).exists():
            missing.append(rel_path)

    if missing:
        raise FileNotFoundError(f"Missing required artifact(s) in {source_dir}: {missing}. Run scripts/prepare_data.py first.")

    if not dry_run:
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "title": dataset_title,
        "slug": dataset_slug,
        "owner": user_handle,
        "files": {},
    }

    print("Staging clean canonical artifacts:")
    for rel_path in REQUIRED_FILES:
        src_file = src / rel_path
        dest_file = stage / Path(rel_path).name
        file_sha = compute_file_sha256(src_file)
        file_size_mb = src_file.stat().st_size / (1024 * 1024)

        manifest["files"][Path(rel_path).name] = {
            "source": str(rel_path),
            "sha256": file_sha,
            "size_mb": round(file_size_mb, 2),
        }
        print(f"  + {Path(rel_path).name} ({file_size_mb:.1f} MB) -> sha256: {file_sha[:12]}...")

        if not dry_run:
            shutil.copy2(src_file, dest_file)

    # Also stage public-official.json if present
    public_official = Path("artifacts/raw/public-official.json")
    if public_official.exists():
        file_sha = compute_file_sha256(public_official)
        manifest["files"]["public-official.json"] = {
            "source": "artifacts/raw/public-official.json",
            "sha256": file_sha,
            "size_mb": round(public_official.stat().st_size / (1024 * 1024), 2),
        }
        print(f"  + public-official.json ({public_official.stat().st_size/1024:.1f} KB)")
        if not dry_run:
            shutil.copy2(public_official, stage / "public-official.json")

    # Staging metadata.json for Kaggle CLI
    kaggle_meta = {
        "title": dataset_title,
        "id": f"{user_handle}/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }

    if not dry_run:
        with open(stage / "dataset-metadata.json", "w", encoding="utf-8") as f:
            json.dump(kaggle_meta, f, indent=2)
        with open(stage / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Also write root kaggle_dataset/dataset-metadata.json
        with open(Path("kaggle_dataset/dataset-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(kaggle_meta, f, indent=2)

    print(f"\nSuccessfully staged {len(manifest['files'])} clean files to {stage}.")
    print(f"Kaggle Dataset Title: '{dataset_title}' | ID: '{user_handle}/{dataset_slug}'")


def main():
    parser = argparse.ArgumentParser(description="Package clean LegalQA dataset for Kaggle.")
    parser.add_argument("--source", default="artifacts/task2", help="Source artifact directory")
    parser.add_argument("--staging", default="kaggle_dataset/staged", help="Staging output directory")
    parser.add_argument("--title", default="LegalQA", help="Kaggle dataset display title")
    parser.add_argument("--dry_run", action="store_true", help="Simulate staging without copying files")
    args = parser.parse_args()

    package_kaggle_dataset(
        source_dir=args.source,
        staging_dir=args.staging,
        dataset_title=args.title,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
