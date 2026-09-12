#!/usr/bin/env python3
"""
Packages canonical dataset artifacts for direct upload to Kaggle.
Strictly ensures ZERO code files or code directories are included in the package.
Generates a cryptographic dataset_manifest.json containing exact SHA-256 hashes.
"""

import os
import sys
import glob
import json
import argparse
import hashlib
from typing import Dict, Any

def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def package_dataset(source_dir: str, title: str = "LegalQA", slug: str = "legalqa-task2-clean-data", owner: str = "phucdangg") -> Dict[str, Any]:
    source_dir = os.path.abspath(source_dir)
    if not os.path.isdir(source_dir):
        raise ValueError(f"Source directory {source_dir} not found.")

    # 1. Clean out any code artifacts if accidentally present
    for bad_name in ["code", "src", "__pycache__", ".git"]:
        bad_path = os.path.join(source_dir, bad_name)
        if os.path.exists(bad_path):
            print(f"Removing invalid artifact from dataset: {bad_path}")
            if os.path.isdir(bad_path):
                import shutil
                shutil.rmtree(bad_path)
            else:
                os.remove(bad_path)

    for py_file in glob.glob(os.path.join(source_dir, "**/*.py"), recursive=True):
        print(f"Removing code file from dataset package: {py_file}")
        os.remove(py_file)

    # 2. Audit and hash data files
    manifest: Dict[str, Any] = {
        "title": title,
        "slug": slug,
        "owner": owner,
        "runtime_api_version": 16,
        "files": {},
        "indexes": {},
    }

    tracked_files = [
        "legal_chunks.parquet",
        "qa_unique.parquet",
        "known_qa.json",
        "qa_citations.parquet",
        "retrieval_labels.parquet",
        "fold_assignments.parquet",
        "reranker_training_pairs.parquet",
        "public-official.json",
    ]

    for fname in tracked_files:
        fpath = os.path.join(source_dir, fname)
        if os.path.exists(fpath):
            size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
            sha = compute_sha256(fpath)
            manifest["files"][fname] = {
                "source": fname,
                "sha256": sha,
                "size_mb": size_mb,
            }
            print(f"  Tracked file: {fname:<32} {size_mb:>8.2f} MB  SHA: {sha[:12]}...")

    # Indexes
    bm25_dir = os.path.join(source_dir, "indexes", "bm25")
    if os.path.isdir(bm25_dir):
        bm25_files = sorted([os.path.relpath(p, bm25_dir) for p in glob.glob(os.path.join(bm25_dir, "**"), recursive=True) if os.path.isfile(p)])
        manifest["indexes"]["indexes/bm25"] = {
            "source": "indexes/bm25",
            "files_count": len(bm25_files),
            "files": bm25_files,
        }

    dek21_dir = os.path.join(source_dir, "indexes", "dek21")
    if os.path.isdir(dek21_dir):
        dek21_files = sorted([os.path.relpath(p, dek21_dir) for p in glob.glob(os.path.join(dek21_dir, "**"), recursive=True) if os.path.isfile(p)])
        manifest["indexes"]["indexes/dek21"] = {
            "source": "indexes/dek21",
            "files_count": len(dek21_files),
            "files": dek21_files,
        }

    # Save clean dataset_manifest.json
    manifest_out = os.path.join(source_dir, "dataset_manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved canonical dataset manifest to {manifest_out}")

    # Ensure dataset-metadata.json exists
    metadata_path = os.path.join(source_dir, "dataset-metadata.json")
    meta = {
        "title": title,
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved dataset metadata to {metadata_path}")

    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package clean dataset for Kaggle")
    parser.add_argument("--source-dir", default="kaggle_dataset", help="Source dataset directory")
    args = parser.parse_args()

    print(f"=== Packaging Kaggle Dataset: {args.source_dir} ===")
    package_dataset(args.source_dir)
    print("=== Packaging Complete ===")
