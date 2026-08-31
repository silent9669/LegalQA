"""Package clean, self-contained LegalQA dataset and code runtime artifacts for Kaggle (V7)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.hashing import sha256_file
from src.common.security import assert_no_secrets_in_workspace
from src.task2.production_config import load_production_selection

RUNTIME_API_VERSION = 7

REQUIRED_FILES = [
    "data/legal_chunks.parquet",
    "data/qa_unique.parquet",
    "data/known_qa.json",
    "data/qa_citations.parquet",
    "data/retrieval_labels.parquet",
    "data/fold_assignments.parquet",
]

OPTIONAL_DATA_FILES = [
    "data/reranker_training_pairs.parquet",
]

OPTIONAL_DIRS = [
    "indexes/bm25",
    "indexes/dek21",
]


def get_git_sha() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def package_kaggle_dataset(
    source_dir: str = "artifacts/task2",
    staging_dir: str = "kaggle_dataset/staged",
    dataset_title: str = "LegalQA",
    dataset_slug: str = "legalqa-task2-clean-data",
    user_handle: str = "phucdangg",
    profile: str = "default",  # "default" or "final_training"
    production_config_path: str = "configs/production_selection.yaml",
    include_code: bool = True,
    dry_run: bool = False,
) -> None:
    print(f"=== Packaging Self-Contained Kaggle Dataset '{dataset_title}' (Profile: {profile.upper()} | API: v{RUNTIME_API_VERSION}) ===")
    src = Path(source_dir)
    stage = Path(staging_dir)

    # Security check
    assert_no_secrets_in_workspace(Path.cwd())

    # Verify required source files exist
    missing = []
    for rel_path in REQUIRED_FILES:
        if not (src / rel_path).exists():
            missing.append(rel_path)

    if profile == "final_training":
        # Strict checks for final_training profile
        public_official = Path("artifacts/raw/public-official.json")
        if not public_official.exists():
            missing.append("artifacts/raw/public-official.json")

        bm25_idx = src / "indexes" / "bm25"
        if not bm25_idx.exists() or not any(bm25_idx.iterdir()):
            missing.append("indexes/bm25")

        dek21_idx = src / "indexes" / "dek21"
        if not dek21_idx.exists() or not any(dek21_idx.iterdir()):
            missing.append("indexes/dek21")

        if not os.path.exists(production_config_path):
            missing.append(production_config_path)
        else:
            try:
                prod_cfg = load_production_selection(production_config_path)
                if prod_cfg.use_task_tuned_reranker:
                    rerank_pairs = src / "data" / "reranker_training_pairs.parquet"
                    if not rerank_pairs.exists():
                        missing.append("data/reranker_training_pairs.parquet (required by production_selection tuned reranker)")
            except Exception as e:
                missing.append(f"Invalid {production_config_path}: {e}")

    if missing:
        raise FileNotFoundError(
            f"Missing required artifact(s) for profile '{profile}' in {source_dir}: {missing}."
        )

    if not dry_run:
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "title": dataset_title,
        "slug": dataset_slug,
        "owner": user_handle,
        "profile": profile,
        "runtime_api_version": RUNTIME_API_VERSION,
        "git_sha": get_git_sha(),
        "files": {},
        "indexes": {},
        "code": {},
    }

    print("Staging clean canonical data artifacts:")
    for rel_path in REQUIRED_FILES:
        src_file = src / rel_path
        dest_file = stage / Path(rel_path).name
        file_sha = sha256_file(src_file)
        file_size_mb = src_file.stat().st_size / (1024 * 1024)

        manifest["files"][Path(rel_path).name] = {
            "source": str(rel_path),
            "sha256": file_sha,
            "size_mb": round(file_size_mb, 2),
        }
        print(f"  + {Path(rel_path).name} ({file_size_mb:.1f} MB) -> sha256: {file_sha[:12]}...")

        if not dry_run:
            shutil.copy2(src_file, dest_file)

    # Stage optional training files (e.g. reranker_training_pairs.parquet)
    for opt_rel in OPTIONAL_DATA_FILES:
        src_opt = src / opt_rel
        if src_opt.exists():
            dest_file = stage / Path(opt_rel).name
            file_sha = sha256_file(src_opt)
            file_size_mb = src_opt.stat().st_size / (1024 * 1024)
            manifest["files"][Path(opt_rel).name] = {
                "source": str(opt_rel),
                "sha256": file_sha,
                "size_mb": round(file_size_mb, 2),
            }
            print(f"  + {Path(opt_rel).name} ({file_size_mb:.1f} MB) -> sha256: {file_sha[:12]}...")
            if not dry_run:
                shutil.copy2(src_opt, dest_file)
        else:
            print(f"  - {opt_rel} (not found, run scripts/mine_retrieval_negatives.py to generate)")

    # Stage public-official.json if present
    public_official = Path("artifacts/raw/public-official.json")
    if public_official.exists():
        file_sha = sha256_file(public_official)
        manifest["files"]["public-official.json"] = {
            "source": "artifacts/raw/public-official.json",
            "sha256": file_sha,
            "size_mb": round(public_official.stat().st_size / (1024 * 1024), 2),
        }
        print(f"  + public-official.json ({public_official.stat().st_size / 1024:.1f} KB)")
        if not dry_run:
            shutil.copy2(public_official, stage / "public-official.json")

    # Staging precomputed indexes
    print("Staging precomputed retrieval indexes:")
    for opt_dir in OPTIONAL_DIRS:
        src_opt = src / opt_dir
        if src_opt.exists() and any(src_opt.iterdir()):
            dest_opt = stage / opt_dir
            if not dry_run:
                dest_opt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_opt, dest_opt, dirs_exist_ok=True)

            idx_files = [str(f.relative_to(src_opt)) for f in src_opt.rglob("*") if f.is_file()]
            manifest["indexes"][opt_dir] = {
                "source": str(opt_dir),
                "files_count": len(idx_files),
                "files": idx_files[:10],
            }
            print(f"  + {opt_dir}/ ({len(idx_files)} index files staged)")
        else:
            print(f"  - {opt_dir}/ (not found or empty, skipped)")

    # Staging Code Runtime (src/, scripts/, configs/, requirements-kaggle.txt)
    if include_code:
        print("Staging code runtime into code/LegalQA/ :")
        code_root = stage / "code" / "LegalQA"
        code_manifest: Dict[str, Any] = {
            "git_sha": get_git_sha(),
            "runtime_api_version": RUNTIME_API_VERSION,
            "files": {},
        }

        ignore_patterns = shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".DS_Store",
            "artifacts",
            "kaggle_dataset",
            ".pytest_cache",
            "logs",
        )

        # 1. src/
        src_dir = Path("src")
        if src_dir.exists() and not dry_run:
            shutil.copytree(src_dir, code_root / "src", dirs_exist_ok=True, ignore=ignore_patterns)
        for py_file in src_dir.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                code_manifest["files"][str(py_file)] = sha256_file(py_file)

        # 2. scripts/
        scripts_dir = Path("scripts")
        if scripts_dir.exists() and not dry_run:
            shutil.copytree(scripts_dir, code_root / "scripts", dirs_exist_ok=True, ignore=ignore_patterns)
        for py_file in scripts_dir.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                code_manifest["files"][str(py_file)] = sha256_file(py_file)

        # 3. configs/
        cfg_dir = Path("configs")
        if cfg_dir.exists() and not dry_run:
            shutil.copytree(cfg_dir, code_root / "configs", dirs_exist_ok=True, ignore=ignore_patterns)
        for yaml_file in cfg_dir.rglob("*.yaml"):
            code_manifest["files"][str(yaml_file)] = sha256_file(yaml_file)

        # 4. requirements-kaggle.txt
        req_file = Path("requirements-kaggle.txt")
        if req_file.exists() and not dry_run:
            shutil.copy2(req_file, code_root / "requirements-kaggle.txt")
            code_manifest["files"]["requirements-kaggle.txt"] = sha256_file(req_file)

        manifest["code"] = {
            "root": "code/LegalQA",
            "files_count": len(code_manifest["files"]),
            "runtime_api_version": RUNTIME_API_VERSION,
        }

        if not dry_run:
            with open(stage / "code_manifest.json", "w", encoding="utf-8") as f:
                json.dump(code_manifest, f, indent=2)
            with open(code_root / "code_manifest.json", "w", encoding="utf-8") as f:
                json.dump(code_manifest, f, indent=2)
        print(f"  + Staged {len(code_manifest['files'])} code files (src, scripts, configs) and code_manifest.json (v{RUNTIME_API_VERSION}).")

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

        with open(Path("kaggle_dataset/dataset-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(kaggle_meta, f, indent=2)

    print(f"\nSuccessfully staged clean dataset to {stage}.")
    print(f"Kaggle Dataset Title: '{dataset_title}' | ID: '{user_handle}/{dataset_slug}' | Runtime API: v{RUNTIME_API_VERSION}")


def main():
    parser = argparse.ArgumentParser(description="Package clean LegalQA dataset and code for Kaggle.")
    parser.add_argument("--source", default="artifacts/task2", help="Source artifact directory")
    parser.add_argument("--staging", default="kaggle_dataset/staged", help="Staging output directory")
    parser.add_argument("--title", default="LegalQA", help="Kaggle dataset display title")
    parser.add_argument("--profile", default="default", choices=["default", "final_training"])
    parser.add_argument("--dry_run", action="store_true", help="Simulate staging without copying files")
    parser.add_argument("--no_code", action="store_true", help="Omit code runtime from dataset")
    args = parser.parse_args()

    package_kaggle_dataset(
        source_dir=args.source,
        staging_dir=args.staging,
        dataset_title=args.title,
        profile=args.profile,
        include_code=not args.no_code,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
