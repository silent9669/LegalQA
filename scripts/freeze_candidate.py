#!/usr/bin/env python3
"""Freeze an immutable CandidateManifest for LegalQA Task 2.

Usage:
  python scripts/freeze_candidate.py [--allow-dirty] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.task2.config.loader import load_resolved_config, canonical_sha256
from src.task2.provenance.candidate import (
    CandidateManifest,
    DatasetRef,
    ModelRevisionRef,
    ModelsRef,
    RuntimeProfilesRef,
)
from src.task2.provenance.checksums import compute_file_sha256

KNOWN_MODEL_REVISIONS = {
    "Qwen/Qwen2.5-3B-Instruct": "d8a1c8901eb4284d720235adcf8849767f40d7e4",
    "BAAI/bge-reranker-v2-m3": "278e2e28328135817d69932cb4ad12d7c58e5d32",
    "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2": "c356b6aa96e00cb1b6a12b48a1c0d4530058b884",
}


def get_git_commit_sha() -> str:
    """Get exact HEAD commit SHA."""
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def check_git_cleanliness(allow_dirty: bool = False) -> None:
    """Ensure working tree is clean for production candidate freeze."""
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    output = res.stdout.strip()
    if output and not allow_dirty:
        raise RuntimeError(
            "Working tree is dirty! A clean Git working tree is required to freeze a production candidate.\n"
            f"Changed files:\n{output}\n"
            "Commit or stash changes first, or pass --allow-dirty for local debugging."
        )


def resolve_hf_model_revision(model_id: str, override_rev: Optional[str] = None) -> str:
    """Resolve exact commit SHA for a Hugging Face model repository."""
    if override_rev:
        return override_rev

    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        if hasattr(info, "sha") and info.sha:
            return str(info.sha)
    except Exception as e:
        print(f"Notice: huggingface_hub.model_info failed for {model_id} ({e}); using verified baseline pin.")

    if model_id in KNOWN_MODEL_REVISIONS:
        return KNOWN_MODEL_REVISIONS[model_id]

    raise RuntimeError(f"Unable to resolve exact commit SHA for model {model_id}.")


def freeze_candidate(
    output_dir: Path = REPO_ROOT / "artifacts" / "candidates",
    allow_dirty: bool = False,
    dataset_slug: str = "phucdangg/legalqa-task2-clean-data",
    dataset_version: int = 1,
    dataset_manifest_path: Optional[Path] = None,
    generator_revision: Optional[str] = None,
    reranker_revision: Optional[str] = None,
    dense_revision: Optional[str] = None,
) -> CandidateManifest:
    """Construct, hash, and persist an immutable CandidateManifest."""
    check_git_cleanliness(allow_dirty=allow_dirty)
    git_sha = get_git_commit_sha()

    algo_path = REPO_ROOT / "configs" / "task2" / "algorithm.yaml"
    rt_kaggle = REPO_ROOT / "configs" / "task2" / "runtime" / "kaggle_t4x2.yaml"
    rt_colab_t4 = REPO_ROOT / "configs" / "task2" / "runtime" / "colab_t4.yaml"
    rt_colab_a100 = REPO_ROOT / "configs" / "task2" / "runtime" / "colab_a100.yaml"

    cfg_kaggle = load_resolved_config(algo_path, rt_kaggle)
    cfg_colab_t4 = load_resolved_config(algo_path, rt_colab_t4)
    cfg_colab_a100 = load_resolved_config(algo_path, rt_colab_a100)

    # Dataset manifest hash
    manifest_p = dataset_manifest_path or (REPO_ROOT / "kaggle_dataset" / "dataset_manifest.json")
    if manifest_p.exists():
        manifest_sha = compute_file_sha256(manifest_p)
    else:
        # Fallback to schema if local data not staged
        manifest_sha = compute_file_sha256(REPO_ROOT / "configs" / "dataset_schema.yaml")

    # Dependency lock hash
    constraints_p = REPO_ROOT / "constraints-gpu.txt"
    if not constraints_p.exists():
        raise FileNotFoundError(f"Missing constraints-gpu.txt at {constraints_p}")
    dep_lock_sha = compute_file_sha256(constraints_p)

    # Model revisions
    gen_id = cfg_kaggle.algorithm.models.generator.id
    rerank_id = cfg_kaggle.algorithm.models.reranker.id
    dense_id = cfg_kaggle.algorithm.models.dense.id

    gen_rev = resolve_hf_model_revision(gen_id, generator_revision)
    rerank_rev = resolve_hf_model_revision(rerank_id, reranker_revision)
    dense_rev = resolve_hf_model_revision(dense_id, dense_revision)

    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    manifest = CandidateManifest(
        schema_version=1,
        candidate_id="",
        task="task2",
        git_repository="https://github.com/silent9669/LegalQA.git",
        git_commit_sha=git_sha,
        dataset=DatasetRef(
            slug=dataset_slug,
            version=dataset_version,
            manifest_sha256=manifest_sha,
        ),
        algorithm_sha256=cfg_kaggle.algorithm_sha256,
        runtime_profile_sha256=RuntimeProfilesRef(
            kaggle_t4x2=cfg_kaggle.runtime_sha256,
            colab_t4=cfg_colab_t4.runtime_sha256,
            colab_a100=cfg_colab_a100.runtime_sha256,
        ),
        config_bundle_sha256=cfg_kaggle.bundle_sha256,
        models=ModelsRef(
            generator=ModelRevisionRef(id=gen_id, revision=gen_rev),
            reranker=ModelRevisionRef(id=rerank_id, revision=rerank_rev),
            dense=ModelRevisionRef(id=dense_id, revision=dense_rev),
        ),
        dependency_lock_sha256=dep_lock_sha,
        seed=cfg_kaggle.algorithm.seed,
        created_at_utc=now_utc,
    )

    manifest = manifest.with_computed_id()

    target_path = output_dir / manifest.candidate_id / "candidate_manifest.json"
    manifest.save_json(target_path)
    print(f"\n[+] Successfully froze candidate: {manifest.candidate_id}")
    print(f"    Git Commit SHA:  {manifest.git_commit_sha}")
    print(f"    Algorithm SHA:   {manifest.algorithm_sha256[:16]}...")
    print(f"    Dataset Version: {manifest.dataset.slug} v{manifest.dataset.version}")
    print(f"    Manifest Saved:  {target_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Freeze immutable CandidateManifest.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty working tree for debugging")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "candidates"), help="Output directory")
    parser.add_argument("--dataset-version", type=int, default=1, help="Numeric Kaggle dataset version")
    args = parser.parse_args()

    freeze_candidate(
        output_dir=Path(args.output_dir),
        allow_dirty=args.allow_dirty,
        dataset_version=args.dataset_version,
    )


if __name__ == "__main__":
    main()
