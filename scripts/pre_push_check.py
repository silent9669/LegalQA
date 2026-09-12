#!/usr/bin/env python3
"""Canonical Pre-Push Verification Gate for LegalQA Task 2.

Supported modes:
  python scripts/pre_push_check.py --mode fast
  python scripts/pre_push_check.py --mode full (default)

Guarantees:
1. Python syntax compilation (compileall on src, scripts, tests)
2. Authoritative YAML configuration parsing and schema validation
3. Workspace secret and credential scanning (zero API keys/tokens)
4. Git working tree hygiene (no untracked heavy weights/parquets, no tracked .env)
5. Parameter budget audit (< 4.0B hard limit, fails closed on error)
6. Pure dataset packaging invariants and referential integrity
7. Automated pytest test suites (contracts, provenance, notebooks, unit, integration)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def print_step(title: str) -> None:
    print(f"\n[+] {title}...")


def fail(msg: str) -> None:
    print(f"\n[X] PRE-PUSH CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def run_gate(cmd: List[str], description: str) -> None:
    print_step(description)
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        fail(f"Step '{description}' exited with non-zero code {res.returncode}")
    print(f"  OK: {description}")


def check_python_compilation() -> None:
    run_gate([sys.executable, "-m", "compileall", "src", "scripts", "tests", "-q"], "Compiling Python syntax")


def check_yaml_syntax() -> None:
    print_step("Validating YAML configuration files")
    import yaml
    from src.task2.config.loader import load_resolved_config

    yaml_files = list(REPO_ROOT.glob("configs/**/*.yaml")) + list(REPO_ROOT.glob("configs/*.yaml"))
    if not yaml_files:
        fail("No YAML config files found under configs/")

    for yf in set(yaml_files):
        try:
            with open(yf, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
            print(f"  OK (syntax): {yf.relative_to(REPO_ROOT)}")
        except Exception as e:
            fail(f"Invalid YAML syntax in {yf}: {e}")

    # Check authoritative task2 configs resolve properly
    base_task2 = REPO_ROOT / "configs" / "task2"
    algo_path = base_task2 / "algorithm.yaml"
    for rt_name in ["kaggle_t4x2.yaml", "colab_t4.yaml", "colab_a100.yaml"]:
        rt_path = base_task2 / "runtime" / rt_name
        try:
            cfg = load_resolved_config(algo_path, rt_path)
            print(f"  OK (resolved): algorithm + {rt_name} (bundle={cfg.bundle_sha256[:12]}...)")
        except Exception as e:
            fail(f"Failed to resolve authoritative config pair ({algo_path.name} + {rt_name}): {e}")


def check_secret_scan() -> None:
    print_step("Scanning workspace for secrets and leaked credentials")
    from src.common.security import assert_no_secrets_in_workspace
    try:
        assert_no_secrets_in_workspace(REPO_ROOT, exclude_tests=True)
        print("  OK: Zero credentials or sensitive tokens detected in workspace.")
    except RuntimeError as e:
        fail(str(e))


def check_notebook_syntax() -> None:
    print_step("Validating Jupyter notebooks")
    notebook_files = [
        REPO_ROOT / "notebooks" / "kaggle_smoke.ipynb",
        REPO_ROOT / "notebooks" / "colab_a100_train.ipynb",
    ]
    for nbf in notebook_files:
        if not nbf.exists():
            fail(f"Required notebook missing: {nbf}")
        try:
            with open(nbf, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "cells" in data, "Notebook missing 'cells'"
            assert len(data["cells"]) >= 5, "Notebook has fewer than 5 cells"
            print(f"  OK: {nbf.relative_to(REPO_ROOT)} ({len(data['cells'])} cells)")
        except Exception as e:
            fail(f"Invalid notebook format in {nbf}: {e}")


def check_git_hygiene() -> None:
    print_step("Auditing Git working tree hygiene")
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(REPO_ROOT))
    lines = res.stdout.strip().splitlines()

    heavy_exts = (".parquet", ".safetensors", ".pt", ".bin", ".npy", ".npz", ".zip", ".tar.gz")
    heavy_artifacts = [
        line for line in lines
        if any(ext in line for ext in heavy_exts) and not line.startswith(" D")
    ]
    if heavy_artifacts:
        fail(f"Found untracked heavy binary artifacts in git status:\n" + "\n".join(heavy_artifacts))

    # Check for tracked .env
    tracked_res = subprocess.run(["git", "ls-files", ".env*"], capture_output=True, text=True, cwd=str(REPO_ROOT))
    tracked_env = [f for f in tracked_res.stdout.strip().splitlines() if not f.endswith(".example")]
    if tracked_env:
        fail(f"Security violation: Tracked .env files found in git: {tracked_env}")

    print("  OK: Git hygiene clean (no untracked heavy binaries, no tracked .env).")


def check_parameter_budget() -> None:
    print_step("Auditing competition parameter budget (< 4.0B)")
    from scripts.audit_parameters import audit_parameter_budget
    try:
        res = audit_parameter_budget(
            config_path=str(REPO_ROOT / "configs" / "task2" / "algorithm.yaml"),
            stack="stack_a",
        )
        total = res["total_learned_parameters"]
        limit = res["limit"]
        if not res["is_compliant"]:
            fail(f"Parameter budget exceeded: {total:,} >= {limit:,} ({res['margin']:,} over)")
        print(f"  OK: Parameter budget compliant: {total:,} / {limit:,} ({res['margin']:,} safe margin)")
    except Exception as e:
        fail(f"Parameter budget audit failed with error: {e}")


def check_dataset_staging() -> None:
    print_step("Validating dataset schema and invariants")
    dataset_dir = REPO_ROOT / "kaggle_dataset"
    if (dataset_dir / "qa_unique.parquet").exists():
        from src.task2.dataset.validator import validate_dataset
        report = validate_dataset(
            data_dir=str(dataset_dir),
            schema_path=str(REPO_ROOT / "configs" / "dataset_schema.yaml"),
        )
        if report["status"] != "PASS":
            fail(f"Dataset validation failed: {report.get('errors')}")
        print(f"  OK: Dataset verified ({len(report.get('tables_validated', {}))} tables, zero code bundle)")
    else:
        print("  NOTICE: Local dataset parquet files not present; skipping heavy byte checks.")


def run_tests(mode: str) -> None:
    print_step(f"Running automated test suite via pytest (mode={mode})")

    if mode == "fast":
        test_paths = [
            "tests/contracts/test_config_authority.py",
            "tests/contracts/test_no_legacy_config_references.py",
            "tests/contracts/test_dependency_lock.py",
            "tests/provenance",
            "tests/notebooks",
            "tests/unit/test_generation_config.py",
        ]
    else:
        test_paths = [
            "tests/contracts",
            "tests/notebooks",
            "tests/unit",
            "tests/integration",
            "tests/provenance",
        ]
        if (REPO_ROOT / "kaggle_dataset" / "qa_unique.parquet").exists():
            test_paths.append("tests/dataset")

    cmd = [sys.executable, "-m", "pytest"] + test_paths + ["-v"]
    run_gate(cmd, f"Pytest ({mode} mode)")


def main() -> None:
    parser = argparse.ArgumentParser(description="LegalQA Task 2 Pre-Push Verification Gate")
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="full",
        help="Verification mode: fast (quick CPU checks) or full (complete suite)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print(f"   LegalQA Task 2 — Canonical Pre-Push Gate [mode: {args.mode.upper()}]   ")
    print("=" * 65)

    check_python_compilation()
    check_yaml_syntax()
    check_secret_scan()
    check_notebook_syntax()
    check_git_hygiene()
    check_parameter_budget()
    check_dataset_staging()
    run_tests(args.mode)

    print("\n" + "=" * 65)
    print(" [PASS] ALL PRE-PUSH CHECKS COMPLETED SUCCESSFULLY!        ")
    print(" Repository is clean, compliant, and ready to promote.")
    print("=" * 65)


if __name__ == "__main__":
    main()
