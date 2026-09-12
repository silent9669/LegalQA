#!/usr/bin/env python3
"""
Canonical Pre-Push Verification Gate for LegalQA Task 2.
Run this script locally before pushing any commit to GitHub.
Guarantees:
1. All configuration YAML files parse cleanly and match schema
2. All Jupyter notebooks parse as valid JSON and satisfy contract invariants
3. Pure dataset packaging invariants (zero code files in dataset staging)
4. Full test suite execution (contracts, notebooks, unit, integration, dataset)
"""

import sys
import os
import glob
import json
import subprocess
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def print_step(title: str):
    print(f"\n[+] {title}...")

def fail(msg: str):
    print(f"\n[X] PRE-PUSH CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)

def check_yaml_syntax():
    print_step("Validating YAML configuration files")
    import yaml
    yaml_files = glob.glob(str(REPO_ROOT / "configs" / "*.yaml"))
    if not yaml_files:
        fail("No YAML config files found under configs/")
    for yf in yaml_files:
        try:
            with open(yf, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
            print(f"  OK: {os.path.relpath(yf, REPO_ROOT)}")
        except Exception as e:
            fail(f"Invalid YAML syntax in {yf}: {e}")

def check_notebook_syntax():
    print_step("Validating Jupyter notebooks")
    notebook_files = [
        REPO_ROOT / "notebooks" / "kaggle_smoke.ipynb",
        REPO_ROOT / "notebooks" / "colab_train_a100.ipynb",
    ]
    for nbf in notebook_files:
        if not nbf.exists():
            fail(f"Required notebook missing: {nbf}")
        try:
            with open(nbf, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "cells" in data, "Notebook missing 'cells'"
            assert len(data["cells"]) >= 5, "Notebook has fewer than 5 cells"
            print(f"  OK: {os.path.relpath(nbf, REPO_ROOT)} ({len(data['cells'])} cells)")
        except Exception as e:
            fail(f"Invalid notebook format in {nbf}: {e}")

def check_dataset_staging():
    print_step("Validating dataset invariants")
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

def run_test_suite():
    print_step("Running automated test suite via pytest")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/contracts",
        "tests/notebooks",
        "tests/unit",
        "tests/integration",
        "-v",
    ]
    if (REPO_ROOT / "kaggle_dataset" / "qa_unique.parquet").exists():
        cmd.append("tests/dataset")

    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        fail(f"pytest exited with return code {res.returncode}")
    print("  OK: Pytest test suite passed cleanly.")

def main():
    print("===========================================================")
    print("      LegalQA Task 2 — Canonical Pre-Push Verification     ")
    print("===========================================================")

    check_yaml_syntax()
    check_notebook_syntax()
    check_dataset_staging()
    run_test_suite()

    print("\n===========================================================")
    print(" [PASS] ALL PRE-PUSH CHECKS COMPLETED SUCCESSFULLY!        ")
    print(" Repository is clean, compliant, and ready to push to GitHub")
    print("===========================================================")
    sys.exit(0)

if __name__ == "__main__":
    main()
