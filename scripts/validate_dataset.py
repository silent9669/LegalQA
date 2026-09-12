#!/usr/bin/env python3
"""
CLI entry point to validate dataset integrity against canonical schema.
Exits with 0 on PASS, 1 on FAIL.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.task2.dataset.validator import validate_dataset

def main():
    parser = argparse.ArgumentParser(description="Validate LegalQA canonical dataset release")
    parser.add_argument("--data-dir", default="kaggle_dataset/staged", help="Path to dataset directory")
    parser.add_argument("--schema-path", default="configs/dataset_schema.yaml", help="Path to schema YAML")
    parser.add_argument("--report-out", default="kaggle_dataset/staged/validation_report.json", help="Report output path")
    args = parser.parse_args()

    print(f"Validating dataset in: {args.data_dir}")
    print(f"Against schema:        {args.schema_path}")

    report = validate_dataset(data_dir=args.data_dir, schema_path=args.schema_path)

    print("\n=== Validation Summary ===")
    print(f"Status:            {report['status']}")
    print(f"Manifest Verified: {report.get('manifest_verified')}")
    print(f"Tables Validated:  {list(report.get('tables_validated', {}).keys())}")

    if report.get("warnings"):
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f" [!] {w}")

    if report.get("errors"):
        print("\nErrors:")
        for e in report["errors"]:
            print(f" [X] {e}")

    try:
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote validation report to {args.report_out}")
    except Exception as e:
        print(f"Notice: Could not write report to {args.report_out}: {e}")

    if report["status"] != "PASS":
        print("\nDATASET VALIDATION FAILED")
        sys.exit(1)
    else:
        print("\nDATASET VALIDATION PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()
