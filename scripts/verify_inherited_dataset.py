#!/usr/bin/env python3
"""CLI tool to verify inherited dataset bytes against authoritative provenance manifest."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task2.data_contract import verify_inherited_dataset


def main():
    parser = argparse.ArgumentParser(description="Verify inherited dataset provenance and file integrity")
    parser.add_argument("--root", required=True, type=Path, help="Root directory containing dataset")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "inherited" / "dataset_manifest_v15.json",
        help="Path to authoritative dataset manifest",
    )
    parser.add_argument("--no-hashes", action="store_true", help="Skip SHA256 verification, check existence only")
    parser.add_argument("--check-structural", action="store_true", help="Check row counts and tensor shapes")

    args = parser.parse_args()

    print(f"Verifying dataset at: {args.root}")
    print(f"Using manifest: {args.manifest}")

    result = verify_inherited_dataset(
        root=args.root,
        manifest_path=args.manifest,
        check_hashes=not args.no_hashes,
        check_structural=args.check_structural,
    )

    print(f"\nResult: {'PASS' if result.is_valid else 'FAIL'}")
    print(result.summary)

    if result.missing_files:
        print("\nMissing files:")
        for f in result.missing_files:
            print(f"  - {f}")

    if result.hash_mismatches:
        print("\nHash mismatches:")
        for m in result.hash_mismatches:
            print(f"  - {m}")

    if result.structural_mismatches:
        print("\nStructural errors:")
        for s in result.structural_mismatches:
            print(f"  - {s}")

    sys.exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    main()
