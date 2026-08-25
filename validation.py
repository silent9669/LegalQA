import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.run_oof_validation import run_5fold_oof_validation

def main():
    parser = argparse.ArgumentParser(
        description="Run CodaBench 5-Fold OOF Validation for DSC 2026 Task 2 LegalQA"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to evaluate for validation (default: 100, pass 0 for full dataset)"
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--train",
        default="data/raw/train.json",
        help="Path to train.json"
    )
    parser.add_argument(
        "--warmup",
        default="data/raw/warmup.json",
        help="Path to warmup.json"
    )
    parser.add_argument(
        "--chunks",
        default="artifacts/chunks/legal_chunks.parquet",
        help="Path to legal_chunks.parquet"
    )
    args = parser.parse_args()

    sample_limit = args.samples if args.samples > 0 else None
    print(f"Starting CodaBench 5-Fold OOF Validation Benchmark (samples={sample_limit}, splits={args.splits})...")
    scores = run_5fold_oof_validation(
        train_path=args.train,
        warmup_path=args.warmup,
        chunks_parquet_path=args.chunks,
        n_splits=args.splits,
        sample_limit=sample_limit
    )
    print("Validation finished successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
