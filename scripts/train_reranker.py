"""Thin CLI script to trigger BGE Reranker fine-tuning."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.task2.training.train_reranker import train_bge_reranker


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Cross-Encoder Reranker on LegalQA")
    parser.add_argument("--pairs_path", default="artifacts/task2/data/reranker_training_pairs.parquet")
    parser.add_argument("--output_dir", default="artifacts/task2/checkpoints/reranker/best")
    parser.add_argument("--model_name", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_fold", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_bge_reranker(
        pairs_path=args.pairs_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fold=args.val_fold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
