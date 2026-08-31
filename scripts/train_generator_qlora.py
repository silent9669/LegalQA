"""Thin CLI script for QLoRA fine-tuning on Dual T4 GPUs."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.task2.training.train_generator import run_qlora_training


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT Fine-Tuning for Kaggle Dual T4 GPUs")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--qa_path", default="artifacts/task2/data/qa_unique.parquet")
    parser.add_argument("--labels_path", default="artifacts/task2/data/retrieval_labels.parquet")
    parser.add_argument("--chunks_path", default="artifacts/task2/data/legal_chunks.parquet")
    parser.add_argument("--output_dir", default="artifacts/task2/checkpoints/generator/hf_adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_fold", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_qlora_training(
        model_name=args.model,
        qa_path=args.qa_path,
        labels_path=args.labels_path,
        chunks_path=args.chunks_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        val_fold=args.val_fold,
        resume_from_checkpoint=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
