"""Thin CLI script to evaluate exact trained checkpoints on held-out validation fold."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.task2.evaluation import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate exact trained checkpoints on held-out fold")
    parser.add_argument("--qa_path", default="artifacts/task2/data/qa_unique.parquet")
    parser.add_argument("--fold_path", default="artifacts/task2/data/fold_assignments.parquet")
    parser.add_argument("--chunks_path", default="artifacts/task2/data/legal_chunks.parquet")
    parser.add_argument("--held_out_fold", type=int, default=0)
    parser.add_argument("--bm25_dir", default="artifacts/task2/indexes/bm25")
    parser.add_argument("--dense_dir", default="artifacts/task2/indexes/dek21")
    parser.add_argument("--dense_model", default="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2")
    parser.add_argument("--reranker_checkpoint", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--generator_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--sample_size", type=int, default=50)
    parser.add_argument("--eval_output_dir", default="artifacts/task2/evaluations")
    args = parser.parse_args()

    evaluate_checkpoint(
        qa_path=args.qa_path,
        fold_path=args.fold_path,
        chunks_path=args.chunks_path,
        held_out_fold=args.held_out_fold,
        bm25_dir=args.bm25_dir,
        dense_dir=args.dense_dir,
        dense_model=args.dense_model,
        reranker_checkpoint=args.reranker_checkpoint,
        generator_model=args.generator_model,
        adapter_path=args.adapter,
        sample_size=args.sample_size,
        eval_output_dir=args.eval_output_dir,
    )


if __name__ == "__main__":
    main()
