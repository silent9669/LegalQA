"""Build sparse BM25 and dense DEk21 indexes on canonical legal chunks."""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever


def build_bm25_index(chunks_path: str, index_dir: str) -> BM25Retriever:
    print(f"Loading legal chunks from {chunks_path}...")
    df = pd.read_parquet(chunks_path)
    records = df.to_dict("records")

    print(f"Building BM25 Index for {len(records)} chunks...")
    retriever = BM25Retriever()
    retriever.fit(records)

    retriever.save(index_dir, save_corpus_meta=False)
    print(f"BM25 index saved to {index_dir} without duplicate corpus metadata.")
    return retriever


def build_dek21_index(
    chunks_path: str,
    index_dir: str,
    model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
    batch_size: int = 128,
    use_mock: bool = False,
) -> DEk21Retriever:
    print(f"Loading legal chunks from {chunks_path}...")
    df = pd.read_parquet(chunks_path)
    records = df.to_dict("records")

    print(f"Building DEk21 Embeddings for {len(records)} chunks (mock={use_mock}, batch_size={batch_size})...")
    retriever = DEk21Retriever(model_name="mock" if use_mock else model_name)
    if use_mock:
        retriever.fit_mock(records[:1000])
    else:
        retriever.fit(records, batch_size=batch_size, show_progress=True)

    retriever.save_index(index_dir)
    print(f"DEk21 embeddings saved to {index_dir}.")
    return retriever


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indexes for LegalQA.")
    parser.add_argument("--chunks", default="artifacts/task2/data/legal_chunks.parquet", help="Path to legal_chunks.parquet")
    parser.add_argument("--bm25_dir", default="artifacts/task2/indexes/bm25", help="BM25 index output directory")
    parser.add_argument("--dek21_dir", default="artifacts/task2/indexes/dek21", help="DEk21 index output directory")
    parser.add_argument("--skip_bm25", action="store_true", help="Skip BM25 index building")
    parser.add_argument("--build_dek21", action="store_true", help="Build DEk21 dense index")
    parser.add_argument("--dek21_mock", action="store_true", help="Build mock DEk21 index for fast verification")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for dense encoding")
    args = parser.parse_args()

    if not os.path.exists(args.chunks):
        print(f"Error: Chunks file not found at {args.chunks}. Run scripts/prepare_data.py first.", file=sys.stderr)
        sys.exit(1)

    if not args.skip_bm25:
        build_bm25_index(args.chunks, args.bm25_dir)

    if args.build_dek21 or args.dek21_mock:
        build_dek21_index(args.chunks, args.dek21_dir, batch_size=args.batch_size, use_mock=args.dek21_mock)

    print("Index build finished successfully!")


if __name__ == "__main__":
    main()
