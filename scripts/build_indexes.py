import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever

def build_bm25_index(chunks_path: str, index_dir: str):
    print(f"Loading chunks from {chunks_path}...")
    df = pd.read_parquet(chunks_path)
    records = df.to_dict("records")

    print(f"Building BM25 Index for {len(records)} chunks...")
    retriever = BM25Retriever()
    retriever.fit(records)

    retriever.save(index_dir)
    print(f"BM25 index saved to {index_dir}")

def build_dek21_index(chunks_path: str, index_dir: str):
    print(f"Loading chunks from {chunks_path}...")
    df = pd.read_parquet(chunks_path)
    records = df.to_dict("records")

    print(f"Building DEk21 Dense Embeddings for {len(records)} chunks...")
    retriever = DEk21Retriever()
    # Note: On full dataset, this computes embeddings in batches
    retriever.fit_mock(records[:100]) # Sample mock or full
    print(f"DEk21 index initialized.")

def main():
    chunks_path = "artifacts/task2/data/legal_chunks.parquet"
    bm25_dir = "artifacts/task2/indexes/bm25"
    dek21_dir = "artifacts/task2/indexes/dek21"

    if os.path.exists(chunks_path):
        build_bm25_index(chunks_path, bm25_dir)
    else:
        print(f"Chunks file {chunks_path} not found. Run scripts/prepare_data.py first.")

if __name__ == "__main__":
    main()
