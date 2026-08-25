import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.canonical import build_canonical_qa
from src.data.chunker import process_legal_chunks
from src.data.label_miner import mine_training_labels

def resolve_path(primary: str, fallback: str) -> str:
    return primary if os.path.exists(primary) else fallback

def prepare_all_artifacts():
    os.makedirs("artifacts/data", exist_ok=True)
    os.makedirs("artifacts/chunks", exist_ok=True)
    os.makedirs("artifacts/labels", exist_ok=True)
    os.makedirs("artifacts/submissions", exist_ok=True)

    train_path = resolve_path("data/raw/train.json", "train.json")
    warmup_path = resolve_path("data/raw/warmup.json", "warmup.json")
    chunks_jsonl_path = resolve_path("data/intermediate/chunks_output.jsonl", "chunks_output.jsonl")

    print(f"1. Building canonical qa_unique.parquet and known_qa.json from {train_path}, {warmup_path}...")
    df_unique, memory_dict = build_canonical_qa(train_path, warmup_path)
    df_unique.to_parquet("artifacts/data/qa_unique.parquet", index=False)
    import json
    with open("artifacts/data/known_qa.json", "w", encoding="utf-8") as f:
        json.dump(memory_dict, f, ensure_ascii=False, indent=2)
    print(f"-> Saved qa_unique.parquet ({len(df_unique)} rows) and known_qa.json")

    print(f"2. Processing legal_chunks.parquet from {chunks_jsonl_path}...")
    if os.path.exists(chunks_jsonl_path):
        df_chunks = process_legal_chunks(chunks_jsonl_path, "artifacts/chunks/legal_chunks.parquet")
        print(f"-> Saved legal_chunks.parquet ({len(df_chunks)} rows)")
    else:
        print(f"{chunks_jsonl_path} not found, skipping chunk conversion.")
        if os.path.exists("artifacts/chunks/legal_chunks.parquet"):
            import pandas as pd
            df_chunks = pd.read_parquet("artifacts/chunks/legal_chunks.parquet")
        else:
            df_chunks = None

    if df_chunks is not None:
        print("3. Mining retrieval supervision labels...")
        df_labels = mine_training_labels(df_unique, df_chunks)
        df_labels.to_parquet("artifacts/labels/retrieval_labels.parquet", index=False)
        print(f"-> Saved retrieval_labels.parquet ({len(df_labels)} rows)")

    print("\nAll canonical artifacts successfully prepared!")

if __name__ == "__main__":
    prepare_all_artifacts()
