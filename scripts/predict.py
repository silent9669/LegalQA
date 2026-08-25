import json
import argparse
import os
import sys
import zipfile
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import LegalQAPipeline
from src.data.canonical import build_canonical_qa
from src.memory.exact_memory import ExactMemory
from src.retrieval.bm25_retriever import SimpleBM25
from src.reranking.cross_encoder import SimpleLexicalReranker
from src.postprocess.article_stitcher import ArticleStitcher

def resolve_path(primary: str, *fallbacks: str) -> str:
    if os.path.exists(primary):
        return primary
    for fb in fallbacks:
        if os.path.exists(fb):
            return fb
    return primary

def run_prediction(
    input_json_path: str = "artifacts/raw/public-official.json",
    output_json_path: str = "artifacts/submissions/submission.json",
    train_path: str = "artifacts/raw/train.json",
    warmup_path: str = "artifacts/raw/warmup.json",
    chunks_parquet_path: str = "artifacts/chunks/legal_chunks.parquet"
):
    input_json_path = resolve_path(input_json_path, "data/raw/public-official.json", "public-official.json")
    train_path = resolve_path(train_path, "data/raw/train.json", "train.json")
    warmup_path = resolve_path(warmup_path, "data/raw/warmup.json", "warmup.json")

    print(f"1. Loading Canonical QA & Building Exact Memory from {train_path}, {warmup_path}...")
    df_unique, memory_dict = build_canonical_qa(train_path, warmup_path)
    exact_mem = ExactMemory(memory_dict)

    corpus = []
    retriever = None
    reranker = None
    stitcher = None

    if os.path.exists(chunks_parquet_path):
        print(f"2. Loading legal chunks from {chunks_parquet_path}...")
        df_chunks = pd.read_parquet(chunks_parquet_path)
        for _, row in df_chunks.iterrows():
            corpus.append({
                "chunk_id": str(row["chunk_id"]),
                "id": str(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "context_id": str(row["context_id"]),
                "document_number": str(row["document_number"]) if pd.notna(row["document_number"]) else "",
                "document_title": str(row["document_title"]),
                "name": str(row["name"]),
                "article_number": str(row["article_number"]) if pd.notna(row["article_number"]) else "",
                "article_title": str(row["article_title"]) if pd.notna(row["article_title"]) else "",
                "dieu": str(row["dieu"]) if pd.notna(row["dieu"]) else "",
                "khoan": str(row["khoan"]) if pd.notna(row["khoan"]) else "",
                "clause": str(row["clause"]) if pd.notna(row["clause"]) else "",
                "part": int(row["part"]) if pd.notna(row["part"]) else 1,
                "n_parts": int(row["n_parts"]) if pd.notna(row["n_parts"]) else 1,
                "content": str(row["content"]),
                "text": str(row["searchable_text"]),
                "raw_text": str(row["raw_text"])
            })
        print(f"-> Loaded {len(corpus)} chunks. Building Inverted Index & Article Stitcher...")
        retriever = SimpleBM25(corpus)
        retriever.chunk_map = {doc["id"]: doc for doc in corpus}
        reranker = SimpleLexicalReranker()
        stitcher = ArticleStitcher(corpus)
        print("-> Components successfully initialized.")

    pipeline = LegalQAPipeline(
        exact_memory=exact_mem,
        retriever=retriever,
        reranker=reranker,
        article_stitcher=stitcher
    )

    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    submission = {}
    exact_memory_hits = 0

    print(f"3. Generating predictions for {len(input_data)} items from {input_json_path}...")
    for qid, item in tqdm(input_data.items(), total=len(input_data), desc="Predicting"):
        q_text = item.get('question', '')
        if exact_mem.lookup(qid, q_text):
            exact_memory_hits += 1
        ans = pipeline.predict(qid, q_text)
        submission[str(qid)] = {"answer": ans}

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    # Package into submission.json.zip
    zip_path = output_json_path + ".zip"
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_json_path, arcname=os.path.basename(output_json_path))

    print(f"\n★ Successfully wrote {len(submission)} predictions to {output_json_path}")
    print(f"★ Created submission zip archive: {zip_path}")
    print(f"★ Exact QA Memory Hits: {exact_memory_hits}/{len(input_data)} ({exact_memory_hits/max(1, len(input_data))*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LegalQA Prediction Pipeline")
    parser.add_argument("--input", default="artifacts/raw/public-official.json", help="Path to input test queries JSON")
    parser.add_argument("--output", default="artifacts/submissions/submission.json", help="Path to output submission JSON")
    parser.add_argument("--train", default="artifacts/raw/train.json", help="Path to train.json")
    parser.add_argument("--warmup", default="artifacts/raw/warmup.json", help="Path to warmup.json")
    parser.add_argument("--chunks", default="artifacts/chunks/legal_chunks.parquet", help="Path to legal_chunks.parquet")
    args = parser.parse_args()

    run_prediction(args.input, args.output, args.train, args.warmup, args.chunks)
