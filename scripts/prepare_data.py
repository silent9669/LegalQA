import glob
import json
import os
import sys
import multiprocessing
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.legal_parser import parse_legal_document
from src.task2.qa_memory import QAMemory
from src.common.evidence import parse_citations_from_answer, mine_hard_negatives

def _parse_single_file(fp: str) -> list[dict]:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        doc_id = str(data.get("id", "")).strip()
        doc_name = str(data.get("name", "")).strip()
        passage = str(data.get("passage", "")).strip()
        return parse_legal_document(doc_id=doc_id, doc_name=doc_name, passage=passage)
    except Exception as e:
        return []

def prepare_legal_chunks(raw_contexts_dir: str, output_parquet: str) -> pd.DataFrame:
    print(f"Scanning raw context JSONs in {raw_contexts_dir}...")
    files = glob.glob(os.path.join(raw_contexts_dir, "*.json"))
    print(f"Found {len(files)} context files. Processing with {multiprocessing.cpu_count()} CPU cores...")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        chunk_lists = list(tqdm(pool.imap_unordered(_parse_single_file, files, chunksize=50), total=len(files), desc="Parsing Legal Chunks"))

    all_chunks = [c for sublist in chunk_lists for c in sublist]
    print(f"Parsed total {len(all_chunks)} legal chunks.")
    df_chunks = pd.DataFrame(all_chunks)
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_chunks.to_parquet(output_parquet, index=False)
    print(f"Saved legal chunks to {output_parquet}")
    return df_chunks

def prepare_qa_data(train_path: str, warmup_path: str, out_data_dir: str, all_chunks: list[dict]):
    print("Loading official QA records...")
    records = []
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
            if isinstance(t_data, dict):
                for k, v in t_data.items():
                    records.append({"id": k, "question": v.get("question", ""), "answer": v.get("answer", ""), "source_split": "train"})
            elif isinstance(t_data, list):
                for r in t_data:
                    r["source_split"] = "train"
                    records.append(r)

    if os.path.exists(warmup_path):
        with open(warmup_path, "r", encoding="utf-8") as f:
            w_data = json.load(f)
            if isinstance(w_data, dict):
                for k, v in w_data.items():
                    records.append({"id": k, "question": v.get("question", ""), "answer": v.get("answer", ""), "source_split": "warmup"})
            elif isinstance(w_data, list):
                for r in w_data:
                    r["source_split"] = "warmup"
                    records.append(r)

    print(f"Total loaded QA records: {len(records)}")
    os.makedirs(out_data_dir, exist_ok=True)

    # 1. QA Memory & Unique QA
    qa_memory = QAMemory.from_records(records)
    known_qa_json = os.path.join(out_data_dir, "known_qa.json")
    qa_unique_parquet = os.path.join(out_data_dir, "qa_unique.parquet")
    qa_memory.save(known_qa_json, qa_unique_parquet)
    print(f"Saved {len(qa_memory.df)} unique QA pairs.")

    # 2. QA Citations & Retrieval Labels
    citation_rows = []
    label_rows = []

    for r in tqdm(records, desc="Mining Citations & Labels"):
        ans = r.get("answer", "")
        citations = parse_citations_from_answer(ans)
        qa_id = r.get("id", "")
        q = r.get("question", "")

        for cit in citations:
            citation_rows.append({
                "qa_id": qa_id,
                "question": q,
                "doc_number": cit["doc_number"],
                "article": cit["article"],
                "clause": cit["clause"]
            })

    if citation_rows:
        df_cit = pd.DataFrame(citation_rows)
        df_cit.to_parquet(os.path.join(out_data_dir, "qa_citations.parquet"), index=False)
        print(f"Saved {len(df_cit)} citation records.")

def main():
    raw_contexts = "artifacts/raw/selected-contexts"
    chunks_out = "artifacts/task2/data/legal_chunks.parquet"
    train_json = "artifacts/raw/train.json"
    warmup_json = "artifacts/raw/warmup.json"
    data_dir = "artifacts/task2/data"

    if os.path.exists(raw_contexts):
        df_chunks = prepare_legal_chunks(raw_contexts, chunks_out)
        prepare_qa_data(train_json, warmup_json, data_dir, df_chunks.to_dict("records"))
    else:
        print(f"Raw contexts directory not found at {raw_contexts}")

if __name__ == "__main__":
    main()
