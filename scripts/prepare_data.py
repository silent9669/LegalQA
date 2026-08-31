"""Prepare and build all canonical LegalQA data artifacts, citations, retrieval labels, and fold assignments."""

from __future__ import annotations

import glob
import hashlib
import json
import multiprocessing
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.evidence import (
    CorpusLookupIndex,
    mine_hard_negatives,
    parse_citations_from_answer,
    resolve_citations_to_chunks,
)
from src.common.legal_parser import parse_legal_document
from src.common.normalize import normalize_question
from src.task2.qa_memory import QAMemory


def _parse_single_file(fp: str) -> List[Dict[str, Any]]:
    """Worker function to parse a single raw context JSON file."""
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        doc_id = str(data.get("id", "")).strip()
        doc_name = str(data.get("name", "")).strip()
        passage = str(data.get("passage", "")).strip()
        if not doc_id or not passage:
            return []
        return parse_legal_document(doc_id=doc_id, doc_name=doc_name, passage=passage)
    except Exception as e:
        print(f"Warning: Failed to parse context file {fp}: {e}", file=sys.stderr)
        return []


def prepare_legal_chunks(raw_contexts_dir: str, output_parquet: str) -> pd.DataFrame:
    """Parse all context JSON files in directory into structured legal chunks."""
    print(f"Scanning raw context JSONs in {raw_contexts_dir}...")
    files = sorted(glob.glob(os.path.join(raw_contexts_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON context files found in {raw_contexts_dir}")

    print(f"Found {len(files)} context files. Parsing with {multiprocessing.cpu_count()} CPU cores...")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        chunk_lists = list(tqdm(pool.imap_unordered(_parse_single_file, files, chunksize=50), total=len(files), desc="Parsing Legal Chunks"))

    all_chunks = [c for sublist in chunk_lists for c in sublist]
    print(f"Parsed total {len(all_chunks)} legal chunks.")
    df_chunks = pd.DataFrame(all_chunks)

    # Ensure required columns are present
    required_cols = [
        "chunk_id", "doc_id", "doc_name", "legal_number", "year",
        "chapter_number", "section_number", "article_number", "clause_number",
        "point_label", "parent_article_id", "parent_clause_id",
        "text_raw", "text_norm", "start_char", "end_char"
    ]
    for col in required_cols:
        if col not in df_chunks.columns:
            df_chunks[col] = None

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_chunks.to_parquet(output_parquet, index=False)
    print(f"Saved legal chunks to {output_parquet} ({len(df_chunks)} rows).")
    return df_chunks


def assign_group_blocked_folds(df_qa: pd.DataFrame, num_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Assign fold_id (0..num_folds-1) deterministically grouped by question_norm to prevent leakage."""
    unique_groups = df_qa["question_norm"].unique()

    def get_fold(q_norm: str) -> int:
        h = int(hashlib.md5(f"{seed}_{q_norm}".encode("utf-8")).hexdigest(), 16)
        return h % num_folds

    group_to_fold = {q: get_fold(q) for q in unique_groups}
    df_qa["fold_id"] = df_qa["question_norm"].map(group_to_fold)
    return df_qa


def prepare_qa_and_labels(
    train_path: str,
    warmup_path: str,
    out_data_dir: str,
    df_chunks: pd.DataFrame,
    num_folds: int = 5,
) -> None:
    """Load QA records, construct QA memory, mine citations, resolve retrieval labels, and build fold assignments."""
    print("Loading official QA records...")
    records: List[Dict[str, Any]] = []

    for path, split_name in [(train_path, "train"), (warmup_path, "warmup")]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    records.append({
                        "id": str(k).strip(),
                        "question": str(v.get("question", "")).strip(),
                        "answer": str(v.get("answer", "")).strip(),
                        "source_split": split_name,
                    })
            elif isinstance(data, list):
                for r in data:
                    records.append({
                        "id": str(r.get("id", "")).strip(),
                        "question": str(r.get("question", "")).strip(),
                        "answer": str(r.get("answer", "")).strip(),
                        "source_split": split_name,
                    })

    print(f"Loaded total {len(records)} QA records.")
    os.makedirs(out_data_dir, exist_ok=True)

    # 1. Build and save QA Memory
    qa_memory = QAMemory.from_records(records)
    df_qa = qa_memory.df.copy()

    # Assign deterministic group-blocked folds
    df_qa = assign_group_blocked_folds(df_qa, num_folds=num_folds)

    # Save known_qa.json and qa_unique.parquet
    known_qa_json = os.path.join(out_data_dir, "known_qa.json")
    qa_unique_parquet = os.path.join(out_data_dir, "qa_unique.parquet")
    qa_memory.df = df_qa
    qa_memory.save(known_qa_json, qa_unique_parquet)
    print(f"Saved {len(df_qa)} QA records to {qa_unique_parquet} (Conflicts: {len(qa_memory.conflicts)}).")

    # 2. Save Fold Assignments table
    fold_df = df_qa[["qa_id", "question_norm", "fold_id", "source_split", "is_conflict"]].copy()
    fold_parquet = os.path.join(out_data_dir, "fold_assignments.parquet")
    fold_df.to_parquet(fold_parquet, index=False)
    print(f"Saved fold assignments to {fold_parquet}.")

    # 3. Mine Citations and Resolve to Chunks with Fast O(1) Lookup
    print("Building corpus lookup index for fast citation resolution...")
    lookup = CorpusLookupIndex(df_chunks)

    citation_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []

    qa_to_fold = dict(zip(df_qa["qa_id"], df_qa["fold_id"]))

    resolved_count = 0
    total_citations = 0

    for r in tqdm(records, desc="Mining Citations & Resolving Labels"):
        qa_id = r["id"]
        q = r["question"]
        ans = r["answer"]
        fold_id = qa_to_fold.get(qa_id, 0)

        cits = parse_citations_from_answer(ans)
        total_citations += len(cits)

        for cit in cits:
            citation_rows.append({
                "qa_id": qa_id,
                "question": q,
                "doc_identifier": cit.get("doc_identifier", ""),
                "article": cit.get("article", ""),
                "clause": cit.get("clause", ""),
            })

        resolved = resolve_citations_to_chunks(cits, df_chunks, lookup)
        if resolved:
            resolved_count += 1
            for res in resolved:
                pos_chunk_id = res["positive_chunk_id"]
                pos_art_id = res["positive_article_id"]
                pos_doc_name = res["doc_names"][0] if res["doc_names"] else ""

                negs = mine_hard_negatives(
                    positive_chunk_id=pos_chunk_id,
                    positive_article_id=pos_art_id,
                    positive_doc_name=pos_doc_name,
                    chunks_df=lookup,
                    max_per_type=5,
                )

                label_rows.append({
                    "qa_id": qa_id,
                    "question": q,
                    "fold_id": fold_id,
                    "positive_chunk_id": pos_chunk_id,
                    "positive_article_id": pos_art_id,
                    "positive_doc_name": pos_doc_name,
                    "hard_negatives": negs,
                    "num_negatives": sum(len(v) for v in negs.values()),
                })

    # Save qa_citations.parquet
    if citation_rows:
        df_cit = pd.DataFrame(citation_rows)
        df_cit.to_parquet(os.path.join(out_data_dir, "qa_citations.parquet"), index=False)
        print(f"Saved {len(df_cit)} citations to {os.path.join(out_data_dir, 'qa_citations.parquet')}.")

    # Save retrieval_labels.parquet
    if label_rows:
        df_labels = pd.DataFrame(label_rows)
        df_labels.to_parquet(os.path.join(out_data_dir, "retrieval_labels.parquet"), index=False)
        print(f"Saved {len(df_labels)} retrieval supervision labels to {os.path.join(out_data_dir, 'retrieval_labels.parquet')}.")
        print(f"Citation Resolution Summary: {resolved_count}/{len(records)} QA items resolved to positive corpus chunks ({resolved_count/len(records)*100:.1f}%).")
    else:
        print("Warning: No retrieval labels could be resolved from citations!", file=sys.stderr)


def main():
    raw_contexts = "artifacts/raw/selected-contexts"
    chunks_out = "artifacts/task2/data/legal_chunks.parquet"
    train_json = "artifacts/raw/train.json"
    warmup_json = "artifacts/raw/warmup.json"
    data_dir = "artifacts/task2/data"

    if not os.path.exists(raw_contexts):
        print(f"Error: Raw contexts directory not found at {raw_contexts}", file=sys.stderr)
        sys.exit(1)

    df_chunks = prepare_legal_chunks(raw_contexts, chunks_out)
    prepare_qa_and_labels(train_json, warmup_json, data_dir, df_chunks)
    print("\nData preparation and canonical modeling completed successfully!")


if __name__ == "__main__":
    main()
