"""Mine high-value hard negatives from actual BM25 + Dense retrieval candidates for Reranker fine-tuning."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.bm25 import BM25Retriever
from src.common.dense import DenseRetriever
from src.common.evidence import CorpusLookupIndex, mine_retrieval_hard_negatives
from src.common.rrf import reciprocal_rank_fusion


def mine_all_retrieval_negatives(
    data_dir: str = "artifacts/task2/data",
    bm25_dir: str = "artifacts/task2/indexes/bm25",
    dense_dir: str = "artifacts/task2/indexes/dek21",
    output_path: str = "artifacts/task2/data/reranker_training_pairs.parquet",
    max_negatives_per_query: int = 10,
    top_k_retrieve: int = 50,
) -> pd.DataFrame:
    print("=== Mining Retrieval-Grounded Hard Negatives for Reranker Training ===")
    chunks_path = os.path.join(data_dir, "legal_chunks.parquet")
    qa_path = os.path.join(data_dir, "qa_unique.parquet")
    labels_path = os.path.join(data_dir, "retrieval_labels.parquet")

    if not os.path.exists(chunks_path) or not os.path.exists(qa_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing required data artifacts in {data_dir}. Run scripts/prepare_data.py first.")

    print(f"Loading corpus from {chunks_path}...")
    df_chunks = pd.read_parquet(chunks_path)
    lookup = CorpusLookupIndex(df_chunks)
    chunk_text_map = dict(zip(df_chunks["chunk_id"], df_chunks["text_raw"]))

    print(f"Loading retrieval labels from {labels_path}...")
    df_labels = pd.read_parquet(labels_path)
    df_qa = pd.read_parquet(qa_path)

    qa_to_fold = dict(zip(df_qa["qa_id"], df_qa["fold_id"])) if "fold_id" in df_qa.columns else {}

    # Group resolved positive chunks by qa_id
    qa_to_positives: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: {"chunks": set(), "articles": set(), "docs": set()})
    for _, row in df_labels.iterrows():
        qid = str(row["qa_id"]).strip()
        cid = str(row.get("positive_chunk_id", "")).strip()
        art = str(row.get("positive_article_id", "")).strip()
        doc = str(row.get("positive_doc_name", "")).strip()
        if cid:
            qa_to_positives[qid]["chunks"].add(cid)
        if art:
            qa_to_positives[qid]["articles"].add(art)
        if doc:
            qa_to_positives[qid]["docs"].add(doc)

    print("Loading retrievers...")
    bm25 = BM25Retriever.load(bm25_dir, corpus_path=chunks_path) if os.path.exists(bm25_dir) else BM25Retriever()
    if not bm25.corpus:
        bm25.fit(df_chunks.to_dict("records"))

    dense = None
    if os.path.exists(dense_dir) and os.path.exists(os.path.join(dense_dir, "embeddings.npy")):
        try:
            dense = DenseRetriever.load_index(dense_dir, corpus_path=chunks_path)
        except Exception as e:
            print(f"Dense index load skipped ({e}), mining with BM25 only.")

    training_pairs: List[Dict[str, Any]] = []
    qa_list = list(qa_to_positives.keys())
    print(f"Mining hard negatives for {len(qa_list)} labeled queries...")

    qa_row_map = {str(r["qa_id"]): r for _, r in df_qa.iterrows()}

    for qid in tqdm(qa_list, desc="Mining Hard Negatives"):
        if qid not in qa_row_map:
            continue
        q_row = qa_row_map[qid]
        question = str(q_row.get("question_raw") or q_row.get("question", "")).strip()
        if not question:
            continue

        pos_info = qa_to_positives[qid]
        pos_chunks = pos_info["chunks"]
        pos_arts = pos_info["articles"]
        pos_docs = pos_info["docs"]

        # Run BM25 search
        bm25_res = bm25.search(question, top_k=top_k_retrieve)
        dense_res = dense.search(question, top_k=top_k_retrieve) if dense else []

        if bm25_res and dense_res:
            candidates = reciprocal_rank_fusion([bm25_res, dense_res], k=60)
        else:
            candidates = bm25_res or dense_res

        negs = mine_retrieval_hard_negatives(
            qa_id=qid,
            question=question,
            positive_chunk_ids=pos_chunks,
            positive_article_ids=pos_arts,
            positive_doc_names=pos_docs,
            retrieved_candidates=candidates,
            lookup=lookup,
            max_negatives=max_negatives_per_query,
        )

        fold_id = qa_to_fold.get(qid, 0)
        for pos_cid in pos_chunks:
            pos_text = chunk_text_map.get(pos_cid, "")
            for neg in negs:
                neg_cid = neg["negative_chunk_id"]
                neg_text = chunk_text_map.get(neg_cid, "")
                if pos_text and neg_text:
                    training_pairs.append({
                        "qa_id": qid,
                        "fold_id": fold_id,
                        "question": question,
                        "positive_chunk_id": pos_cid,
                        "positive_text": pos_text,
                        "negative_chunk_id": neg_cid,
                        "negative_text": neg_text,
                        "negative_type": neg["negative_type"],
                        "retrieval_rank": neg["retrieval_rank"],
                    })

    df_pairs = pd.DataFrame(training_pairs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_pairs.to_parquet(output_path, index=False)

    print(f"\nSuccessfully generated {len(df_pairs)} training pairs across {len(qa_list)} queries.")
    print(f"Saved reranker training dataset to {output_path}")

    # Summary statistics
    if not df_pairs.empty:
        type_counts = df_pairs["negative_type"].value_counts().to_dict()
        print("Negative Type Breakdown:")
        for k, v in type_counts.items():
            print(f" - {k}: {v:,} ({v/len(df_pairs)*100:.1f}%)")

    return df_pairs


def main():
    parser = argparse.ArgumentParser(description="Mine retrieval hard negatives for Reranker fine-tuning")
    parser.add_argument("--data_dir", default="artifacts/task2/data")
    parser.add_argument("--bm25_dir", default="artifacts/task2/indexes/bm25")
    parser.add_argument("--dense_dir", default="artifacts/task2/indexes/dek21")
    parser.add_argument("--output", default="artifacts/task2/data/reranker_training_pairs.parquet")
    parser.add_argument("--max_negatives", type=int, default=10)
    args = parser.parse_args()

    mine_all_retrieval_negatives(
        data_dir=args.data_dir,
        bm25_dir=args.bm25_dir,
        dense_dir=args.dense_dir,
        output_path=args.output,
        max_negatives_per_query=args.max_negatives,
    )


if __name__ == "__main__":
    main()
