import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.canonical import build_canonical_qa, normalize_vietnamese_text
from src.evaluation.codabench_eval import evaluate_predictions
from src.memory.exact_memory import ExactMemory
from src.retrieval.bm25_retriever import SimpleBM25
from src.reranking.cross_encoder import SimpleLexicalReranker
from src.postprocess.article_stitcher import ArticleStitcher
from src.pipeline import LegalQAPipeline

def resolve_path(primary: str, fallback: str) -> str:
    return primary if os.path.exists(primary) else fallback

def kfold_split(n_samples: int, n_splits: int = 5, shuffle: bool = True, seed: int = 42):
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.setdiff1d(indices, val_idx)
        yield train_idx, val_idx

def run_5fold_oof_validation(
    train_path: str = "data/raw/train.json",
    warmup_path: str = "data/raw/warmup.json",
    chunks_parquet_path: str = "artifacts/chunks/legal_chunks.parquet",
    n_splits: int = 5,
    sample_limit: int = 250
):
    train_path = resolve_path(train_path, "train.json")
    warmup_path = resolve_path(warmup_path, "warmup.json")

    print("=== Step 1: Loading Datasets & Building Canonical QA ===")
    df_qa, full_mem_dict = build_canonical_qa(train_path, warmup_path)
    print(f"Total Canonical QA Pairs: {len(df_qa)}")

    if sample_limit and sample_limit < len(df_qa):
        print(f"Evaluating representative slice of {sample_limit} QA for rigorous 5-fold OOF benchmark...")
        df_eval = df_qa.sample(n=sample_limit, random_state=42).reset_index(drop=True)
    else:
        df_eval = df_qa.reset_index(drop=True)

    print("=== Step 2: Loading Legal Chunks & Building Article Stitcher ===")
    corpus = []
    if os.path.exists(chunks_parquet_path):
        print(f"Reading legal chunks from {chunks_parquet_path}...")
        df_c = pd.read_parquet(chunks_parquet_path)
        for _, row in df_c.iterrows():
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
                "text": str(row["searchable_text"])
            })
        print(f"Loaded {len(corpus)} legal chunks. Building Inverted Index...")
    else:
        raise FileNotFoundError(f"Missing {chunks_parquet_path}")

    retriever = SimpleBM25(corpus)
    retriever.chunk_map = {doc["id"]: doc for doc in corpus}
    reranker = SimpleLexicalReranker()
    stitcher = ArticleStitcher(corpus)
    print("Article Stitcher and Inverted BM25 Index successfully initialized.")

    print(f"\n=== Step 3: Running {n_splits}-Fold OOF Validation ===")
    oof_predictions = {}
    oof_ground_truth = {}
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold_split(len(df_eval), n_splits=n_splits, seed=42)):
        val_df = df_eval.iloc[val_idx]
        train_df = df_eval.iloc[train_idx]

        # Fold-isolated exact memory (zero data leakage)
        by_id_fold = {row['id']: row['answer'] for _, row in train_df.iterrows()}
        by_q_fold = {row['normalized_question']: row['answer'] for _, row in train_df.iterrows()}
        fold_mem = ExactMemory({"by_id": by_id_fold, "by_question": by_q_fold})

        pipeline = LegalQAPipeline(
            exact_memory=fold_mem,
            retriever=retriever,
            reranker=reranker,
            article_stitcher=stitcher
        )

        fold_preds = {}
        fold_truth = {}

        for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Fold {fold+1}/{n_splits}"):
            qid = str(row['id'])
            q_text = row['question']
            gold_ans = row['answer']

            pred_ans = pipeline.predict(qid, q_text)

            fold_preds[qid] = {"answer": pred_ans}
            fold_truth[qid] = gold_ans
            oof_predictions[qid] = {"answer": pred_ans}
            oof_ground_truth[qid] = gold_ans

        fold_metric = evaluate_predictions(fold_preds, fold_truth)
        fold_scores.append(fold_metric)
        print(f"Fold {fold+1} Scores -> METEOR = {fold_metric['meteor']:.4f}, ROUGE-L = {fold_metric['rouge']:.4f}")

    print("\n=========================================================")
    print("      FINAL 5-FOLD OUT-OF-FOLD (OOF) BENCHMARK REPORT    ")
    print("=========================================================")
    overall_metric = evaluate_predictions(oof_predictions, oof_ground_truth)
    print(f"★ Overall OOF METEOR : {overall_metric['meteor']:.4f}")
    print(f"★ Overall OOF ROUGE-L: {overall_metric['rouge']:.4f}")

    mean_meteor = np.mean([s['meteor'] for s in fold_scores])
    std_meteor = np.std([s['meteor'] for s in fold_scores])
    mean_rouge = np.mean([s['rouge'] for s in fold_scores])
    std_rouge = np.std([s['rouge'] for s in fold_scores])
    print(f"★ 5-Fold Mean METEOR : {mean_meteor:.4f} ± {std_meteor:.4f}")
    print(f"★ 5-Fold Mean ROUGE-L: {mean_rouge:.4f} ± {std_rouge:.4f}")
    print("=========================================================\n")

    return overall_metric

if __name__ == "__main__":
    run_5fold_oof_validation(sample_limit=100)
