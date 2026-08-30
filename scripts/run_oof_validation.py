import argparse
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.translate.meteor_score import meteor_score

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever
from src.common.rrf import reciprocal_rank_fusion
from src.common.reranker import BGEReranker
from src.common.normalize import clean_legal_text, normalize_question
from src.task2.qa_memory import QAMemory
from src.task2.article_stitcher import ArticleStitcher
from src.task2.generator import QwenGenerator
from src.task2.source_snap import snap_facts_to_evidence, select_best_answer_candidate
from src.task2.predict import LegalQAPipeline

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def calculate_official_meteor(references: list[str], predictions: list[str]) -> float:
    scores = []
    for r, p in zip(references, predictions):
        r_tokens = str(r).split()
        p_tokens = str(p).split()
        scores.append(meteor_score([r_tokens], p_tokens))
    return float(np.mean(scores))

def calculate_rouge_l(references: list[str], predictions: list[str]) -> float:
    if rouge_scorer is None:
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = []
    for r, p in zip(references, predictions):
        sc = scorer.score(str(r), str(p))['rougeL'].fmeasure
        scores.append(sc)
    return float(np.mean(scores))

def assign_question_blocked_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    def _hash_question(q: str) -> int:
        norm = normalize_question(q)
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        return int(h, 16) % n_splits

    df["fold_id"] = df["question_raw"].apply(_hash_question)
    return df

def run_oof_validation(qa_path: str, bm25_dir: str, num_eval_samples: int = 100, n_splits: int = 5):
    print(f"Loading QA dataset from {qa_path}...")
    df_qa = pd.read_parquet(qa_path)
    df_qa = assign_question_blocked_folds(df_qa, n_splits=n_splits)

    print(f"Loading BM25 index from {bm25_dir}...")
    bm25 = BM25Retriever.load(bm25_dir)
    dense = DEk21Retriever(model_name="mock")
    reranker = BGEReranker(model_name="mock")
    stitcher = ArticleStitcher(bm25.corpus)
    generator = QwenGenerator(runtime="fallback")

    all_oof_results = []
    fold_meteors = []
    fold_rouges = []

    samples_per_fold = max(1, num_eval_samples // n_splits)
    print(f"Evaluating {samples_per_fold} samples per fold across {n_splits} folds...")

    for fold_id in range(n_splits):
        val_subset = df_qa[df_qa["fold_id"] == fold_id].head(samples_per_fold)
        train_subset = df_qa[df_qa["fold_id"] != fold_id]

        # Zero-leakage fold memory: val queries cannot lookup themselves
        fold_mem = QAMemory.from_records(train_subset.to_dict("records"))
        pipeline = LegalQAPipeline(fold_mem, bm25, dense, reranker, stitcher, generator)

        fold_preds = []
        fold_refs = []

        for _, row in tqdm(val_subset.iterrows(), total=len(val_subset), desc=f"Fold {fold_id}"):
            qid = row["qa_id"]
            q = row["question_raw"]
            ref_ans = row["answer_raw"]

            pred = pipeline.predict_single(qid, q)
            fold_preds.append(pred)
            fold_refs.append(ref_ans)

            all_oof_results.append({
                "qa_id": qid,
                "fold_id": fold_id,
                "question": q,
                "reference": ref_ans,
                "prediction": pred,
                "meteor": meteor_score([ref_ans.split()], pred.split())
            })

        fold_m = calculate_official_meteor(fold_refs, fold_preds)
        fold_r = calculate_rouge_l(fold_refs, fold_preds)
        fold_meteors.append(fold_m)
        fold_rouges.append(fold_r)
        print(f"Fold {fold_id} -> METEOR: {fold_m:.4f} | ROUGE-L: {fold_r:.4f}")

    mean_meteor = float(np.mean(fold_meteors))
    std_meteor = float(np.std(fold_meteors))
    mean_rouge = float(np.mean(fold_rouges))

    print("\n=======================================================")
    print(f"5-Fold OOF METEOR:  {mean_meteor:.4f} ± {std_meteor:.4f}")
    print(f"5-Fold OOF ROUGE-L: {mean_rouge:.4f}")
    print("=======================================================")

    df_oof = pd.DataFrame(all_oof_results)
    oof_out = "artifacts/task2/data/oof_predictions.parquet"
    df_oof.to_parquet(oof_out, index=False)
    print(f"Saved OOF predictions to {oof_out}")

def main():
    parser = argparse.ArgumentParser(description="LegalQA Task 2 5-Fold OOF Validation")
    parser.add_argument("--qa_path", default="artifacts/task2/data/qa_unique.parquet")
    parser.add_argument("--bm25_dir", default="artifacts/task2/indexes/bm25")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    if os.path.exists(args.qa_path) and os.path.exists(args.bm25_dir):
        run_oof_validation(args.qa_path, args.bm25_dir, num_eval_samples=args.samples, n_splits=args.folds)
    else:
        print(f"Prerequisites missing. Ensure {args.qa_path} and {args.bm25_dir} exist.")

if __name__ == "__main__":
    main()
