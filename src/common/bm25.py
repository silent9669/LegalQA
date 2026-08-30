import json
import os
import math
import numpy as np
import pandas as pd
from collections import Counter
from src.common.normalize import clean_legal_text, tokenize_vietnamese, extract_legal_signals

try:
    import bm25s
except ImportError:
    bm25s = None

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_ids = []
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.df = Counter()
        self.corpus_size = 0
        self.bm25s_index = None

    def fit(self, corpus: list[dict]):
        self.corpus = corpus
        self.doc_ids = [c["chunk_id"] for c in corpus]
        self.corpus_size = len(corpus)

        tokenized_corpus = []
        for c in corpus:
            tokens = c.get("text_norm", "").split()
            if not tokens:
                tokens = tokenize_vietnamese(c.get("text_raw", "")).split()
            tokenized_corpus.append(tokens)
            self.doc_len.append(len(tokens))
            for t in set(tokens):
                self.df[t] += 1

        self.avg_doc_len = sum(self.doc_len) / max(1, self.corpus_size)

        if bm25s is not None and self.corpus_size > 0:
            try:
                self.bm25s_index = bm25s.BM25(k1=self.k1, b=self.b)
                self.bm25s_index.index(tokenized_corpus)
            except Exception:
                self.bm25s_index = None

    def search(self, query: str, top_k: int = 60) -> list[dict]:
        if not self.corpus:
            return []

        signals = extract_legal_signals(query)
        seg_query = tokenize_vietnamese(query.lower())
        q_tokens = seg_query.split()

        scores = [0.0] * self.corpus_size
        candidate_indices = set()

        if self.bm25s_index is not None and seg_query:
            try:
                tokens = bm25s.tokenize(seg_query, stopwords=None, show_progress=False)
                bm25_res = self.bm25s_index.retrieve(tokens, k=min(max(top_k * 3, 200), self.corpus_size), show_progress=False)
                doc_indices = bm25_res.documents[0]
                bm25_scores = bm25_res.scores[0]
                for idx, sc in zip(doc_indices, bm25_scores):
                    if isinstance(idx, (int, np.integer)) and 0 <= idx < self.corpus_size:
                        scores[idx] = float(sc)
                        candidate_indices.add(int(idx))
            except Exception as e:
                pass

        if not candidate_indices:
            # Fallback pure python BM25
            for t in q_tokens:
                if t not in self.df:
                    continue
                df_val = self.df[t]
                idf = math.log((self.corpus_size - df_val + 0.5) / (df_val + 0.5) + 1.0)
                for i, c in enumerate(self.corpus):
                    text_tokens = c.get("text_norm", "").split()
                    tf = text_tokens.count(t)
                    if tf > 0:
                        doc_l = self.doc_len[i]
                        score = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * (doc_l / self.avg_doc_len)))
                        scores[i] += score
                        candidate_indices.add(i)

        # Apply Legal Entity Booster on candidate pool
        for i in (candidate_indices if candidate_indices else range(self.corpus_size)):
            raw = self.corpus[i].get("text_raw", "")
            # Boost exact document number matches
            for d in signals.get("doc_numbers", []):
                if d in raw:
                    scores[i] += 25.0
            # Boost exact article number matches
            for a in signals.get("articles", []):
                if f"Điều {a}." in raw or f"Điều {a} " in raw:
                    scores[i] += 12.0
            # Boost exact clause matches
            for cl in signals.get("clauses", []):
                if f"Khoản {cl}." in raw or f"\n{cl}. " in raw:
                    scores[i] += 6.0

        # Sort top K
        eval_indices = list(candidate_indices) if candidate_indices else range(self.corpus_size)
        ranked_indices = sorted(eval_indices, key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if scores[idx] <= 0:
                continue
            item = dict(self.corpus[idx])
            item["score"] = float(scores[idx])
            item["rank"] = rank
            results.append(item)
        return results

    def save(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        df_corpus = pd.DataFrame(self.corpus)
        df_corpus.to_parquet(os.path.join(index_dir, "corpus_meta.parquet"), index=False)
        if self.bm25s_index is not None:
            self.bm25s_index.save(os.path.join(index_dir, "bm25s_index"))

    @classmethod
    def load(cls, index_dir: str):
        parquet_path = os.path.join(index_dir, "corpus_meta.parquet")
        json_path = os.path.join(index_dir, "corpus_meta.json")
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            corpus = df.to_dict("records")
        elif os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
        else:
            corpus = []

        retriever = cls()
        bm25s_dir = os.path.join(index_dir, "bm25s_index")
        if bm25s is not None and os.path.exists(os.path.join(bm25s_dir, "params.index.json")):
            retriever.corpus = corpus
            retriever.doc_ids = [c["chunk_id"] for c in corpus]
            retriever.corpus_size = len(corpus)
            retriever.bm25s_index = bm25s.BM25.load(bm25s_dir, mmap=True)
        else:
            retriever.fit(corpus)
        return retriever
