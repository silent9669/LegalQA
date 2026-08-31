"""BM25 Sparse Retriever with Vietnamese word segmentation and statutory legal signal boosting."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.common.normalize import clean_legal_text, extract_legal_signals, tokenize_vietnamese

try:
    import bm25s
except ImportError:
    bm25s = None


class BM25Retriever:
    """Fast lexical retriever supporting bm25s C-bindings and inverted index Python fallback."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_ids: List[str] = []
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Counter = Counter()
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        self.corpus_size: int = 0
        self.bm25s_index: Any = None

    def fit(self, corpus: List[Dict[str, Any]]) -> None:
        """Fit BM25 index on a collection of chunk dictionaries."""
        # Reset mutable state
        self.corpus = corpus
        self.doc_ids = [c.get("chunk_id", str(i)) for i, c in enumerate(corpus)]
        self.corpus_size = len(corpus)
        self.doc_len = []
        self.df = Counter()
        self.inverted_index = defaultdict(list)
        self.bm25s_index = None

        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        tokenized_corpus = []
        for idx, c in enumerate(corpus):
            tokens = c.get("text_norm", "").split()
            if not tokens:
                tokens = tokenize_vietnamese(c.get("text_raw", "")).split()
            tokenized_corpus.append(tokens)
            doc_l = len(tokens)
            self.doc_len.append(doc_l)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.df[t] += 1
                self.inverted_index[t].append(idx)

        self.avg_doc_len = sum(self.doc_len) / max(1, self.corpus_size)

        if bm25s is not None and self.corpus_size > 0:
            try:
                self.bm25s_index = bm25s.BM25(k1=self.k1, b=self.b)
                self.bm25s_index.index(tokenized_corpus)
            except Exception:
                self.bm25s_index = None

    def search(self, query: str, top_k: int = 60) -> List[Dict[str, Any]]:
        """Search query across indexed corpus and return ranked results with legal entity boosts."""
        if not self.corpus or self.corpus_size == 0 or not query.strip():
            return []

        signals = extract_legal_signals(query)
        seg_query = tokenize_vietnamese(query.lower())
        q_tokens = seg_query.split()

        scores = [0.0] * self.corpus_size
        candidate_indices = set()

        if self.bm25s_index is not None and seg_query:
            try:
                tokens = bm25s.tokenize(seg_query, stopwords=None, show_progress=False)
                bm25_res = self.bm25s_index.retrieve(
                    tokens,
                    k=min(max(top_k * 4, 300), self.corpus_size),
                    show_progress=False,
                )
                doc_indices = bm25_res.documents[0]
                bm25_scores = bm25_res.scores[0]
                for idx, sc in zip(doc_indices, bm25_scores):
                    if isinstance(idx, (int, np.integer)) and 0 <= idx < self.corpus_size:
                        scores[idx] = float(sc)
                        candidate_indices.add(int(idx))
            except Exception:
                pass

        if not candidate_indices:
            # Fast inverted index fallback
            matched_scores: Dict[int, float] = defaultdict(float)
            for t in q_tokens:
                if t in self.df:
                    df_val = self.df[t]
                    idf = math.log((self.corpus_size - df_val + 0.5) / (df_val + 0.5) + 1.0)
                    for doc_idx in self.inverted_index[t]:
                        text_tokens = self.corpus[doc_idx].get("text_norm", "").split()
                        tf = text_tokens.count(t)
                        if tf > 0:
                            doc_l = self.doc_len[doc_idx]
                            denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_l / self.avg_doc_len))
                            score = idf * (tf * (self.k1 + 1.0)) / max(1e-6, denom)
                            matched_scores[doc_idx] += score

            for idx, sc in matched_scores.items():
                scores[idx] = sc
                candidate_indices.add(idx)

        # Apply Legal Entity Booster on candidate pool
        eval_indices = candidate_indices if candidate_indices else range(min(500, self.corpus_size))
        for i in eval_indices:
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

        ranked_indices = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)[:top_k]
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if scores[idx] <= 0:
                continue
            item = dict(self.corpus[idx])
            item["score"] = float(scores[idx])
            item["rank"] = rank
            results.append(item)

        return results

    def save(self, index_dir: str, save_corpus_meta: bool = False) -> None:
        """Save BM25 index and parameters without duplicating full corpus parquet by default."""
        os.makedirs(index_dir, exist_ok=True)
        manifest = {
            "corpus_size": self.corpus_size,
            "avg_doc_len": self.avg_doc_len,
            "k1": self.k1,
            "b": self.b,
            "has_bm25s": self.bm25s_index is not None,
        }
        with open(os.path.join(index_dir, "bm25_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if save_corpus_meta:
            df_corpus = pd.DataFrame(self.corpus)
            df_corpus.to_parquet(os.path.join(index_dir, "corpus_meta.parquet"), index=False)

        if self.bm25s_index is not None:
            self.bm25s_index.save(os.path.join(index_dir, "bm25s_index"))

    @classmethod
    def load(cls, index_dir: str, corpus_path: Optional[str] = None) -> BM25Retriever:
        """Load BM25 index referencing canonical corpus path."""
        corpus: List[Dict[str, Any]] = []

        if corpus_path and os.path.exists(corpus_path):
            df = pd.read_parquet(corpus_path)
            corpus = df.to_dict("records")
        else:
            parquet_path = os.path.join(index_dir, "corpus_meta.parquet")
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
                corpus = df.to_dict("records")

        retriever = cls()
        bm25s_dir = os.path.join(index_dir, "bm25s_index")
        if bm25s is not None and os.path.exists(os.path.join(bm25s_dir, "params.index.json")):
            retriever.corpus = corpus
            retriever.doc_ids = [c["chunk_id"] for c in corpus]
            retriever.corpus_size = len(corpus)
            try:
                retriever.bm25s_index = bm25s.BM25.load(bm25s_dir, mmap=True)
            except Exception:
                retriever.fit(corpus)
        else:
            retriever.fit(corpus)

        return retriever
