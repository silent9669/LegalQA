"""BM25 Sparse Retriever with Vietnamese word segmentation and statutory legal signal boosting."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.normalize import clean_legal_text, extract_legal_signals, tokenize_vietnamese

try:
    import bm25s
except ImportError:
    bm25s = None


class BM25Retriever:
    """Fast lexical retriever supporting bm25s C-bindings and exact inverted index Python fallback."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_ids: List[str] = []
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Counter = Counter()
        # Inverted index stores (doc_idx, tf) for zero-truncation exact BM25
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.corpus_size: int = 0
        self.bm25s_index: Any = None

    def fit(self, corpus: List[Dict[str, Any]]) -> None:
        """Fit BM25 index on a collection of chunk dictionaries without any posting truncation."""
        self.corpus = corpus
        self.doc_ids = [str(c.get("chunk_id", i)) for i, c in enumerate(corpus)]
        self.corpus_size = len(corpus)
        self.doc_len = []
        self.df = Counter()
        self.postings = defaultdict(list)
        self.bm25s_index = None

        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        tokenized_corpus = []
        for idx, c in enumerate(corpus):
            raw_tokens = c.get("text_norm", "")
            if not raw_tokens:
                raw_tokens = tokenize_vietnamese(c.get("text_raw", ""))
            tokens = raw_tokens.split() if isinstance(raw_tokens, str) else list(raw_tokens)
            tokenized_corpus.append(tokens)

            doc_l = len(tokens)
            self.doc_len.append(doc_l)

            counts = Counter(tokens)
            for t, tf in counts.items():
                self.df[t] += 1
                self.postings[t].append((idx, tf))

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

        scores: Dict[int, float] = defaultdict(float)

        if self.bm25s_index is not None and q_tokens:
            try:
                bm25_res = self.bm25s_index.retrieve(
                    [q_tokens],
                    k=min(max(top_k * 4, 300), self.corpus_size),
                    show_progress=False,
                )
                doc_indices = bm25_res.documents[0]
                bm25_scores = bm25_res.scores[0]
                for idx, sc in zip(doc_indices, bm25_scores):
                    if isinstance(idx, (int, np.integer)) and 0 <= idx < self.corpus_size:
                        scores[int(idx)] = float(sc)
            except Exception:
                pass

        if not scores:
            # Exact Python Inverted Index BM25 fallback
            for t in q_tokens:
                if t in self.postings:
                    df_val = self.df[t]
                    # Standard Robertson-Spärck Jones IDF
                    idf = math.log((self.corpus_size - df_val + 0.5) / (df_val + 0.5) + 1.0)
                    if idf <= 0:
                        idf = 1e-4

                    for doc_idx, tf in self.postings[t]:
                        doc_l = self.doc_len[doc_idx]
                        denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_l / max(1e-6, self.avg_doc_len)))
                        score = idf * (tf * (self.k1 + 1.0)) / max(1e-6, denom)
                        scores[doc_idx] += score

        if not scores:
            return []

        # Candidate pool for entity boost (top candidates by lexical BM25)
        top_candidates = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:max(top_k * 4, 200)]

        # Apply Legal Entity Booster on candidate pool
        for i in top_candidates:
            raw = self.corpus[i].get("text_raw", "")
            raw_upper = raw.upper()
            # Boost exact document number matches
            for d in signals.get("doc_numbers", []):
                if d.upper() in raw_upper:
                    scores[i] += 25.0
            # Boost exact article number matches
            for a in signals.get("articles", []):
                if re.search(rf'\b[Đđ]iều\s+{re.escape(a)}\b', raw, re.IGNORECASE):
                    scores[i] += 12.0
            # Boost exact clause matches
            for cl in signals.get("clauses", []):
                if re.search(rf'\b(?:[Kk]hoản\s+{re.escape(cl)}|\n{re.escape(cl)}\.)\b', raw, re.IGNORECASE):
                    scores[i] += 6.0

        ranked_indices = sorted(top_candidates, key=lambda i: scores[i], reverse=True)[:top_k]
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if scores[idx] <= 0:
                continue
            item = dict(self.corpus[idx])
            item["score"] = float(scores[idx])
            item["bm25_score"] = float(scores[idx])
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
            "doc_ids": self.doc_ids,
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
                # Also fit postings for python fallback compatibility
                tokenized_corpus = []
                for idx, c in enumerate(corpus):
                    raw_tokens = c.get("text_norm", "")
                    if not raw_tokens:
                        raw_tokens = tokenize_vietnamese(c.get("text_raw", ""))
                    tokens = raw_tokens.split() if isinstance(raw_tokens, str) else list(raw_tokens)
                    tokenized_corpus.append(tokens)
                    doc_l = len(tokens)
                    retriever.doc_len.append(doc_l)
                    counts = Counter(tokens)
                    for t, tf in counts.items():
                        retriever.df[t] += 1
                        retriever.postings[t].append((idx, tf))
                retriever.avg_doc_len = sum(retriever.doc_len) / max(1, retriever.corpus_size)
            except Exception:
                retriever.fit(corpus)
        else:
            retriever.fit(corpus)

        return retriever
