import os
import sys
from typing import Any, Dict, List, Optional

try:
    import torch
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None
    torch = None


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
        self.model_name = model_name
        if device is None and torch is not None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device or "cpu"
        self.model = None

    def _lazy_init(self):
        if self.model is None and self.model_name != "mock" and CrossEncoder is not None:
            print(f"Loading Cross-Encoder Reranker {self.model_name} on {self.device}...")
            self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        """Rerank candidates for a single query."""
        if not candidates:
            return []

        if self.model_name == "mock" or CrossEncoder is None:
            q_words = set(query.lower().split())
            scored = []
            for c in candidates:
                txt = c.get("text_raw", "").lower()
                overlap = sum(1 for w in q_words if w in txt)
                item = dict(c)
                item["rerank_score"] = float(overlap)
                scored.append(item)
            scored = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)
            for rank, item in enumerate(scored[:top_k], start=1):
                item["rank"] = rank
            return scored[:top_k]

        self._lazy_init()
        pairs = [[query, c.get("text_raw", "")[:1800]] for c in candidates]
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        scored = []
        for item, sc in zip(candidates, scores):
            entry = dict(item)
            entry["rerank_score"] = float(sc)
            scored.append(entry)

        scored = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)
        for rank, item in enumerate(scored[:top_k], start=1):
            item["rank"] = rank
        return scored[:top_k]

    def rerank_batch(
        self,
        queries: List[str],
        candidate_lists: List[List[Dict[str, Any]]],
        top_k: int = 8,
        batch_size: int = 32,
        pair_batch_size: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Batched cross-encoder reranking across multiple queries."""
        if not queries or not candidate_lists:
            return []

        effective_batch_size = pair_batch_size or batch_size

        if self.model_name == "mock" or CrossEncoder is None:
            return [self.rerank(q, cands, top_k=top_k) for q, cands in zip(queries, candidate_lists)]

        self._lazy_init()

        pairs = []
        mapping = []  # (query_idx, cand_idx)

        for q_idx, (query, candidates) in enumerate(zip(queries, candidate_lists)):
            for c_idx, cand in enumerate(candidates):
                txt = cand.get("text_raw", "")[:1800]
                pairs.append([query, txt])
                mapping.append((q_idx, c_idx))

        if not pairs:
            return [[] for _ in queries]

        scores = self.model.predict(pairs, batch_size=effective_batch_size, show_progress_bar=False)

        scored_by_query: List[List[Dict[str, Any]]] = [[] for _ in queries]
        for (q_idx, c_idx), score_val in zip(mapping, scores):
            entry = dict(candidate_lists[q_idx][c_idx])
            entry["rerank_score"] = float(score_val)
            scored_by_query[q_idx].append(entry)

        results: List[List[Dict[str, Any]]] = []
        for q_idx, c_list in enumerate(scored_by_query):
            sorted_cands = sorted(c_list, key=lambda x: x["rerank_score"], reverse=True)
            for rank, item in enumerate(sorted_cands[:top_k], start=1):
                item["rank"] = rank
            results.append(sorted_cands[:top_k])

        return results
