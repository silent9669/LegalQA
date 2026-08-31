"""Dense DEk21 v2 Retriever (CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.common.normalize import tokenize_vietnamese

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    torch = None


class DEk21Retriever:
    """DEk21 v2 Dense Retriever with batched embedding, persistent storage, and FP16 support."""

    def __init__(
        self,
        model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        revision: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.revision = revision
        if device is None:
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.corpus: List[Dict[str, Any]] = []
        self.corpus_embeddings: Optional[np.ndarray] = None

    def _lazy_init(self) -> None:
        if self.model is None and self.model_name != "mock" and SentenceTransformer is not None:
            kwargs = {"device": self.device}
            if self.revision:
                kwargs["revision"] = self.revision
            self.model = SentenceTransformer(self.model_name, **kwargs)
            if self.device == "cuda" and torch is not None and hasattr(self.model, "half"):
                self.model.half()

    def encode_texts(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Encode list of strings into normalized 768-dim numpy float32 embeddings."""
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        if self.model_name == "mock" or SentenceTransformer is None:
            np.random.seed(42)
            emb = np.random.randn(len(texts), 768).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / np.maximum(norms, 1e-12)

        self._lazy_init()
        segmented = [tokenize_vietnamese(t) for t in texts]
        embeddings = self.model.encode(
            segmented,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.array(embeddings, dtype=np.float32)

    def fit_mock(self, corpus: List[Dict[str, Any]]) -> None:
        """Fit with mock embeddings for fast local tests."""
        self.corpus = corpus
        raw_texts = [c.get("text_raw", "") for c in corpus]
        self.corpus_embeddings = self.encode_texts(raw_texts)

    def fit(self, corpus: List[Dict[str, Any]], batch_size: int = 64, show_progress: bool = True) -> None:
        """Encode entire corpus and store L2-normalized embeddings in memory."""
        self.corpus = corpus
        raw_texts = [c.get("text_raw", "") for c in corpus]
        self.corpus_embeddings = self.encode_texts(raw_texts, batch_size=batch_size, show_progress=show_progress)

    def search(self, query: str, top_k: int = 60) -> List[Dict[str, Any]]:
        """Dense similarity search via inner product on normalized embeddings."""
        if self.corpus_embeddings is None or len(self.corpus) == 0 or not query.strip():
            return []

        q_emb = self.encode_texts([query], show_progress=False)[0]
        sims = np.dot(self.corpus_embeddings, q_emb)

        top_indices = np.argsort(sims)[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            item = dict(self.corpus[idx])
            item["dense_score"] = float(sims[idx])
            item["rank"] = rank
            results.append(item)
        return results

    def save_index(self, index_dir: str) -> None:
        """Save precomputed corpus embeddings and metadata."""
        os.makedirs(index_dir, exist_ok=True)
        if self.corpus_embeddings is not None:
            emb_path = os.path.join(index_dir, "embeddings.npy")
            np.save(emb_path, self.corpus_embeddings)

        meta = {
            "model_name": self.model_name,
            "revision": self.revision,
            "corpus_size": len(self.corpus),
            "dim": self.corpus_embeddings.shape[1] if self.corpus_embeddings is not None else 768,
        }
        with open(os.path.join(index_dir, "dek21_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load_index(cls, index_dir: str, corpus_path: Optional[str] = None, model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", device: Optional[str] = None) -> DEk21Retriever:
        """Load precomputed embeddings from disk."""
        meta_path = os.path.join(index_dir, "dek21_manifest.json")
        revision = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                model_name = meta.get("model_name", model_name)
                revision = meta.get("revision")

        retriever = cls(model_name=model_name, revision=revision, device=device)

        emb_path = os.path.join(index_dir, "embeddings.npy")
        if os.path.exists(emb_path):
            retriever.corpus_embeddings = np.load(emb_path)

        if corpus_path and os.path.exists(corpus_path):
            df = pd.read_parquet(corpus_path)
            retriever.corpus = df.to_dict("records")

        return retriever
