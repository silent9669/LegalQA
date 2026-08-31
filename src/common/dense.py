"""Generic Dense Retriever with Multi-Model Support (DEk21, BGE-M3) and GPU Exact Inner-Product Top-K."""

from __future__ import annotations

import hashlib
import json
import os
import sys
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


def compute_chunk_ids_hash(doc_ids: List[str]) -> str:
    """Compute deterministic SHA256 hash across chunk IDs."""
    h = hashlib.sha256()
    for cid in doc_ids:
        h.update(str(cid).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class DenseRetriever:
    """Dense Retriever with exact GPU FP16 top-K search, row verification, and multi-model support."""

    def __init__(
        self,
        model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        revision: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "float16",
        final_mode: bool = False,
    ):
        self.model_name = model_name
        self.revision = revision
        self.dtype_str = dtype
        self.final_mode = final_mode
        if device is None:
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.corpus: List[Dict[str, Any]] = []
        self.doc_ids: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.gpu_tensor: Optional[Any] = None

    def _get_dim(self) -> int:
        if "bge-m3" in self.model_name.lower():
            return 1024
        return 768

    def _lazy_init(self) -> None:
        if self.model is None and self.model_name != "mock" and SentenceTransformer is not None:
            kwargs = {"device": self.device}
            if self.revision:
                kwargs["revision"] = self.revision
            self.model = SentenceTransformer(self.model_name, **kwargs)
            if self.device.startswith("cuda") and torch is not None and hasattr(self.model, "half"):
                self.model.half()

    def encode_texts(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Encode list of strings into L2-normalized numpy embeddings."""
        dim = self._get_dim()
        if not texts:
            return np.empty((0, dim), dtype=np.float32)

        if self.model_name == "mock" or SentenceTransformer is None:
            np.random.seed(42)
            emb = np.random.randn(len(texts), dim).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / np.maximum(norms, 1e-12)

        self._lazy_init()
        # DEk21 uses Vietnamese word tokenization; BGE-M3 handles raw text
        if "dek21" in self.model_name.lower():
            processed_texts = [tokenize_vietnamese(t) for t in texts]
        else:
            processed_texts = texts

        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.array(embeddings, dtype=np.float32)

    def fit_mock(self, corpus: List[Dict[str, Any]]) -> None:
        """Fit with mock normalized embeddings for fast testing."""
        self.corpus = corpus
        self.doc_ids = [str(c.get("chunk_id", i)) for i, c in enumerate(corpus)]
        raw_texts = [c.get("text_raw", "") for c in corpus]
        self.corpus_embeddings = self.encode_texts(raw_texts)
        self._sync_gpu_tensor()

    def fit(self, corpus: List[Dict[str, Any]], batch_size: int = 64, show_progress: bool = True) -> None:
        """Encode entire corpus and store L2-normalized embeddings."""
        self.corpus = corpus
        self.doc_ids = [str(c.get("chunk_id", i)) for i, c in enumerate(corpus)]
        raw_texts = [c.get("text_raw", "") for c in corpus]
        self.corpus_embeddings = self.encode_texts(raw_texts, batch_size=batch_size, show_progress=show_progress)
        self._sync_gpu_tensor()

    def _sync_gpu_tensor(self) -> None:
        """Move corpus tensor to GPU in FP16 for exact hardware top-K search."""
        if torch is None or self.corpus_embeddings is None:
            return

        if self.device.startswith("cuda") and torch.cuda.is_available():
            target_dtype = torch.float16 if self.dtype_str == "float16" else torch.float32
            try:
                t = torch.as_tensor(self.corpus_embeddings, dtype=target_dtype, device=self.device)
                self.gpu_tensor = t
            except Exception as e:
                if self.final_mode:
                    raise RuntimeError(f"FINAL_PIPELINE_ERROR: Failed to allocate GPU corpus tensor on {self.device}: {e}")
                print(f"Warning: Could not allocate GPU corpus tensor ({e}), using CPU inner product.", file=sys.stderr)
                self.gpu_tensor = None

    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """Exact dense inner-product search with GPU acceleration when available."""
        if self.corpus_embeddings is None or len(self.corpus) == 0 or not query.strip():
            return []

        top_k = min(top_k, len(self.corpus))

        # 1. GPU Exact Matrix Search
        if self.gpu_tensor is not None and torch is not None:
            try:
                q_emb = self.encode_texts([query], show_progress=False)
                target_dtype = self.gpu_tensor.dtype
                q_tensor = torch.as_tensor(q_emb, dtype=target_dtype, device=self.device)

                with torch.inference_mode():
                    sims = torch.matmul(q_tensor, self.gpu_tensor.T).squeeze(0)
                    scores, top_indices = torch.topk(sims, k=top_k)

                scores_np = scores.cpu().float().numpy()
                indices_np = top_indices.cpu().numpy()

                results: List[Dict[str, Any]] = []
                for rank, (idx, sc) in enumerate(zip(indices_np, scores_np), start=1):
                    item = dict(self.corpus[int(idx)])
                    item["score"] = float(sc)
                    item["dense_score"] = float(sc)
                    item["rank"] = rank
                    results.append(item)
                return results
            except Exception as e:
                if self.final_mode:
                    raise RuntimeError(f"FINAL_PIPELINE_ERROR: GPU dense search failed on {self.device}: {e}") from e
                print(f"GPU search fallback to CPU: {e}", file=sys.stderr)

        if self.final_mode and self.device.startswith("cuda"):
            raise RuntimeError("FINAL_PIPELINE_ERROR: GPU tensor not initialized in final mode.")

        # 2. CPU / NumPy Exact Search
        q_emb = self.encode_texts([query], show_progress=False)[0].astype(np.float32)
        emb_matrix = np.asarray(self.corpus_embeddings, dtype=np.float32)
        sims = np.dot(emb_matrix, q_emb)

        if len(sims) > top_k * 4:
            part_idx = np.argpartition(sims, -top_k)[-top_k:]
            sorted_part = part_idx[np.argsort(-sims[part_idx])]
            top_indices = sorted_part
        else:
            top_indices = np.argsort(sims)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            item = dict(self.corpus[int(idx)])
            item["score"] = float(sims[idx])
            item["dense_score"] = float(sims[idx])
            item["rank"] = rank
            results.append(item)
        return results

    def search_batch(self, queries: List[str], top_k: int = 50, batch_size: int = 64) -> List[List[Dict[str, Any]]]:
        """Batched exact dense retrieval for high inference throughput."""
        if self.corpus_embeddings is None or len(self.corpus) == 0 or not queries:
            return [[] for _ in queries]

        top_k = min(top_k, len(self.corpus))

        # 1. GPU Batched Top-K
        if self.gpu_tensor is not None and torch is not None:
            try:
                all_results: List[List[Dict[str, Any]]] = []
                for b_start in range(0, len(queries), batch_size):
                    b_queries = queries[b_start:b_start + batch_size]
                    q_embs = self.encode_texts(b_queries, batch_size=batch_size, show_progress=False)
                    q_tensor = torch.as_tensor(q_embs, dtype=self.gpu_tensor.dtype, device=self.device)

                    with torch.inference_mode():
                        sim_matrix = torch.matmul(q_tensor, self.gpu_tensor.T)
                        scores_batch, indices_batch = torch.topk(sim_matrix, k=top_k, dim=1)

                    scores_np = scores_batch.cpu().float().numpy()
                    indices_np = indices_batch.cpu().numpy()

                    for row_indices, row_scores in zip(indices_np, scores_np):
                        row_res = []
                        for rank, (idx, sc) in enumerate(zip(row_indices, row_scores), start=1):
                            item = dict(self.corpus[int(idx)])
                            item["score"] = float(sc)
                            item["dense_score"] = float(sc)
                            item["rank"] = rank
                            row_res.append(item)
                        all_results.append(row_res)
                return all_results
            except Exception as e:
                if self.final_mode:
                    raise RuntimeError(f"FINAL_PIPELINE_ERROR: GPU batched dense search failed: {e}") from e
                print(f"GPU batch search fallback to CPU: {e}", file=sys.stderr)

        # 2. CPU Batched Fallback
        all_results = []
        for q in queries:
            all_results.append(self.search(q, top_k=top_k))
        return all_results

    def save_index(self, index_dir: str, dtype: str = "float16") -> None:
        """Save precomputed corpus embeddings in FP16/FP32 with complete hash and provenance manifest."""
        os.makedirs(index_dir, exist_ok=True)
        emb_sha = ""
        if self.corpus_embeddings is not None:
            target_np_dtype = np.float16 if dtype == "float16" else np.float32
            emb_to_save = np.asarray(self.corpus_embeddings, dtype=target_np_dtype)
            emb_path = os.path.join(index_dir, "embeddings.npy")
            np.save(emb_path, emb_to_save)
            with open(emb_path, "rb") as f:
                emb_sha = hashlib.sha256(f.read()).hexdigest()

        doc_ids_sha = compute_chunk_ids_hash(self.doc_ids)

        meta = {
            "model_id": self.model_name,
            "model_name": self.model_name,
            "revision": self.revision,
            "dim": self._get_dim(),
            "dtype": dtype,
            "normalized": True,
            "corpus_rows": len(self.corpus),
            "doc_ids": self.doc_ids,
            "chunk_ids_sha256": doc_ids_sha,
            "embeddings_sha256": emb_sha,
        }

        with open(os.path.join(index_dir, "dense_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(index_dir, "dek21_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load_index(
        cls,
        index_dir: str,
        corpus_path: Optional[str] = None,
        model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        device: Optional[str] = None,
        dtype: str = "float16",
        final_mode: bool = False,
        expected_model_name: Optional[str] = None,
        expected_dtype: Optional[str] = None,
        verify_embeddings_hash: bool = False,
    ) -> DenseRetriever:
        """Load precomputed embeddings from disk using mmap and verify row alignment and hash integrity."""
        meta_path = os.path.join(index_dir, "dense_manifest.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(index_dir, "dek21_manifest.json")

        revision = None
        saved_doc_ids = []
        saved_chunk_sha = ""
        saved_emb_sha = ""
        expected_rows = None
        expected_dim = None

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                manifest_model = meta.get("model_id") or meta.get("model_name", model_name)
                revision = meta.get("revision")
                saved_doc_ids = meta.get("doc_ids", [])
                saved_chunk_sha = meta.get("chunk_ids_sha256", "")
                saved_emb_sha = meta.get("embeddings_sha256", "")
                dtype = meta.get("dtype", dtype)
                expected_rows = meta.get("corpus_rows")
                expected_dim = meta.get("dim")

                # Verify expected model name
                if expected_model_name and manifest_model != expected_model_name:
                    raise ValueError(
                        f"FINAL_PIPELINE_ERROR: Dense model mismatch! Expected '{expected_model_name}', but index has '{manifest_model}'"
                    )
                model_name = manifest_model

        retriever = cls(model_name=model_name, revision=revision, device=device, dtype=dtype, final_mode=final_mode)

        emb_path = os.path.join(index_dir, "embeddings.npy")
        if not os.path.exists(emb_path):
            if final_mode:
                raise FileNotFoundError(f"FINAL_PIPELINE_ERROR: Dense embeddings missing at {emb_path}")
        else:
            # Load with mmap preserving FP16 dtype without expanding to FP32 in RAM
            retriever.corpus_embeddings = np.load(emb_path, mmap_mode="r")
            if expected_rows is not None and retriever.corpus_embeddings.shape[0] != expected_rows:
                raise ValueError(
                    f"FINAL_PIPELINE_ERROR: Embedding rows ({retriever.corpus_embeddings.shape[0]}) != manifest rows ({expected_rows})"
                )
            if expected_dim is not None and retriever.corpus_embeddings.shape[1] != expected_dim:
                raise ValueError(
                    f"FINAL_PIPELINE_ERROR: Embedding dim ({retriever.corpus_embeddings.shape[1]}) != manifest dim ({expected_dim})"
                )
            if expected_dtype is not None and str(retriever.corpus_embeddings.dtype) != expected_dtype:
                raise ValueError(
                    f"FINAL_PIPELINE_ERROR: Embedding dtype ({retriever.corpus_embeddings.dtype}) != expected dtype ({expected_dtype})"
                )
            if verify_embeddings_hash and saved_emb_sha:
                with open(emb_path, "rb") as f:
                    curr_emb_sha = hashlib.sha256(f.read()).hexdigest()
                if curr_emb_sha != saved_emb_sha:
                    raise ValueError("FINAL_PIPELINE_ERROR: Dense embeddings.npy SHA256 checksum mismatch against manifest!")

        if corpus_path and os.path.exists(corpus_path):
            df = pd.read_parquet(corpus_path)
            retriever.corpus = df.to_dict("records")
            retriever.doc_ids = [str(c.get("chunk_id", i)) for i, c in enumerate(retriever.corpus)]

            if retriever.corpus_embeddings is not None:
                if len(retriever.corpus) != len(retriever.corpus_embeddings):
                    raise ValueError(
                        f"FINAL_PIPELINE_ERROR: Dense index row count mismatch! Corpus has {len(retriever.corpus)} rows, "
                        f"but embeddings.npy has {len(retriever.corpus_embeddings)} rows."
                    )
                if saved_chunk_sha:
                    curr_sha = compute_chunk_ids_hash(retriever.doc_ids)
                    if curr_sha != saved_chunk_sha:
                        if final_mode:
                            raise ValueError("FINAL_PIPELINE_ERROR: Dense chunk_ids_sha256 mismatch against manifest.")
                        print("Warning: chunk_id alignment differs from manifest. Verify corpus integrity.", file=sys.stderr)

        retriever._sync_gpu_tensor()
        return retriever
