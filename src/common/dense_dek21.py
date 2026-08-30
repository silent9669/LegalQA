import numpy as np
from src.common.normalize import tokenize_vietnamese

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class DEk21Retriever:
    def __init__(self, model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.corpus = []
        self.corpus_embeddings = None

    def _lazy_init(self):
        if self.model is None and self.model_name != "mock" and SentenceTransformer is not None:
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.model_name == "mock" or SentenceTransformer is None:
            # Deterministic hash-based mock embedding for tests
            np.random.seed(42)
            emb = np.random.randn(len(texts), 768).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / np.maximum(norms, 1e-12)

        self._lazy_init()
        segmented = [tokenize_vietnamese(t) for t in texts]
        embeddings = self.model.encode(segmented, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    def fit_mock(self, corpus: list[dict]):
        self.corpus = corpus
        self.corpus_embeddings = self.encode_texts([c.get("text_raw", "") for c in corpus])

    def fit(self, corpus: list[dict], batch_size: int = 64):
        self.corpus = corpus
        raw_texts = [c.get("text_raw", "") for c in corpus]
        self.corpus_embeddings = self.encode_texts(raw_texts)

    def search(self, query: str, top_k: int = 60) -> list[dict]:
        if self.corpus_embeddings is None or len(self.corpus) == 0:
            return []

        q_emb = self.encode_texts([query])[0]
        sims = np.dot(self.corpus_embeddings, q_emb)

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            item = dict(self.corpus[idx])
            item["score"] = float(sims[idx])
            item["rank"] = rank
            results.append(item)
        return results
