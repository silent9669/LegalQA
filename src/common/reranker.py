import os

try:
    import torch
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None
    torch = None

class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
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

    def rerank(self, query: str, candidates: list[dict], top_k: int = 8) -> list[dict]:
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
        # Clean candidates text and pass pairs
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
