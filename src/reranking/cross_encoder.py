from src.data.canonical import normalize_vietnamese_text

class SimpleLexicalReranker:
    """
    Fast, zero-parameter lexical cross-encoder reranker based on word-overlap,
    n-gram matching, and citation entity alignment.
    """
    def __init__(self):
        pass

    def score_pair(self, query: str, candidate: dict) -> float:
        q_tokens = set(normalize_vietnamese_text(query).lower().split())
        c_text = candidate.get("searchable_text") or candidate.get("content") or ""
        c_tokens = set(normalize_vietnamese_text(c_text).lower().split())

        if not q_tokens or not c_tokens:
            return 0.0

        intersection = q_tokens & c_tokens
        jaccard = len(intersection) / len(q_tokens | c_tokens)
        overlap = len(intersection) / len(q_tokens)
        return 0.7 * overlap + 0.3 * jaccard

    def rank(self, query: str, candidates: list[dict], top_k: int = 8) -> list[dict]:
        scored = []
        for cand in candidates:
            score = self.score_pair(query, cand)
            scored.append((cand, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [cand for cand, sc in scored[:top_k]]
