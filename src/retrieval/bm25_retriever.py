import math
from collections import Counter, defaultdict
from src.data.canonical import normalize_vietnamese_text
from src.retrieval.query_analyzer import analyze_query

class SimpleBM25:
    """
    High-performance Inverted Index BM25 lexical retriever with Legal Entity Boosting.
    Zero learned parameter footprint (< 4B compliant).
    """
    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_ids = [doc["id"] for doc in corpus]
        self.num_docs = len(corpus)
        self.doc_len = []

        # Fast document metadata lookup
        self.doc_num_map = {}
        self.art_num_map = {}
        self.doc_name_map = {}

        # Inverted index: token -> list of (doc_idx, count)
        self.inverted_index = defaultdict(list)
        self.df = Counter()

        total_len = 0
        for idx, doc in enumerate(corpus):
            doc_id = doc.get("id") or str(idx)
            self.doc_num_map[doc_id] = str(doc.get("document_number") or "").strip().lower()
            self.art_num_map[doc_id] = str(doc.get("article_number") or "").strip()
            self.doc_name_map[doc_id] = str(doc.get("name") or doc.get("document_title") or "").strip().lower()

            tokens = normalize_vietnamese_text(doc.get("text") or doc.get("content", "")).lower().split()
            d_len = len(tokens)
            self.doc_len.append(d_len)
            total_len += d_len

            counts = Counter(tokens)
            for token, cnt in counts.items():
                self.inverted_index[token].append((idx, cnt))
                self.df[token] += 1

        self.avgdl = total_len / max(1, self.num_docs)
        self.idf = {}
        for token, freq in self.df.items():
            self.idf[token] = math.log((self.num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        q_tokens = normalize_vietnamese_text(query).lower().split()
        if not q_tokens or not self.num_docs:
            return []

        # Analyze query for legal entities
        q_analysis = analyze_query(query)
        q_doc_num = q_analysis.get("doc_number", "").lower() if q_analysis.get("doc_number") else ""
        q_art_num = q_analysis.get("article_number", "")
        q_code_keywords = q_analysis.get("code_keywords", [])

        doc_scores = defaultdict(float)
        for token in set(q_tokens):
            if token not in self.inverted_index:
                continue
            idf = self.idf.get(token, 0.0)
            if idf <= 0:
                continue
            postings = self.inverted_index[token]
            for doc_idx, freq in postings:
                d_len = self.doc_len[doc_idx]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (d_len / self.avgdl))
                doc_scores[doc_idx] += idf * (numerator / denominator)

        if not doc_scores:
            return []

        # Apply Legal Entity Boosts
        for doc_idx in doc_scores:
            cid = self.corpus_ids[doc_idx]
            c_doc_num = self.doc_num_map.get(cid, "")
            c_art_num = self.art_num_map.get(cid, "")
            c_name = self.doc_name_map.get(cid, "")

            # 1. Exact document number boost (+35.0)
            if q_doc_num and c_doc_num and q_doc_num == c_doc_num:
                doc_scores[doc_idx] += 35.0
            elif q_doc_num and q_doc_num in c_name:
                doc_scores[doc_idx] += 25.0

            # 2. Exact article number boost (+15.0)
            if q_art_num and c_art_num and q_art_num == c_art_num:
                doc_scores[doc_idx] += 15.0

            # 3. Named code boost (+20.0)
            if q_code_keywords:
                for kw in q_code_keywords:
                    if kw.lower() in c_name:
                        doc_scores[doc_idx] += 10.0
                        break

        top_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.corpus_ids[doc_idx], score) for doc_idx, score in top_items]
