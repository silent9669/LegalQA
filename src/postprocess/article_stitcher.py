from collections import defaultdict
from src.data.canonical import normalize_vietnamese_text

class ArticleStitcher:
    """
    Assembles fragmented micro-chunks belonging to the same Article (doc_id, dieu)
    in sequential part order to reconstruct the complete statutory Article text.
    """
    def __init__(self, chunks_list: list[dict]):
        self.article_map = defaultdict(list)
        self.doc_title_map = {}

        for chunk in chunks_list:
            doc_id = str(chunk.get("doc_id") or chunk.get("context_id") or "")
            dieu = str(chunk.get("dieu") or "").strip()
            part = int(chunk.get("part") or 1)

            if doc_id and chunk.get("name"):
                self.doc_title_map[doc_id] = chunk["name"]

            if doc_id and dieu:
                key = f"{doc_id}::{dieu}"
                self.article_map[key].append((part, chunk))

        # Sort each article's chunks by part index
        for key in self.article_map:
            self.article_map[key].sort(key=lambda x: x[0])

    def get_full_article(self, doc_id: str, dieu: str) -> dict | None:
        key = f"{str(doc_id)}::{str(dieu).strip()}"
        if key not in self.article_map:
            return None

        parts = self.article_map[key]
        if not parts:
            return None

        first_chunk = parts[0][1]
        full_content = "\n".join([p[1].get("content", "").strip() for p in parts if p[1].get("content")])

        stitched = dict(first_chunk)
        stitched["content"] = full_content
        stitched["n_parts"] = len(parts)
        stitched["part"] = 1
        return stitched

    def expand_chunk(self, chunk: dict) -> dict:
        """
        If a retrieved chunk is part of a multi-part Article, returns the stitched full Article.
        Otherwise returns the chunk as-is.
        """
        doc_id = str(chunk.get("doc_id") or chunk.get("context_id") or "")
        dieu = str(chunk.get("dieu") or "").strip()

        if doc_id and dieu:
            full_art = self.get_full_article(doc_id, dieu)
            if full_art:
                return full_art
        return chunk
