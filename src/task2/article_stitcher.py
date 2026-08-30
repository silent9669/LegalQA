class ArticleStitcher:
    def __init__(self, all_chunks: list[dict]):
        self.article_to_chunks = {}
        for c in all_chunks:
            p_art = c.get("parent_article_id")
            if p_art:
                if p_art not in self.article_to_chunks:
                    self.article_to_chunks[p_art] = []
                self.article_to_chunks[p_art].append(c)

        for p_art in self.article_to_chunks:
            self.article_to_chunks[p_art].sort(key=lambda x: x.get("start_char", 0))

    def stitch(self, seed_chunks: list[dict], max_chars: int = 4000) -> dict:
        if not seed_chunks:
            return {"parent_article_id": "", "doc_name": "", "stitched_text": "", "focused_text": ""}

        top_seed = seed_chunks[0]
        p_art = top_seed.get("parent_article_id", "")
        doc_name = top_seed.get("doc_name", "")
        focused_text = top_seed.get("text_raw", "")

        siblings = self.article_to_chunks.get(p_art, [top_seed])

        pieces = []
        seen = set()
        for sib in siblings:
            cid = sib["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                pieces.append(sib.get("text_raw", ""))

        stitched_text = "\n\n".join(pieces)
        if len(stitched_text) > max_chars:
            stitched_text = stitched_text[:max_chars] + "\n..."

        return {
            "parent_article_id": p_art,
            "doc_name": doc_name,
            "stitched_text": stitched_text,
            "focused_text": focused_text
        }
