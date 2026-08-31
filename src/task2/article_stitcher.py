"""Structured Article Stitcher and Evidence Packer for LegalQA Task 2."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from src.common.normalize import clean_legal_text, prettify_doc_title


class ArticleStitcher:
    """Stitches structured statutory units (documents, articles, clauses) from top retrieval seeds."""

    def __init__(self, all_chunks: List[Dict[str, Any]]):
        self.article_to_chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.doc_to_articles: Dict[str, Set[str]] = {}

        for c in all_chunks:
            p_art = str(c.get("parent_article_id") or "").strip()
            doc_name = str(c.get("doc_name") or c.get("doc_id") or "").strip()

            if p_art:
                if p_art not in self.article_to_chunks:
                    self.article_to_chunks[p_art] = []
                self.article_to_chunks[p_art].append(c)

                if doc_name:
                    if doc_name not in self.doc_to_articles:
                        self.doc_to_articles[doc_name] = set()
                    self.doc_to_articles[doc_name].add(p_art)

        # Sort chunks within each parent article by their start_char / clause number
        for p_art, chunks in self.article_to_chunks.items():
            chunks.sort(key=lambda x: (
                int(x.get("start_char") or 0),
                int(re.findall(r'\d+', str(x.get("clause_number", "0")))[0]) if re.findall(r'\d+', str(x.get("clause_number", "0"))) else 0,
            ))

    def stitch(self, seed_chunks: List[Dict[str, Any]], max_chars: int = 3500) -> Dict[str, Any]:
        """Pack structured legal evidence from ranked seeds, maintaining document order and headers."""
        if not seed_chunks:
            return {
                "parent_article_id": "",
                "doc_name": "",
                "stitched_text": "",
                "focused_text": "",
                "articles_included": [],
                "doc_names": [],
            }

        top_seed = seed_chunks[0]
        primary_art = str(top_seed.get("parent_article_id", "")).strip()
        primary_doc = str(top_seed.get("doc_name", "")).strip()
        focused_text = str(top_seed.get("text_raw", "")).strip()

        collected_pieces: List[str] = []
        seen_chunk_ids: Set[str] = set()
        seen_articles: List[str] = []
        seen_docs: Set[str] = set()
        current_len = 0

        # 1. Primary Article Stitching: include all sibling clauses of the top seed in order
        if primary_art and primary_art in self.article_to_chunks:
            seen_articles.append(primary_art)
            if primary_doc:
                seen_docs.add(primary_doc)

            for sib in self.article_to_chunks[primary_art]:
                cid = str(sib.get("chunk_id", ""))
                txt = str(sib.get("text_raw", "")).strip()
                if cid and cid not in seen_chunk_ids and txt:
                    seen_chunk_ids.add(cid)
                    piece_len = len(txt) + 2
                    if current_len + piece_len <= max_chars:
                        collected_pieces.append(txt)
                        current_len += piece_len
                    else:
                        remaining = max_chars - current_len
                        if remaining > 100:
                            collected_pieces.append(txt[:remaining].rstrip() + "...")
                            current_len = max_chars
                        break
        else:
            # Fallback if primary article not in index map
            if focused_text:
                collected_pieces.append(focused_text)
                seen_chunk_ids.add(str(top_seed.get("chunk_id", "")))
                current_len += len(focused_text)

        # 2. Secondary Articles: include remaining high-confidence seeds if budget permits
        for seed in seed_chunks[1:]:
            if current_len >= max_chars:
                break

            s_art = str(seed.get("parent_article_id", "")).strip()
            s_doc = str(seed.get("doc_name", "")).strip()
            s_cid = str(seed.get("chunk_id", ""))
            s_raw = str(seed.get("text_raw", "")).strip()

            if s_art and s_art in seen_articles:
                continue

            if s_art and s_art in self.article_to_chunks:
                seen_articles.append(s_art)
                if s_doc:
                    seen_docs.add(s_doc)

                for sib in self.article_to_chunks[s_art]:
                    cid = str(sib.get("chunk_id", ""))
                    txt = str(sib.get("text_raw", "")).strip()
                    if cid and cid not in seen_chunk_ids and txt:
                        seen_chunk_ids.add(cid)
                        piece_len = len(txt) + 2
                        if current_len + piece_len <= max_chars:
                            collected_pieces.append(txt)
                            current_len += piece_len
                        else:
                            remaining = max_chars - current_len
                            if remaining > 100:
                                collected_pieces.append(txt[:remaining].rstrip() + "...")
                                current_len = max_chars
                            break
            elif s_raw and s_cid not in seen_chunk_ids:
                seen_chunk_ids.add(s_cid)
                piece_len = len(s_raw) + 2
                if current_len + piece_len <= max_chars:
                    collected_pieces.append(s_raw)
                    current_len += piece_len
                else:
                    remaining = max_chars - current_len
                    if remaining > 100:
                        collected_pieces.append(s_raw[:remaining].rstrip() + "...")
                        current_len = max_chars
                    break

        stitched_text = "\n\n".join(collected_pieces)

        return {
            "parent_article_id": primary_art,
            "doc_name": primary_doc,
            "stitched_text": stitched_text,
            "focused_text": focused_text,
            "articles_included": seen_articles,
            "doc_names": list(seen_docs),
            "total_chars": len(stitched_text),
        }
