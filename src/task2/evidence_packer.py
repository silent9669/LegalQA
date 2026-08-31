"""Structured Statutory Evidence Packer supporting multiple candidate evidence granularities."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from src.common.normalize import clean_legal_text, prettify_doc_title


class EvidencePacker:
    """Packs structured statutory units (documents, articles, clauses) into multi-granularity candidate evidence."""

    def __init__(self, all_chunks: List[Dict[str, Any]]):
        self.article_to_chunks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.doc_to_articles: Dict[str, Set[str]] = defaultdict(set)
        self.chunk_by_id: Dict[str, Dict[str, Any]] = {}

        for c in all_chunks:
            cid = str(c.get("chunk_id", "")).strip()
            p_art = str(c.get("parent_article_id") or "").strip()
            doc_name = str(c.get("doc_name") or c.get("doc_id") or "").strip()

            if cid:
                self.chunk_by_id[cid] = c

            if p_art:
                self.article_to_chunks[p_art].append(c)
                if doc_name:
                    self.doc_to_articles[doc_name].add(p_art)

        # Sort chunks within each parent article by their start_char and clause number
        for p_art, chunks in self.article_to_chunks.items():
            chunks.sort(key=lambda x: (
                int(x.get("start_char") or 0),
                int(re.findall(r'\d+', str(x.get("clause_number", "0")))[0]) if re.findall(r'\d+', str(x.get("clause_number", "0"))) else 0,
            ))

    def pack_evidence(
        self,
        seed_chunks: List[Dict[str, Any]],
        pack_type: str = "multi_seed_2500_chars",
        max_chars: int = 3500,
    ) -> Dict[str, Any]:
        """Pack structured legal evidence according to the requested granularity."""
        if not seed_chunks:
            return {
                "text": "",
                "stitched_text": "",
                "focused_text": "",
                "pack_type": pack_type,
                "parent_article_id": "",
                "doc_name": "",
                "doc_names": [],
                "articles_included": [],
                "article_ids": [],
                "clause_ids": [],
                "chars": 0,
                "total_chars": 0,
                "top_doc_name": "",
                "top_article_num": "",
                "top_clause_num": "",
            }

        top_seed = seed_chunks[0]
        top_doc = str(top_seed.get("doc_name") or top_seed.get("doc_id") or "").strip()
        top_art_id = str(top_seed.get("parent_article_id", "")).strip()
        top_art_num = str(top_seed.get("article_number", "")).strip()
        top_clause_num = str(top_seed.get("clause_number", "")).strip()
        focused_text = str(top_seed.get("text_raw", "")).strip()

        # 1. Focused Clause
        if pack_type == "focused_clause":
            return {
                "text": focused_text,
                "stitched_text": focused_text,
                "focused_text": focused_text,
                "pack_type": "focused_clause",
                "parent_article_id": top_art_id,
                "doc_name": top_doc,
                "doc_names": [top_doc] if top_doc else [],
                "articles_included": [top_art_id] if top_art_id else [],
                "article_ids": [top_art_id] if top_art_id else [],
                "clause_ids": [str(top_seed.get("chunk_id", ""))],
                "chars": len(focused_text),
                "total_chars": len(focused_text),
                "top_doc_name": top_doc,
                "top_article_num": top_art_num,
                "top_clause_num": top_clause_num,
            }

        # 2. Primary Full Article / Relevant Siblings
        if pack_type in ("primary_full_article", "primary_article_relevant_siblings"):
            pieces: List[str] = []
            seen_cids: Set[str] = set()
            included_clauses: List[str] = []

            if top_art_id and top_art_id in self.article_to_chunks:
                for sib in self.article_to_chunks[top_art_id]:
                    cid = str(sib.get("chunk_id", ""))
                    txt = str(sib.get("text_raw", "")).strip()
                    if cid and cid not in seen_cids and txt:
                        seen_cids.add(cid)
                        included_clauses.append(cid)
                        pieces.append(txt)
            else:
                pieces.append(focused_text)
                included_clauses.append(str(top_seed.get("chunk_id", "")))

            packed_text = "\n\n".join(pieces)
            return {
                "text": packed_text,
                "stitched_text": packed_text,
                "focused_text": focused_text,
                "pack_type": pack_type,
                "parent_article_id": top_art_id,
                "doc_name": top_doc,
                "doc_names": [top_doc] if top_doc else [],
                "articles_included": [top_art_id] if top_art_id else [],
                "article_ids": [top_art_id] if top_art_id else [],
                "clause_ids": included_clauses,
                "chars": len(packed_text),
                "total_chars": len(packed_text),
                "top_doc_name": top_doc,
                "top_article_num": top_art_num,
                "top_clause_num": top_clause_num,
            }

        # 3. Multi-Seed / Relevance Selected Packing
        budget = 2500 if pack_type == "multi_seed_2500_chars" else (4000 if pack_type == "multi_seed_4000_chars" else max_chars)
        max_articles = 2 if pack_type == "relevance_selected_top2_articles" else 5

        collected_pieces: List[str] = []
        seen_chunk_ids: Set[str] = set()
        seen_articles: List[str] = []
        seen_docs: Set[str] = set()
        current_len = 0

        for seed in seed_chunks:
            if len(seen_articles) >= max_articles and current_len >= budget:
                break

            s_art = str(seed.get("parent_article_id", "")).strip()
            s_doc = str(seed.get("doc_name") or seed.get("doc_id") or "").strip()
            s_cid = str(seed.get("chunk_id", "")).strip()
            s_raw = str(seed.get("text_raw", "")).strip()

            if s_art and s_art not in seen_articles and len(seen_articles) < max_articles:
                seen_articles.append(s_art)
                if s_doc:
                    seen_docs.add(s_doc)

                # Pack siblings of this article in source order
                if s_art in self.article_to_chunks:
                    for sib in self.article_to_chunks[s_art]:
                        cid = str(sib.get("chunk_id", ""))
                        txt = str(sib.get("text_raw", "")).strip()
                        if cid and cid not in seen_chunk_ids and txt:
                            seen_chunk_ids.add(cid)
                            piece_len = len(txt) + 2
                            if current_len + piece_len <= budget:
                                collected_pieces.append(txt)
                                current_len += piece_len
                            else:
                                remaining = budget - current_len
                                if remaining > 100:
                                    collected_pieces.append(txt[:remaining].rstrip() + "...")
                                    current_len = budget
                                break
            elif s_raw and s_cid not in seen_chunk_ids:
                seen_chunk_ids.add(s_cid)
                piece_len = len(s_raw) + 2
                if current_len + piece_len <= budget:
                    collected_pieces.append(s_raw)
                    current_len += piece_len
                else:
                    remaining = budget - current_len
                    if remaining > 100:
                        collected_pieces.append(s_raw[:remaining].rstrip() + "...")
                        current_len = budget
                    break

        stitched_text = "\n\n".join(collected_pieces) if collected_pieces else focused_text

        return {
            "text": stitched_text,
            "stitched_text": stitched_text,
            "focused_text": focused_text,
            "pack_type": pack_type,
            "parent_article_id": top_art_id,
            "doc_name": top_doc,
            "doc_names": list(seen_docs) if seen_docs else ([top_doc] if top_doc else []),
            "articles_included": seen_articles if seen_articles else ([top_art_id] if top_art_id else []),
            "article_ids": seen_articles if seen_articles else ([top_art_id] if top_art_id else []),
            "clause_ids": list(seen_chunk_ids),
            "chars": len(stitched_text),
            "total_chars": len(stitched_text),
            "top_doc_name": top_doc,
            "top_article_num": top_art_num,
            "top_clause_num": top_clause_num,
        }

    def stitch(self, seed_chunks: List[Dict[str, Any]], max_chars: int = 3500) -> Dict[str, Any]:
        """Backward-compatible stitch method for existing pipeline callers."""
        return self.pack_evidence(seed_chunks, pack_type="multi_seed_2500_chars", max_chars=max_chars)
