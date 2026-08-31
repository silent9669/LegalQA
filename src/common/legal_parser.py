"""Hierarchical Vietnamese Legal Document Parser (Chương -> Mục -> Điều -> Khoản -> Điểm)."""

from __future__ import annotations

import re
import urllib.parse
from typing import Any, Dict, List, Optional

from src.common.normalize import (
    clean_legal_text,
    extract_canonical_doc_keys,
    normalize_legal_number,
    slugify_legal_title,
    tokenize_vietnamese,
)

# Hierarchy regex patterns
CHAPTER_HEADER_REGEX = re.compile(r'(?:^|\n)\s*(?:Chương\s+([IVXLCDM\d]+))[\.\:\s]*(.*?)(?:\n|$)', re.IGNORECASE)
SECTION_HEADER_REGEX = re.compile(r'(?:^|\n)\s*(?:Mục\s+(\d+))[\.\:\s]*(.*?)(?:\n|$)', re.IGNORECASE)

ARTICLE_SPLIT_REGEX = re.compile(r'(?=(?:^|\n)\s*Điều\s+\d+[a-zA-Z]?[\.\s])', re.IGNORECASE | re.MULTILINE)
ARTICLE_HEADER_REGEX = re.compile(r'^(?:Điều\s+(\d+[a-zA-Z]?))[\.\:\s]*(.*?)(?:\n|$)', re.IGNORECASE)

CLAUSE_SPLIT_REGEX = re.compile(r'(?=(?:^|\n)\s*\d+\.\s+)', re.MULTILINE)
CLAUSE_HEADER_REGEX = re.compile(r'^(\d+)\.\s*(.*)', re.DOTALL)


def extract_doc_metadata(doc_name: str) -> Dict[str, Optional[str]]:
    """Extract legal number and year from doc_name."""
    if not doc_name:
        return {"legal_number": None, "year": None}
    unquoted = urllib.parse.unquote(str(doc_name))
    unquoted = re.sub(r'-\d{5,8}$', '', unquoted)

    # 1. Number / Year / Org
    m1 = re.search(r'(\d+)-(\d{4})-([A-Za-z0-9\-]+)', unquoted)
    if m1:
        num, year, org = m1.groups()
        legal_num = normalize_legal_number(f"{num}/{year}/{org}")
        return {"legal_number": legal_num, "year": year}

    # 2. Number / Org / Year
    m2 = re.search(r'(\d+)-([A-Za-z0-9\-]+)-(\d{4})', unquoted)
    if m2:
        num, org, year = m2.groups()
        legal_num = normalize_legal_number(f"{num}/{org}")
        return {"legal_number": legal_num, "year": year}

    # 3. Year match
    m3 = re.search(r'\b(19\d{2}|20\d{2})\b', unquoted)
    year = m3.group(1) if m3 else None
    return {"legal_number": None, "year": year}


def parse_legal_document(doc_id: str, doc_name: str, passage: str) -> List[Dict[str, Any]]:
    """Parse a single legal document passage into hierarchical clause/article chunks with accurate character spans."""
    chunks: List[Dict[str, Any]] = []
    if not passage or not isinstance(passage, str):
        return chunks

    doc_id_clean = str(doc_id).strip()
    doc_meta = extract_doc_metadata(doc_name)
    legal_number = doc_meta["legal_number"]
    doc_year = doc_meta["year"]

    raw_articles = ARTICLE_SPLIT_REGEX.split(passage)
    current_chapter: Optional[str] = None
    current_section: Optional[str] = None

    cursor = 0
    for art_idx, art_text in enumerate(raw_articles):
        art_clean = art_text.strip()
        if not art_clean:
            continue

        art_pos = passage.find(art_text, cursor)
        if art_pos == -1:
            art_pos = cursor
        cursor = art_pos + len(art_text)

        # Check for Chapter or Section markers
        chap_match = CHAPTER_HEADER_REGEX.search(art_clean)
        if chap_match:
            current_chapter = chap_match.group(1)
        sec_match = SECTION_HEADER_REGEX.search(art_clean)
        if sec_match:
            current_section = sec_match.group(1)

        art_match = ARTICLE_HEADER_REGEX.search(art_clean)
        if not art_match:
            # Preamble: only emit if it contains substantive legal rules (> 60 characters and not just titles)
            if len(art_clean) > 60 and not (art_clean.startswith("Chương ") or art_clean.startswith("Mục ")):
                art_num = f"preamble_{art_idx}"
                parent_article_id = f"doc{doc_id_clean}_art_{art_idx}"
                full_raw = f"[DOCUMENT] {doc_name}\n[ARTICLE] {art_clean}"
                chunk_id = f"{parent_article_id}_full"
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id_clean,
                    "doc_name": doc_name,
                    "legal_number": legal_number,
                    "year": doc_year,
                    "chapter_number": current_chapter,
                    "section_number": current_section,
                    "article_number": str(art_num),
                    "clause_number": None,
                    "point_label": None,
                    "parent_article_id": parent_article_id,
                    "parent_clause_id": None,
                    "text_raw": full_raw,
                    "text_norm": tokenize_vietnamese(full_raw),
                    "start_char": art_pos,
                    "end_char": art_pos + len(art_text),
                })
            continue

        art_num = art_match.group(1)
        art_title = art_match.group(2).strip()
        parent_article_id = f"doc{doc_id_clean}_art{art_num}"

        clauses = CLAUSE_SPLIT_REGEX.split(art_clean)
        # Check if we have multiple numbered clauses
        numbered_clauses = [c for c in clauses if CLAUSE_HEADER_REGEX.match(c.strip())]

        if numbered_clauses:
            clause_cursor = 0
            for c_text in numbered_clauses:
                c_clean = c_text.strip()
                if not c_clean:
                    continue

                c_pos = art_text.find(c_text, clause_cursor)
                if c_pos == -1:
                    c_pos = clause_cursor
                clause_cursor = c_pos + len(c_text)

                clause_start = art_pos + c_pos
                clause_end = clause_start + len(c_text)

                c_match = CLAUSE_HEADER_REGEX.match(c_clean)
                clause_num = c_match.group(1) if c_match else "1"
                parent_clause_id = f"{parent_article_id}_p{clause_num}"

                header_prefix = f"[DOCUMENT] {doc_name}\n[ARTICLE] Điều {art_num}. {art_title}\n[CLAUSE] {clause_num}. "
                full_raw = f"{header_prefix}\n{c_clean}"
                chunk_id = parent_clause_id

                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id_clean,
                    "doc_name": doc_name,
                    "legal_number": legal_number,
                    "year": doc_year,
                    "chapter_number": current_chapter,
                    "section_number": current_section,
                    "article_number": str(art_num),
                    "clause_number": str(clause_num),
                    "point_label": None,
                    "parent_article_id": parent_article_id,
                    "parent_clause_id": parent_clause_id,
                    "text_raw": full_raw,
                    "text_norm": tokenize_vietnamese(full_raw),
                    "start_char": clause_start,
                    "end_char": clause_end,
                })
        else:
            # Single complete article chunk
            full_raw = f"[DOCUMENT] {doc_name}\n[ARTICLE] Điều {art_num}. {art_title}\n{art_clean}"
            chunk_id = f"{parent_article_id}_full"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id_clean,
                "doc_name": doc_name,
                "legal_number": legal_number,
                "year": doc_year,
                "chapter_number": current_chapter,
                "section_number": current_section,
                "article_number": str(art_num),
                "clause_number": None,
                "point_label": None,
                "parent_article_id": parent_article_id,
                "parent_clause_id": None,
                "text_raw": full_raw,
                "text_norm": tokenize_vietnamese(full_raw),
                "start_char": art_pos,
                "end_char": art_pos + len(art_text),
            })

    return chunks
