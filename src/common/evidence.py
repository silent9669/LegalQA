"""Legal citation parsing, citation-to-chunk resolution, and hard-negative mining."""

from __future__ import annotations

import re
import urllib.parse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from src.common.normalize import (
    DOC_NUMBER_PATTERN,
    clean_legal_text,
    extract_canonical_doc_keys,
    extract_legal_signals,
    normalize_legal_number,
    slugify_legal_title,
)

CITATION_REGEX = re.compile(
    r'(?:căn\s+cứ\s+|theo\s+|quy\s+định\s+tại\s+)?'
    r'(?:khoản\s+(\d+[a-zA-Z]?)(?:\s*,\s*khoản\s*\d+[a-zA-Z]?)*\s+)?'
    r'Điều\s+(\d+[a-zA-Z]?)\s*'
    r'(?:(?:của\s+)?(?:Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư|Quyết\s+định|Nghị\s+quyết|Công\s+văn|Công\s+điện)\s+)?'
    r'([0-9]{1,5}/[0-9]{4}/[A-ZĐa-z0-9\-_]+|[0-9]{1,5}/[A-ZĐa-z0-9\-_]+|\d{1,5}/\d{4}|[A-ZĐa-zà-ỹ0-9\s,\-]+?)'
    r'(?=\s+(?:thì|được|quy\s+định|về|như\s+sau|có\s+hiệu\s+lực|hướng\s+dẫn|ban\s+hành|ngày|là)\b|[.,;\n]|$)',
    re.IGNORECASE
)


def parse_citations_from_answer(answer: str) -> List[Dict[str, str]]:
    """Extract all statutory citations (doc identifier, article, clause) from a legal answer."""
    citations: List[Dict[str, str]] = []
    if not answer or not isinstance(answer, str):
        return citations

    cleaned = clean_legal_text(answer)
    signals = extract_legal_signals(cleaned)

    # 1. Primary Regex
    for match in CITATION_REGEX.finditer(cleaned):
        clause, art, doc = match.groups()
        doc_clean = doc.strip().rstrip(',.;') if doc else ""
        if art:
            citations.append({
                "doc_identifier": doc_clean,
                "doc_number": doc_clean,
                "article": art,
                "clause": clause or "",
            })

    # 2. If regex didn't find any explicit citation, fallback to structured legal signals
    if not citations and signals["articles"]:
        for art in signals["articles"]:
            doc_num = signals["doc_numbers"][0] if signals["doc_numbers"] else ""
            clause = signals["clauses"][0] if signals["clauses"] else ""
            citations.append({
                "doc_identifier": doc_num,
                "doc_number": doc_num,
                "article": art,
                "clause": clause,
            })

    return citations


class CorpusLookupIndex:
    """Precomputed dictionary indexes over legal_chunks for O(1) resolution and negative mining."""

    def __init__(self, chunks_df: pd.DataFrame):
        self.doc_index: Dict[str, Set[str]] = defaultdict(set)
        self.art_clause_to_chunks: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.art_to_chunks: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.parent_art_to_chunks: Dict[str, List[str]] = defaultdict(list)
        self.doc_name_to_chunks: Dict[str, List[str]] = defaultdict(list)
        self.chunk_to_meta: Dict[str, Dict[str, Any]] = {}
        self.all_chunk_ids: List[str] = []

        if chunks_df.empty:
            return

        doc_col = "doc_name" if "doc_name" in chunks_df.columns else "doc_id"

        for _, row in chunks_df.iterrows():
            cid = str(row["chunk_id"])
            doc = str(row[doc_col])
            art = str(row.get("article_number", "")).lower()
            clause = str(row.get("clause_number", "")).lower() if pd.notna(row.get("clause_number")) else ""
            p_art = str(row.get("parent_article_id", ""))

            self.all_chunk_ids.append(cid)
            self.parent_art_to_chunks[p_art].append(cid)
            self.doc_name_to_chunks[doc].append(cid)

            item = {"chunk_id": cid, "parent_article_id": p_art, "doc_name": doc, "article": art, "clause": clause}
            self.chunk_to_meta[cid] = item
            self.art_to_chunks[(doc, art)].append(item)
            if clause:
                self.art_clause_to_chunks[(doc, art, clause)].append(item)

        unique_doc_names = chunks_df[doc_col].dropna().unique()
        for d in unique_doc_names:
            d_str = str(d).strip()
            for k in extract_canonical_doc_keys(d_str):
                self.doc_index[k].add(d_str)


def build_corpus_doc_index(chunks_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build a multi-key index mapping canonical doc keys to set of doc_names in corpus."""
    doc_index: Dict[str, Set[str]] = {}
    if chunks_df.empty or "doc_name" not in chunks_df.columns:
        return doc_index

    unique_doc_names = chunks_df["doc_name"].dropna().unique()
    for d in unique_doc_names:
        d_str = str(d).strip()
        keys = extract_canonical_doc_keys(d_str)
        for k in keys:
            if k not in doc_index:
                doc_index[k] = set()
            doc_index[k].add(d_str)

    return doc_index


def resolve_citations_to_chunks(
    citations: List[Dict[str, str]],
    chunks_df: pd.DataFrame,
    doc_index: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Resolve extracted citations into matching positive chunk_ids and parent_article_ids."""
    if not citations or chunks_df.empty:
        return []

    if isinstance(doc_index, CorpusLookupIndex):
        lookup = doc_index
    else:
        lookup = CorpusLookupIndex(chunks_df)

    resolved: List[Dict[str, Any]] = []

    for cit in citations:
        doc_id_str = cit.get("doc_identifier") or cit.get("doc_number", "")
        art = str(cit.get("article", "")).lower()
        clause = str(cit.get("clause", "")).lower()

        candidate_doc_names: Set[str] = set()
        for k in extract_canonical_doc_keys(doc_id_str):
            if k in lookup.doc_index:
                candidate_doc_names.update(lookup.doc_index[k])

        if not candidate_doc_names:
            slug = slugify_legal_title(doc_id_str)
            if len(slug) > 10:
                for k, doc_set in lookup.doc_index.items():
                    if len(k) > 10 and (k in slug or slug in k):
                        candidate_doc_names.update(doc_set)

        if not candidate_doc_names:
            continue

        matched_chunk_id = None
        parent_art_id = None

        for doc_name in candidate_doc_names:
            if clause and (doc_name, art, clause) in lookup.art_clause_to_chunks:
                items = lookup.art_clause_to_chunks[(doc_name, art, clause)]
                if items:
                    matched_chunk_id = items[0]["chunk_id"]
                    parent_art_id = items[0]["parent_article_id"]
                    break

            if (doc_name, art) in lookup.art_to_chunks:
                items = lookup.art_to_chunks[(doc_name, art)]
                if items:
                    matched_chunk_id = items[0]["chunk_id"]
                    parent_art_id = items[0]["parent_article_id"]
                    break

        if matched_chunk_id and parent_art_id:
            resolved.append({
                "citation": cit,
                "doc_names": list(candidate_doc_names),
                "positive_chunk_id": matched_chunk_id,
                "positive_article_id": parent_art_id,
            })

    return resolved


def mine_hard_negatives(
    positive_chunk_id: str = "",
    positive_article_id: str = "",
    positive_doc_name: str = "",
    chunks_df: Union[pd.DataFrame, List[Dict[str, Any]], CorpusLookupIndex] = None,
    max_per_type: int = 5,
    query_info: Optional[Dict[str, Any]] = None,
    all_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[str]]:
    """Mine corpus-structural hard negatives across 3 categories with positive filtering."""
    if all_chunks is not None and chunks_df is None:
        chunks_df = all_chunks
    if query_info is not None and not positive_doc_name:
        positive_doc_name = str(query_info.get("doc_name") or query_info.get("doc_id", "")).strip()

    if isinstance(chunks_df, CorpusLookupIndex):
        lookup = chunks_df
    elif isinstance(chunks_df, list):
        lookup = CorpusLookupIndex(pd.DataFrame(chunks_df))
    elif isinstance(chunks_df, pd.DataFrame):
        lookup = CorpusLookupIndex(chunks_df)
    else:
        return {"same_article_wrong_clause": [], "same_doc_wrong_article": [], "different_doc": []}

    # Type A: Same article, different clause
    type_a = [
        cid for cid in lookup.parent_art_to_chunks.get(positive_article_id, [])
        if cid != positive_chunk_id
    ][:max_per_type]

    # Type B: Same document, different article
    all_doc_chunks = lookup.doc_name_to_chunks.get(positive_doc_name, [])
    same_art_set = set(lookup.parent_art_to_chunks.get(positive_article_id, []))
    type_b = [
        cid for cid in all_doc_chunks
        if cid not in same_art_set
    ][:max_per_type]

    # Type C: Different document
    type_c = []
    for cid in lookup.all_chunk_ids[:max_per_type * 10]:
        if cid not in all_doc_chunks and cid != positive_chunk_id:
            type_c.append(cid)
            if len(type_c) >= max_per_type:
                break

    return {
        "same_article_wrong_clause": type_a,
        "same_doc_wrong_article": type_b,
        "different_doc": type_c,
    }


def mine_retrieval_hard_negatives(
    qa_id: str,
    question: str,
    positive_chunk_ids: Set[str],
    positive_article_ids: Set[str],
    positive_doc_names: Set[str],
    retrieved_candidates: List[Dict[str, Any]],
    lookup: CorpusLookupIndex,
    max_negatives: int = 15,
) -> List[Dict[str, Any]]:
    """Mine true hard negatives from actual retrieval candidates, strictly excluding all resolved positives."""
    negatives: List[Dict[str, Any]] = []
    seen_chunk_ids: Set[str] = set(positive_chunk_ids)

    for cand in retrieved_candidates:
        cid = str(cand.get("chunk_id", "")).strip()
        if not cid or cid in seen_chunk_ids:
            continue

        p_art = str(cand.get("parent_article_id", "")).strip()
        if p_art and p_art in positive_article_ids:
            # Skip any chunk belonging to a positive article
            continue

        doc_name = str(cand.get("doc_name", "")).strip()

        # Categorize negative difficulty
        if doc_name and doc_name in positive_doc_names:
            neg_type = "same_doc_wrong_article"
        else:
            neg_type = "cross_doc_false_positive"

        seen_chunk_ids.add(cid)
        negatives.append({
            "qa_id": qa_id,
            "negative_chunk_id": cid,
            "negative_article_id": p_art,
            "negative_doc_name": doc_name,
            "negative_type": neg_type,
            "retrieval_rank": cand.get("rank", 0),
            "retrieval_score": cand.get("score", 0.0),
        })

        if len(negatives) >= max_negatives:
            break

    return negatives
