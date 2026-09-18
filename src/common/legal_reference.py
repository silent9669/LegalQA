"""Legal-reference retrieval arm: exact statute/document/article matching.

Document names in this corpus are slugs starting with a letter
(e.g. ``Thong-tu-12-2019-TT-BTP-...``), so digit-anchored ``re.match``
patterns silently yield an EMPTY index. This module uses ``re.search``
forms (canonical ``num/year`` anywhere, plus slug ``num...year``) and
refuses to masquerade an empty index as a working arm.

The index stores corpus POSITIONS (not one-text-per-id dicts), so repeated
chunk ids keep every fragment.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Canonical "100/2019" (or 100-2019) anywhere in the string.
_FORM_A = re.compile(r"(?<!\d)(\d{1,4})\s*[/\-]\s*((?:19|20)\d{2})(?!\d)")
# Year marker for slug form.
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_NUM = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")

# Query-side patterns.
_DOC = re.compile(
    r"(?:nghị\s*định|thông\s*tư|quyết\s*định|nghị\s*quyết|pháp\s*lệnh"
    r"|bộ\s*luật|luật)\s*(?:số\s*)?"
    r"(\d{1,4}\s*[/\-]\s*\d{2,4}(?:\s*/\s*[A-Za-zĐđ0-9\-]+)*)",
    re.I,
)
_BARE_NUM = re.compile(r"\b(\d{1,4}\s*/\s*\d{2,4}(?:\s*/\s*[A-Za-zĐđ0-9\-]+)*)")
_ART = re.compile(r"\bđiều\s*(\d{1,3}[a-zđ]?)\b", re.I)


def parse_doc_key(name: Any) -> Optional[str]:
    """Parse a document name/slug into a canonical ``num/year`` key.

    ``100/2019/NĐ-CP`` → ``100/2019``; ``Nghi-dinh-100-2019-ND-CP`` →
    ``100/2019``; ``Quyet-dinh-405-QD-BNV-2021`` → ``405/2021``.
    Returns None when no (number, year) pair is identifiable.
    """
    s = unicodedata.normalize("NFC", str(name or ""))
    if not s.strip():
        return None
    m = _FORM_A.search(s)
    if m:
        return f"{int(m.group(1))}/{m.group(2)}"
    y = _YEAR.search(s)
    if not y:
        return None
    before = _NUM.findall(s[: y.start()])
    after = _NUM.findall(s[y.end() :])
    num = before[-1] if before else (after[0] if after else None)
    return f"{int(num)}/{y.group(1)}" if num else None


def normalize_article_key(value: Any) -> Optional[str]:
    """Normalize an article number; preambles/placeholders map to None."""
    s = unicodedata.normalize("NFC", str(value or "")).strip().lower()
    if not s or s.startswith("preamble") or s in ("none", "nan", "null", "0", ""):
        return None
    m = re.search(r"(\d{1,3}[a-zđ]?)", s)
    return m.group(1) if m else None


def build_legal_reference_index(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build doc → positions and (doc, article) → positions maps.

    Returns (index, report). The report records coverage and whether the
    index is empty; an empty index must never be presented as enabled.
    """
    doc_positions: Dict[str, List[int]] = defaultdict(list)
    doc_article_positions: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    parsed_docs = 0
    for pos, row in enumerate(rows):
        key = parse_doc_key(row.get("doc_name", ""))
        if not key:
            continue
        parsed_docs += 1
        doc_positions[key].append(pos)
        article = normalize_article_key(row.get("article_number", ""))
        if article:
            doc_article_positions[(key, article)].append(pos)
    index = {"doc": dict(doc_positions), "doc_article": dict(doc_article_positions)}
    report = {
        "num_rows": len(rows),
        "rows_with_doc_key": parsed_docs,
        "num_doc_keys": len(doc_positions),
        "num_doc_article_keys": len(doc_article_positions),
        "is_empty": not doc_positions,
    }
    return index, report


def search_legal_references(
    queries: List[str],
    index: Dict[str, Any],
    rows: List[Dict[str, Any]],
    k: int = 100,
) -> List[List[Dict[str, Any]]]:
    """Exact statute lookup per query; returns chunk dicts per position.

    An empty index yields empty lists (a disabled arm), never an error, so
    the fusion treats it as a no-op. Queries without a document reference
    also yield empty lists.
    """
    doc_index = index.get("doc", {}) if index else {}
    doc_article_index = index.get("doc_article", {}) if index else {}
    out: List[List[Dict[str, Any]]] = []
    for query in queries:
        s = unicodedata.normalize("NFC", str(query or ""))
        match = _DOC.search(s) or _BARE_NUM.search(s)
        hits: List[Dict[str, Any]] = []
        if match and doc_index:
            key = parse_doc_key(match.group(1))
            if key:
                seen: set = set()
                art = _ART.search(s)
                if art:
                    article = normalize_article_key(art.group(1))
                    for pos in doc_article_index.get((key, article or ""), []):
                        if pos not in seen:
                            seen.add(pos)
                            hits.append(dict(rows[pos]))
                for pos in doc_index.get(key, []):
                    if pos not in seen:
                        seen.add(pos)
                        hits.append(dict(rows[pos]))
        out.append(hits[: max(0, int(k))])
    return out
