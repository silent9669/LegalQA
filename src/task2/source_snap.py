"""Surface-form preservation, multi-date safety, entity snapping, and candidate selection."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from src.common.normalize import prettify_doc_title

DATE_VERBATIM_REGEX = re.compile(r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})', re.IGNORECASE)
DATE_SHORT_REGEX = re.compile(r'\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b')
MONEY_REGEX = re.compile(r'\b(\d{1,3}(?:\.\d{3})+)\s*(?:đồng|VNĐ)?\b', re.IGNORECASE)

INTERNAL_TAGS_REGEX = re.compile(r'\[DOCUMENT\]\s*.*?\n|\[ARTICLE\]\s*|\[CLAUSE\]\s*', re.IGNORECASE)
IMAGE_CAPTION_REGEX = re.compile(r'\(Hình từ Internet\)|\(Ảnh từ Internet\)|\(Nguồn:.*?\)|Hình từ Internet|Ảnh từ Internet', re.IGNORECASE)
DUPLICATE_PREFIX_REGEX = re.compile(r'\b(ngày|tháng|năm)\s+\1\b', re.IGNORECASE)

SOURCE_HEADER = "\n\nTrích dẫn quy định:\n"


def clean_statutory_text(raw_text: str) -> str:
    """Strip internal pipeline tags and noise captions from statutory text."""
    if not raw_text:
        return ""
    cleaned = INTERNAL_TAGS_REGEX.sub('', raw_text).strip()
    cleaned = IMAGE_CAPTION_REGEX.sub('', cleaned).strip()
    lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
    unique_lines = []
    seen = set()
    for l in lines:
        if l not in seen and not l.startswith("Điều preamble_") and l != "0.":
            seen.add(l)
            unique_lines.append(l)
    return "\n".join(unique_lines)


def build_citation_header(doc_name: str, article_num: str = "", clause_num: str = "") -> str:
    """Build canonical statutory citation header."""
    parts = []
    if clause_num and str(clause_num).lower() not in ("none", "nan", "0", "null", ""):
        parts.append(f"khoản {clause_num}")
    if article_num and str(article_num).lower() not in ("none", "nan", "0", "null", "") and not str(article_num).startswith("preamble"):
        parts.append(f"Điều {article_num}")

    clean_doc = prettify_doc_title(doc_name)
    if clean_doc:
        parts.append(clean_doc)

    if parts:
        joined = " ".join(parts)
        return f"Căn cứ {joined} quy định như sau:"
    return "Căn cứ quy định của pháp luật:"


def deduplicate_repetitive_lines(text: str) -> str:
    """Remove verbatim repeated lines while preserving structure."""
    if not text:
        return ""
    lines = text.split("\n")
    cleaned_lines = []
    seen = set()
    for l in lines:
        s = l.strip()
        if not s:
            cleaned_lines.append(l)
            continue
        if s not in seen:
            seen.add(s)
            cleaned_lines.append(l)
    return "\n".join(cleaned_lines)


def snap_facts_to_evidence(generated_text: str, evidence_text: str) -> str:
    """Snap dates, monetary amounts, and legal identifiers to exact evidence forms without global corruption."""
    if not generated_text:
        return ""
    if not evidence_text:
        return generated_text.strip()

    result = IMAGE_CAPTION_REGEX.sub('', generated_text).strip()
    result = deduplicate_repetitive_lines(result)

    # 1. Multi-Date Snap: parse short dates and map each individually to matching verbatim date if present
    evidence_dates = {}
    for match in DATE_VERBATIM_REGEX.finditer(evidence_text):
        d, m, y = match.group(1), match.group(2), match.group(3)
        key = (int(d), int(m), int(y))
        evidence_dates[key] = match.group(0)

    def replace_short_date(m: re.Match) -> str:
        d, mo, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        key = (d, mo, yr)
        if key in evidence_dates:
            return evidence_dates[key]
        return f"ngày {d:02d} tháng {mo} năm {yr}"
        return f"ngày {d:02d} tháng {mo:02d} năm {yr}"

    result = DATE_SHORT_REGEX.sub(replace_short_date, result)

    # Clean any accidental double prefix like 'ngày ngày 01...'
    result = DUPLICATE_PREFIX_REGEX.sub(r'\1', result)
    result = re.sub(r'(?i)\bngày\s+ngày\b', 'ngày', result)
    result = re.sub(r'(?i)\btháng\s+tháng\b', 'tháng', result)
    result = re.sub(r'(?i)\bnăm\s+năm\b', 'năm', result)

    return result


def apply_strategy_f(answer: str, evidence: str, max_chars: int = 1500) -> str:
    """Conditionally append clean statutory context if answer is short and not already quoting evidence."""
    if not answer:
        answer = ""
    ans_clean = answer.strip()
    if not evidence:
        return ans_clean

    clean_ev = clean_statutory_text(evidence)
    if not clean_ev:
        return ans_clean

    # If the answer already quotes substantial parts, avoid duplicate appending
    if len(ans_clean) > 800 or clean_ev[:80] in ans_clean:
        return ans_clean

    truncated = clean_ev[:max_chars].strip()
    return f"{ans_clean}{SOURCE_HEADER}{truncated}"


def select_best_answer_candidate(
    candidates: Dict[str, str],
    doc_name: str = "",
    article_num: str = "",
    clause_num: str = "",
) -> str:
    """Select the highest-quality candidate from extractive, generated, and snapped answers."""
    evidence = candidates.get("stitched_extract") or candidates.get("focused_extract") or ""
    clean_ev = clean_statutory_text(evidence)

    # 1. If we have a high-quality generated or snapped answer with proper grounding
    snapped = candidates.get("snapped", "").strip()
    if snapped and len(snapped) > 30 and not snapped.startswith("Căn cứ quy định pháp luật:\n[DOCUMENT]"):
        return apply_strategy_f(snapped, clean_ev, max_chars=1500)

    gen = candidates.get("generated", "").strip()
    if gen and len(gen) > 30:
        return apply_strategy_f(gen, clean_ev, max_chars=1500)

    # 2. Build structured extractive answer from stitched evidence
    header = build_citation_header(doc_name, article_num, clause_num)
    if clean_ev:
        base_ans = f"{header}\n{clean_ev[:800]}"
        return apply_strategy_f(base_ans, clean_ev, max_chars=1500)

    return header
