"""Candidate generation, factual entity snapping, and candidate ensemble construction for Task 2."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.common.normalize import clean_legal_text, extract_legal_signals, prettify_doc_title

DATE_VERBATIM_REGEX = re.compile(r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})', re.IGNORECASE)
DATE_SHORT_REGEX = re.compile(r'\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b')
MONEY_REGEX = re.compile(r'\b(\d{1,3}(?:\.\d{3})+)\s*(?:đồng|VNĐ)?\b', re.IGNORECASE)

INTERNAL_TAGS_REGEX = re.compile(r'\[DOCUMENT\]\s*.*?\n|\[ARTICLE\]\s*|\[CLAUSE\]\s*', re.IGNORECASE)
IMAGE_CAPTION_REGEX = re.compile(r'\(Hình từ Internet\)|\(Ảnh từ Internet\)|\(Nguồn:.*?\)|Hình từ Internet|Ảnh từ Internet', re.IGNORECASE)
DUPLICATE_PREFIX_REGEX = re.compile(r'\b(ngày|tháng|năm)\s+\1\b', re.IGNORECASE)

SOURCE_HEADER = "\n\nTrích dẫn quy định:\n"


def clean_statutory_text(raw_text: str) -> str:
    """Strip internal pipeline tags, internet image captions, and noise from statutory text."""
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


def trim_at_complete_sentence(text: str, max_chars: int = 1200) -> str:
    """Trim text at complete clause/sentence boundary (., ;, or newline) up to max_chars."""
    if not text or len(text) <= max_chars:
        return text.strip()

    truncated = text[:max_chars]
    # Search backwards for sentence-ending punct or newline
    last_dot = max(truncated.rfind(".\n"), truncated.rfind(". "), truncated.rfind(";\n"), truncated.rfind("; "))
    if last_dot > max_chars * 0.4:
        return truncated[:last_dot + 1].strip()

    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.4:
        return truncated[:last_newline].strip()

    return truncated.rstrip() + "..."


def build_citation_header(doc_name: str, article_num: str = "", clause_num: str = "") -> str:
    """Build canonical statutory citation header (e.g. Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP quy định:)."""
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
    """Snap dates and factual entities in generated text to verbatim statutory evidence forms."""
    if not generated_text:
        return ""
    if not evidence_text:
        return generated_text.strip()

    result = IMAGE_CAPTION_REGEX.sub('', generated_text).strip()
    result = deduplicate_repetitive_lines(result)

    # 1. Multi-Date Snap: parse short dates and map each individually to matching verbatim date
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
        return f"ngày {d:02d} tháng {mo:02d} năm {yr}"

    result = DATE_SHORT_REGEX.sub(replace_short_date, result)

    # Clean double prefixes
    result = DUPLICATE_PREFIX_REGEX.sub(r'\1', result)
    result = re.sub(r'(?i)\bngày\s+ngày\b', 'ngày', result)
    result = re.sub(r'(?i)\btháng\s+tháng\b', 'tháng', result)
    result = re.sub(r'(?i)\bnăm\s+năm\b', 'năm', result)

    return result


def apply_strategy_f(answer: str, evidence: str, max_chars: int = 1500) -> str:
    """Conditionally append structured statutory evidence if answer is concise and not already quoting evidence."""
    if not answer:
        answer = ""
    ans_clean = answer.strip()
    if not evidence:
        return ans_clean

    clean_ev = clean_statutory_text(evidence)
    if not clean_ev:
        return ans_clean

    # Avoid duplicate appending if already quotes evidence
    if len(ans_clean) > 900 or clean_ev[:100] in ans_clean:
        return ans_clean

    truncated = clean_ev[:max_chars].strip()
    return f"{ans_clean}{SOURCE_HEADER}{truncated}"


def generate_candidate_ensemble(
    gen_ans: str = "",
    evidence: str = "",
    exact_ans: str = "",
    fuzzy_ans: str = "",
    doc_name: str = "",
    art_num: str = "",
    clause_num: str = "",
    evidence_packs: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Generate all candidate answer variations for OOF cross-fitting and inference selection."""
    clean_ev = clean_statutory_text(evidence)
    header = build_citation_header(doc_name, art_num, clause_num)

    snapped = snap_facts_to_evidence(gen_ans, clean_ev) if gen_ans else ""
    focused_ext = clean_ev[:800] if clean_ev else ""
    stitched_ext = f"{header}\n{clean_ev[:1500]}" if clean_ev else header

    # Section 15: Structured complete clause/sentence extracts
    focused_clause = f"{header}\n{trim_at_complete_sentence(clean_ev, max_chars=800)}" if clean_ev else header

    candidates: Dict[str, str] = {
        "exact_memory": exact_ans,
        "fuzzy_memory": fuzzy_ans,
        "focused_extract": focused_ext,
        "stitched_extract": stitched_ext,
        "focused_complete_clause": focused_clause,
        "generated": gen_ans,
        "snapped": snapped,
        "strategy_f_300": apply_strategy_f(snapped or gen_ans, clean_ev, max_chars=300),
        "strategy_f_600": apply_strategy_f(snapped or gen_ans, clean_ev, max_chars=600),
        "strategy_f_1000": apply_strategy_f(snapped or gen_ans, clean_ev, max_chars=1000),
        "strategy_f_1500": apply_strategy_f(snapped or gen_ans, clean_ev, max_chars=1500),
    }

    if evidence_packs:
        for p_name, p_text in evidence_packs.items():
            if p_text:
                candidates[f"pack_{p_name}"] = f"{header}\n{clean_statutory_text(p_text)}"

        if "top2_relevance" in evidence_packs and evidence_packs["top2_relevance"]:
            t2 = clean_statutory_text(evidence_packs["top2_relevance"])
            candidates["top2_relevance_complete_units"] = f"{header}\n{trim_at_complete_sentence(t2, max_chars=2000)}"

        if "full_article" in evidence_packs and evidence_packs["full_article"]:
            fa = clean_statutory_text(evidence_packs["full_article"])
            candidates["primary_article_budgeted_units"] = f"{header}\n{trim_at_complete_sentence(fa, max_chars=2500)}"

    return {k: v for k, v in candidates.items() if v}
