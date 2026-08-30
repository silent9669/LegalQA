import re
from src.common.normalize import prettify_doc_title

DATE_VERBATIM_REGEX = re.compile(r'ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}', re.IGNORECASE)
DATE_SHORT_REGEX = re.compile(r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\b')
MONEY_REGEX = re.compile(r'\b\d{1,3}(?:\.\d{3})+(?:\s*đồng|\s*VNĐ)?\b', re.IGNORECASE)

INTERNAL_TAGS_REGEX = re.compile(r'\[DOCUMENT\]\s*.*?\n|\[ARTICLE\]\s*|\[CLAUSE\]\s*', re.IGNORECASE)

IMAGE_CAPTION_REGEX = re.compile(r'\(Hình từ Internet\)|\(Ảnh từ Internet\)|\(Nguồn:.*?\)|Hình từ Internet|Ảnh từ Internet', re.IGNORECASE)

SOURCE_HEADER = "\n\nTrích dẫn quy định:\n"

def clean_statutory_text(raw_text: str) -> str:
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
    if not generated_text or not evidence_text:
        return generated_text or ""

    result = IMAGE_CAPTION_REGEX.sub('', generated_text).strip()
    result = deduplicate_repetitive_lines(result)

    # 1. Snap Dates
    evidence_dates = DATE_VERBATIM_REGEX.findall(evidence_text)
    if evidence_dates:
        for ev_date in evidence_dates:
            result = DATE_SHORT_REGEX.sub(ev_date, result)

    return result

def apply_strategy_f(answer: str, evidence: str, max_chars: int = 1500) -> str:
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

def select_best_answer_candidate(candidates: dict, doc_name: str = "", article_num: str = "", clause_num: str = "") -> str:
    evidence = candidates.get("stitched_extract") or candidates.get("focused_extract") or ""
    clean_ev = clean_statutory_text(evidence)

    # 1. If we have a high quality generated or snapped answer with proper grounding
    if candidates.get("snapped") and len(candidates["snapped"].strip()) > 30 and not candidates["snapped"].startswith("Căn cứ quy định pháp luật:\n[DOCUMENT]"):
        ans = candidates["snapped"].strip()
        return apply_strategy_f(ans, clean_ev, max_chars=1500)

    # 2. Build structured extractive answer from stitched evidence
    header = build_citation_header(doc_name, article_num, clause_num)
    if clean_ev:
        base_ans = f"{header}\n{clean_ev[:800]}"
        return apply_strategy_f(base_ans, clean_ev, max_chars=1500)

    return header
