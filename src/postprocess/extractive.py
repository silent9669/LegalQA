import re
from src.data.canonical import normalize_vietnamese_text

AMOUNT_RANGE_PATTERN = re.compile(
    r'(?:phạt\s+tiền\s+)?(từ\s+[0-9]{1,3}(?:\.[0-9]{3})*\s*(?:đồng|triệu đồng|nghìn đồng)\s+đến\s+[0-9]{1,3}(?:\.[0-9]{3})*\s*(?:đồng|triệu đồng|nghìn đồng))',
    re.IGNORECASE
)
SINGLE_AMOUNT_PATTERN = re.compile(
    r'([0-9]{1,3}(?:\.[0-9]{3})*\s*(?:đồng|triệu đồng|nghìn đồng))',
    re.IGNORECASE
)
DATE_PATTERN = re.compile(
    r'(?:ngày\s+)?([0-9]{1,2}(?:/[0-9]{1,2}/[0-9]{4}|\s+tháng\s+[0-9]{1,2}\s+năm\s+[0-9]{4}))',
    re.IGNORECASE
)
DURATION_PATTERN = re.compile(
    r'([0-9]{1,2}\s*(?:ngày|tháng|năm|giờ|ngày làm việc))\b',
    re.IGNORECASE
)

def format_document_citation(doc_num: str, doc_name: str) -> str:
    """
    Formats formal Vietnamese document citation (e.g. 'Nghị định 90/2017/NĐ-CP').
    """
    if doc_num:
        if not any(k in doc_num for k in ['Nghị định', 'Thông tư', 'Quyết định', 'Luật', 'Bộ luật', 'Nghị quyết', 'TCVN', 'QCVN']):
            if 'ND-CP' in doc_name or 'NĐ-CP' in doc_name or 'ND' in doc_name:
                return f"Nghị định {doc_num}"
            elif 'TT' in doc_name:
                return f"Thông tư {doc_num}"
            elif 'QD' in doc_name or 'QĐ' in doc_name:
                return f"Quyết định {doc_num}"
            elif 'QH' in doc_name or 'Luat' in doc_name or 'Luật' in doc_name:
                return f"Luật {doc_num}"
            elif 'NQ' in doc_name:
                return f"Nghị quyết {doc_num}"
            return doc_num
        return doc_num
    return doc_name

def extract_penalty_amount(text: str) -> str:
    m = AMOUNT_RANGE_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    m_single = SINGLE_AMOUNT_PATTERN.search(text)
    if m_single:
        return m_single.group(1).strip()
    return ""

def extract_conclusion_sentence(question: str, evidence_text: str) -> str:
    """
    Synthesizes a tailored concluding statement answering the user's specific question
    (penalty, effective date, duration, condition).
    """
    q_lower = question.lower()

    # 1. Penalty question
    if any(w in q_lower for w in ["phạt", "mức phạt", "xử phạt", "bao nhiêu tiền"]):
        penalty = extract_penalty_amount(evidence_text)
        if penalty:
            remedy_match = re.search(r'(?:Buộc|Tịch thu|Đình chỉ)\s+([^.\n]+)', evidence_text)
            remedy_str = f" Đồng thời, người vi phạm còn bị {remedy_match.group(0).strip()}." if remedy_match else ""
            clean_q = re.sub(r'^(?:hỏi|cho hỏi|theo quy định|quy định về)?\s*', '', question, flags=re.IGNORECASE)
            clean_q = clean_q.rstrip('?').strip()
            return f"Theo đó, {clean_q} với mức phạt tiền {penalty}.{remedy_str}"

    # 2. Effective date question
    if any(w in q_lower for w in ["hiệu lực", "ngày nào", "từ khi nào", "áp dụng từ"]):
        date_m = DATE_PATTERN.search(evidence_text)
        if date_m:
            return f"Theo đó, văn bản chính thức có hiệu lực thi hành từ {date_m.group(0).strip()}."

    # 3. Duration / deadline question
    if any(w in q_lower for w in ["bao lâu", "thời hạn", "trong thời gian"]):
        dur_m = DURATION_PATTERN.search(evidence_text)
        if dur_m:
            return f"Theo đó, thời hạn thực hiện là {dur_m.group(0).strip()}."

    # General conclusion
    clean_q = question.rstrip('?').strip()
    return f"Theo đó, {clean_q} được thực hiện theo quy định nêu trên."

def generate_extractive_answer(question: str, evidence_chunks: list[dict]) -> str:
    """
    Generates a high-METEOR 3-part structured legal answer:
    1. Citation Preamble: Căn cứ ... quy định [về ...] như sau:
    2. Verbatim Quoted Body: full statutory clauses
    3. Contextual Conclusion: Theo đó, [kết luận chi tiết].
    """
    if not evidence_chunks:
        return "Theo quy định của pháp luật hiện hành, chưa có quy định chi tiết cho trường hợp này."

    top_chunk = evidence_chunks[0]
    doc_title = top_chunk.get('name') or top_chunk.get('document_title') or ''
    doc_num = top_chunk.get('document_number') or ''
    dieu = top_chunk.get('dieu') or ''
    art_num = top_chunk.get('article_number') or ''
    art_title = top_chunk.get('article_title') or ''
    khoan = top_chunk.get('khoan') or top_chunk.get('clause')
    content = top_chunk.get('content', '').strip()

    # Part 1: Clean Citation Preamble
    doc_citation = format_document_citation(doc_num, doc_title)

    basis_parts = []
    if khoan and str(khoan).lower() != 'nan' and str(khoan).strip():
        basis_parts.append(f"khoản {khoan}")
    if art_num:
        basis_parts.append(f"Điều {art_num}")
    elif dieu:
        basis_parts.append(dieu)
    if doc_citation:
        basis_parts.append(doc_citation)

    basis_str = " ".join(basis_parts)

    if art_title and art_title.lower() not in basis_str.lower():
        clean_title = art_title[0].lower() + art_title[1:] if len(art_title) > 1 else art_title.lower()
        preamble = f"Căn cứ {basis_str} quy định về {clean_title} như sau:"
    elif basis_str:
        preamble = f"Căn cứ {basis_str} quy định như sau:"
    else:
        preamble = "Căn cứ theo quy định của pháp luật hiện hành như sau:"

    # Part 2: Body
    body_text = content

    # Part 3: Conclusion
    all_evidence_text = "\n".join([c.get("content", "") for c in evidence_chunks])
    conclusion = extract_conclusion_sentence(question, all_evidence_text)

    return f"{preamble}\n{body_text}\n{conclusion}"
