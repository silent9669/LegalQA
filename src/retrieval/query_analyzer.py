import re
from src.data.canonical import normalize_vietnamese_text

ARTICLE_QUERY_PATTERN = re.compile(r'Điều\s+([0-9]+[a-z]?)', re.IGNORECASE)
CLAUSE_QUERY_PATTERN = re.compile(r'khoản\s+([0-9]+)', re.IGNORECASE)

DOC_NUMBER_QUERY_PATTERN = re.compile(
    r'\b(\d+/\d+/(?:ND-CP|NĐ-CP|TTg|CP|TT-[A-ZĐ]+|QD-[A-ZĐ]+|QĐ-[A-ZĐ]+|QD-TTg|QĐ-TTg|NQ-[A-ZĐ]+|NQ-CP|QH\d+)'
    r'|\d+/(?:QD-[A-ZĐ]+|QĐ-[A-ZĐ]+|NQ-[A-ZĐ]+|QH\d+|CP)'
    r'|TCVN\s+[0-9\s\-:]+'
    r'|QCVN\s+[0-9\s\-:]+/[A-ZĐ]+)\b',
    re.IGNORECASE
)

NAMED_CODES_MAP = {
    "tố tụng hình sự": ["tố tụng hình sự", "to-tung-hinh-su", "101/2015/QH13"],
    "tố tụng dân sự": ["tố tụng dân sự", "to-tung-dan-su", "92/2015/QH13"],
    "hình sự": ["bộ luật hình sự", "hinh-su", "100/2015/QH13"],
    "dân sự": ["bộ luật dân sự", "dan-su", "91/2015/QH13"],
    "lao động": ["bộ luật lao động", "lao-dong", "45/2019/QH14"],
    "doanh nghiệp": ["luật doanh nghiệp", "doanh-nghiep", "59/2020/QH14"],
    "đất đai": ["luật đất đai", "dat-dai", "45/2013/QH13", "31/2024/QH15"],
    "hôn nhân": ["hôn nhân và gia đình", "hon-nhan", "52/2014/QH13"],
    "bảo hiểm xã hội": ["bảo hiểm xã hội", "bao-hiem-xa-hoi", "58/2014/QH13"],
    "giao thông đường bộ": ["giao thông đường bộ", "100/2019/NĐ-CP", "123/2021/NĐ-CP"]
}

def analyze_query(query: str) -> dict:
    """
    Extracts structured legal entities from a user question:
    - doc_number: e.g. '13/2023/NĐ-CP'
    - article: e.g. '43'
    - clause: e.g. '1'
    - named_code: e.g. 'tố tụng hình sự'
    - question_type: 'penalty', 'date', 'duration', 'authority', 'condition'
    """
    q_norm = normalize_vietnamese_text(query)
    q_lower = q_norm.lower()

    # 1. Document number
    doc_m = DOC_NUMBER_QUERY_PATTERN.search(q_norm)
    doc_num = doc_m.group(0).strip().replace('ND-CP', 'NĐ-CP').replace('QD-', 'QĐ-') if doc_m else None

    # 2. Article & Clause
    art_m = ARTICLE_QUERY_PATTERN.search(q_norm)
    art_num = art_m.group(1).strip() if art_m else None

    clause_m = CLAUSE_QUERY_PATTERN.search(q_norm)
    clause_num = clause_m.group(1).strip() if clause_m else None

    # 3. Named Codes
    matched_code_keywords = []
    for code_name, keywords in NAMED_CODES_MAP.items():
        if code_name in q_lower:
            matched_code_keywords.extend(keywords)

    # 4. Question Type
    q_type = "general"
    if any(w in q_lower for w in ["phạt", "mức phạt", "xử phạt", "bao nhiêu tiền"]):
        q_type = "penalty"
    elif any(w in q_lower for w in ["hiệu lực", "ngày nào", "từ khi nào", "áp dụng từ"]):
        q_type = "date"
    elif any(w in q_lower for w in ["bao lâu", "thời hạn", "trong thời gian"]):
        q_type = "duration"
    elif any(w in q_lower for w in ["ai", "cơ quan nào", "thẩm quyền"]):
        q_type = "authority"
    elif any(w in q_lower for w in ["có được", "có phải", "được không", "hay không"]):
        q_type = "condition"

    return {
        "query": query,
        "doc_number": doc_num,
        "article_number": art_num,
        "clause": clause_num,
        "code_keywords": matched_code_keywords,
        "question_type": q_type
    }
