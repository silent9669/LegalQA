import re
import pandas as pd
from src.data.canonical import normalize_vietnamese_text

DOC_PATTERN = re.compile(
    r'(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|QCVN|TCVN)\s+'
    r'(?:số\s+)?([0-9]+/[0-9]+/[A-ZĐ\-]+|[0-9]+/[A-ZĐ\-]+|[A-ZĐ\-0-9\s]+(?:20[0-9]{2}|19[0-9]{2})?)',
    re.IGNORECASE
)
ARTICLE_PATTERN = re.compile(r'Điều\s+([0-9]+[a-z]?)', re.IGNORECASE)
CLAUSE_PATTERN = re.compile(r'khoản\s+([0-9]+)', re.IGNORECASE)
POINT_PATTERN = re.compile(r'điểm\s+([a-zđ])\b', re.IGNORECASE)

def parse_legal_citations(text: str) -> list[dict]:
    """
    Parses legal citations from answers (document number/name, article, clause, point).
    """
    citations = []
    text_norm = normalize_vietnamese_text(text)

    doc_matches = list(DOC_PATTERN.finditer(text_norm))
    art_matches = list(ARTICLE_PATTERN.finditer(text_norm))
    clause_matches = list(CLAUSE_PATTERN.finditer(text_norm))
    point_matches = list(POINT_PATTERN.finditer(text_norm))

    if doc_matches or art_matches:
        doc_num = doc_matches[0].group(1).strip() if doc_matches else None
        art_num = art_matches[0].group(1).strip() if art_matches else None
        clause_num = clause_matches[0].group(1).strip() if clause_matches else None
        point_num = point_matches[0].group(1).strip() if point_matches else None
        citations.append({
            "document_number": doc_num,
            "article": art_num,
            "clause": clause_num,
            "point": point_num
        })
    return citations

def mine_training_labels(df_qa: pd.DataFrame, df_chunks: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs supervision dataset mapping QA pairs to parsed citations and positive evidence chunks.
    """
    rows = []
    for _, qa in df_qa.iterrows():
        cits = parse_legal_citations(qa['answer'])
        rows.append({
            "qa_id": str(qa['id']),
            "query": qa['question'],
            "citations": cits,
            "answer": qa['answer']
        })
    return pd.DataFrame(rows)
