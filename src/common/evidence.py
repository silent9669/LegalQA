import re
from src.common.normalize import extract_legal_signals

CITATION_PATTERN = re.compile(
    r'(?:căn\s+cứ\s+|theo\s+)?(?:khoản\s+(\d+[a-zA-Z]?)\s+)?(?:điều\s+(\d+[a-zA-Z]?)\s+)(?:nghị\s+định|thông\s+tư|luật|quyết\s+định)?\s*([0-9]{1,5}/[0-9]{4}/[A-ZĐ\-]+|[0-9]{1,5}/[A-ZĐ\-]+)?',
    re.IGNORECASE
)

def parse_citations_from_answer(answer: str) -> list[dict]:
    citations = []
    signals = extract_legal_signals(answer)

    for m in CITATION_PATTERN.finditer(answer):
        clause = m.group(1)
        article = m.group(2)
        doc_num = m.group(3) or (signals["doc_numbers"][0] if signals["doc_numbers"] else "")

        if article:
            citations.append({
                "doc_number": doc_num.upper() if doc_num else "",
                "article": article,
                "clause": clause or ""
            })

    if not citations and (signals["articles"] or signals["doc_numbers"]):
        citations.append({
            "doc_number": signals["doc_numbers"][0] if signals["doc_numbers"] else "",
            "article": signals["articles"][0] if signals["articles"] else "",
            "clause": signals["clauses"][0] if signals["clauses"] else ""
        })
    return citations

def mine_hard_negatives(query_info: dict, all_chunks: list[dict], positive_chunk_id: str, positive_article_id: str) -> dict:
    doc_id = str(query_info.get("doc_id", "")).strip()

    same_article_wrong_clause = []
    same_doc_wrong_article = []
    different_doc_hard_negs = []

    for c in all_chunks:
        cid = c["chunk_id"]
        if cid == positive_chunk_id:
            continue
        c_doc = str(c.get("doc_id", "")).strip()
        c_parent_art = c.get("parent_article_id", "")

        if c_parent_art == positive_article_id:
            same_article_wrong_clause.append(cid)
        elif c_doc == doc_id:
            same_doc_wrong_article.append(cid)
        else:
            different_doc_hard_negs.append(cid)

    return {
        "same_article_wrong_clause": same_article_wrong_clause[:5],
        "same_doc_wrong_article": same_doc_wrong_article[:10],
        "different_doc": different_doc_hard_negs[:10]
    }
