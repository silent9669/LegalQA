import re
from src.common.normalize import clean_legal_text, tokenize_vietnamese

ARTICLE_SPLIT_REGEX = re.compile(r'(?=(?:^|\n)\s*Điều\s+\d+[a-zA-Z]?[\.\s])', re.IGNORECASE | re.MULTILINE)
ARTICLE_HEADER_REGEX = re.compile(r'^(?:Điều\s+(\d+[a-zA-Z]?))[\.\:\s]*(.*?)(?:\n|$)', re.IGNORECASE)
CLAUSE_SPLIT_REGEX = re.compile(r'(?=(?:^|\n)\s*\d+\.\s+)', re.MULTILINE)
CLAUSE_HEADER_REGEX = re.compile(r'^(\d+)\.\s*(.*)', re.DOTALL)

def parse_legal_document(doc_id: str, doc_name: str, passage: str) -> list[dict]:
    chunks = []
    if not passage:
        return chunks

    doc_id_clean = str(doc_id).strip()
    raw_articles = ARTICLE_SPLIT_REGEX.split(passage)

    char_offset = 0
    for art_idx, art_text in enumerate(raw_articles):
        art_clean = art_text.strip()
        if not art_clean:
            char_offset += len(art_text)
            continue

        art_match = ARTICLE_HEADER_REGEX.search(art_clean)
        if art_match:
            art_num = art_match.group(1)
            art_title = art_match.group(2).strip()
            parent_article_id = f"doc{doc_id_clean}_art{art_num}"
        else:
            art_num = f"preamble_{art_idx}"
            art_title = ""
            parent_article_id = f"doc{doc_id_clean}_art_{art_idx}"

        clauses = CLAUSE_SPLIT_REGEX.split(art_clean)
        if len(clauses) > 1:
            for clause_idx, clause_text in enumerate(clauses):
                c_clean = clause_text.strip()
                if not c_clean:
                    continue
                c_match = CLAUSE_HEADER_REGEX.match(c_clean)
                clause_num = c_match.group(1) if c_match else str(clause_idx)

                header_prefix = f"[DOCUMENT] {doc_name}\n[ARTICLE] Điều {art_num}. {art_title}\n[CLAUSE] {clause_num}. "
                full_raw = f"{header_prefix}\n{c_clean}"

                chunk_id = f"{parent_article_id}_p{clause_num}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id_clean,
                    "doc_name": doc_name,
                    "parent_article_id": parent_article_id,
                    "article_number": str(art_num),
                    "clause_number": str(clause_num),
                    "text_raw": full_raw,
                    "text_norm": tokenize_vietnamese(full_raw),
                    "start_char": char_offset,
                    "end_char": char_offset + len(art_text)
                })
        else:
            full_raw = f"[DOCUMENT] {doc_name}\n[ARTICLE] Điều {art_num}. {art_title}\n{art_clean}"
            chunk_id = f"{parent_article_id}_full"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id_clean,
                "doc_name": doc_name,
                "parent_article_id": parent_article_id,
                "article_number": str(art_num),
                "clause_number": None,
                "text_raw": full_raw,
                "text_norm": tokenize_vietnamese(full_raw),
                "start_char": char_offset,
                "end_char": char_offset + len(art_text)
            })
        char_offset += len(art_text)

    return chunks
