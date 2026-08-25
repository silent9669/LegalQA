import re
import json
import os
import pandas as pd
from src.data.canonical import normalize_vietnamese_text

ARTICLE_NUM_PATTERN = re.compile(r'Điều\s+([0-9]+[a-z]?)', re.IGNORECASE)

def extract_document_number(name: str) -> str:
    """
    Extracts clean official Vietnamese legal document numbers from filenames / links.
    e.g. 'Nghi-dinh-100-2019-ND-CP-...' -> '100/2019/NĐ-CP'
    """
    if not name:
        return ''
    converted = re.sub(r'(\d+)-(\d+)-([A-ZĐ]+)-([A-ZĐ]+)', r'\1/\2/\3-\4', name, flags=re.IGNORECASE)
    converted = re.sub(r'(\d+)-(\d+)-([A-ZĐ]+)', r'\1/\2/\3', converted, flags=re.IGNORECASE)
    converted = re.sub(r'(\d+)-([A-ZĐ]+)-([A-ZĐ]+)', r'\1/\2-\3', converted, flags=re.IGNORECASE)
    converted = re.sub(r'(\d+)-([A-ZĐ]+)', r'\1/\2', converted, flags=re.IGNORECASE)

    pattern = re.compile(
        r'\b(\d+/\d+/(?:ND-CP|NĐ-CP|TTg|CP|TT-[A-ZĐ]+|QD-[A-ZĐ]+|QĐ-[A-ZĐ]+|QD-TTg|QĐ-TTg|NQ-[A-ZĐ]+|NQ-CP|QH\d+)'
        r'|\d+/(?:QD-[A-ZĐ]+|QĐ-[A-ZĐ]+|NQ-[A-ZĐ]+|QH\d+|CP)'
        r'|TCVN\s+[0-9\s\-:]+'
        r'|QCVN\s+[0-9\s\-:]+/[A-ZĐ]+)\b',
        re.IGNORECASE
    )
    m = pattern.search(converted)
    if m:
        val = m.group(0).strip()
        val = val.replace('ND-CP', 'NĐ-CP').replace('QD-', 'QĐ-').replace('QD-TTg', 'QĐ-TTg')
        return val
    return ''

def extract_article_info(dieu_str: str) -> tuple[str, str]:
    if not dieu_str:
        return "", ""
    m = ARTICLE_NUM_PATTERN.search(dieu_str)
    art_num = m.group(1) if m else ""

    parts = re.split(r'Điều\s+[0-9]+[a-z]?\.?\s*', dieu_str, flags=re.IGNORECASE)
    art_title = parts[1].strip() if len(parts) > 1 else dieu_str.strip()
    return art_num, art_title

def format_searchable_chunk(item: dict) -> tuple[str, str, str, str, str, str]:
    """
    Constructs raw_text (for LLM generation / source-snapping),
    normalized_text, and searchable_text (with prepended document title, article, clause)
    along with extracted document_number, article_number, and article_title.
    """
    doc_title = item.get('name') or ''
    dieu = item.get('dieu') or ''
    khoan = item.get('khoan')
    content = item.get('content') or ''
    link = item.get('link') or ''

    doc_num = extract_document_number(doc_title) or extract_document_number(link)
    art_num, art_title = extract_article_info(dieu)

    header_parts = []
    if doc_title:
        header_parts.append(f"Văn bản: {doc_title}")
    if doc_num:
        header_parts.append(f"Số văn bản: {doc_num}")
    if dieu:
        header_parts.append(f"{dieu}")
    if khoan and str(khoan).lower() != 'nan':
        header_parts.append(f"Khoản {khoan}")

    header_text = "\n".join(header_parts)
    raw_text = f"{header_text}\n{content}".strip() if header_text else content.strip()
    normalized_text = normalize_vietnamese_text(raw_text).lower()

    searchable_text = f"{doc_title} {doc_num} {dieu} {khoan or ''} {content}".strip()
    searchable_text = normalize_vietnamese_text(searchable_text).lower()

    return raw_text, normalized_text, searchable_text, doc_num, art_num, art_title

def process_legal_chunks(chunks_jsonl_path: str, output_parquet_path: str) -> pd.DataFrame:
    """
    Reads parsed chunks from chunks_output.jsonl, formats text representations,
    extracts hierarchical legal metadata, and writes out canonical legal_chunks.parquet.
    """
    records = []
    with open(chunks_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            raw_text, norm_text, search_text, doc_num, art_num, art_title = format_searchable_chunk(item)
            records.append({
                "chunk_id": str(item.get("chunk_id", "")),
                "context_id": str(item.get("doc_id", "")),
                "doc_id": str(item.get("doc_id", "")),
                "document_number": doc_num,
                "document_title": str(item.get("name", "")),
                "name": str(item.get("name", "")),
                "structure": str(item.get("structure", "")),
                "article_number": art_num,
                "article_title": art_title,
                "dieu": item.get("dieu"),
                "khoan": item.get("khoan"),
                "clause": item.get("khoan"),
                "part": item.get("part", 1),
                "n_parts": item.get("n_parts", 1),
                "content": str(item.get("content", "")),
                "raw_text": raw_text,
                "normalized_text": norm_text,
                "searchable_text": search_text
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
    df.to_parquet(output_parquet_path, index=False)
    return df
