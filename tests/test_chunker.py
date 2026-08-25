import pytest
import os
import sys
import json
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.chunker import format_searchable_chunk, process_legal_chunks

def test_format_searchable_chunk():
    record = {
        "doc_id": "100062",
        "name": "Thong-tu-17-2022-TT-BGTVT",
        "dieu": "Điều 1. Sửa đổi một số điều",
        "khoan": "1",
        "content": "Nội dung quy định tại khoản 1"
    }
    raw_text, norm_text, searchable, doc_num, art_num, art_title = format_searchable_chunk(record)
    assert "thong-tu-17-2022-tt-bgtvt" in searchable
    assert "Điều 1." in raw_text
    assert "Khoản 1" in raw_text
    assert "Nội dung quy định tại khoản 1" in raw_text
    assert art_num == "1"
    assert "Sửa đổi một số điều" in art_title

def test_process_legal_chunks(tmp_path):
    jsonl_file = tmp_path / "chunks.jsonl"
    parquet_file = tmp_path / "legal_chunks.parquet"

    sample_items = [
        {
            "chunk_id": "c1",
            "doc_id": "100",
            "name": "Nghi-dinh-10-2020-ND-CP",
            "structure": "dieu",
            "dieu": "Điều 5. Quy định xử phạt",
            "khoan": "1",
            "content": "Xử phạt 5 triệu đồng"
        }
    ]

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for it in sample_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    df = process_legal_chunks(str(jsonl_file), str(parquet_file))
    assert len(df) == 1
    assert os.path.exists(parquet_file)
    assert "document_number" in df.columns
    assert "article_number" in df.columns
    assert "article_title" in df.columns
    assert "context_id" in df.columns
    assert "searchable_text" in df.columns
    assert df.iloc[0]["article_number"] == "5"
