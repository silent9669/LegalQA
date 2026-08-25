import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reranking.cross_encoder import SimpleLexicalReranker

def test_simple_lexical_reranker():
    query = "xử phạt chạy quá tốc độ ô tô"
    candidates = [
        {"chunk_id": "c1", "searchable_text": "thủ tục sang tên xe máy", "content": "sang tên"},
        {"chunk_id": "c2", "searchable_text": "xử phạt chạy quá tốc độ ô tô từ 10 đến 20 km/h", "content": "phạt tiền 5 triệu"}
    ]
    reranker = SimpleLexicalReranker()
    ranked = reranker.rank(query, candidates, top_k=2)
    assert ranked[0]["chunk_id"] == "c2"
