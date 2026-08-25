import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.hybrid_fusion import reciprocal_rank_fusion
from src.retrieval.bm25_retriever import SimpleBM25

def test_reciprocal_rank_fusion():
    list1 = ["chunk_a", "chunk_b", "chunk_c"]
    list2 = ["chunk_b", "chunk_a", "chunk_d"]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    top_chunk, score = fused[0]
    assert top_chunk in ["chunk_a", "chunk_b"]
    assert len(fused) == 4

def test_simple_bm25():
    corpus = [
        {"id": "c1", "text": "quy định về xử phạt tốc độ ô tô"},
        {"id": "c2", "text": "thủ tục đăng ký bảo hiểm xã hội"}
    ]
    bm25 = SimpleBM25(corpus)
    results = bm25.search("xử phạt ô tô", top_k=1)
    assert len(results) == 1
    assert results[0][0] == "c1"
