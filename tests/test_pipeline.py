import pytest
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import LegalQAPipeline
from src.memory.exact_memory import ExactMemory
from src.postprocess.article_stitcher import ArticleStitcher
from scripts.predict import run_prediction

def test_pipeline_exact_memory_branch():
    mem = ExactMemory({"by_id": {"test_1": "Exact Answer 1"}})
    pipe = LegalQAPipeline(exact_memory=mem, retriever=None, reranker=None)
    pred = pipe.predict("test_1", "câu hỏi bất kỳ")
    assert pred == "Exact Answer 1"

def test_pipeline_with_article_stitcher():
    chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "dieu": "Điều 1", "part": 1, "n_parts": 2, "content": "Khoản 1", "name": "Luật A"},
        {"chunk_id": "c2", "doc_id": "d1", "dieu": "Điều 1", "part": 2, "n_parts": 2, "content": "Khoản 2", "name": "Luật A"}
    ]
    stitcher = ArticleStitcher(chunks)

    class MockRetriever:
        def search(self, q, top_k=20):
            return [("c1", 10.0)]
        chunk_map = {"c1": chunks[0], "c2": chunks[1]}

    mem = ExactMemory({})
    pipe = LegalQAPipeline(exact_memory=mem, retriever=MockRetriever(), article_stitcher=stitcher)
    pred = pipe.predict("q1", "Quy định điều 1 thế nào?")
    assert "Khoản 1\nKhoản 2" in pred
    assert "Căn cứ" in pred
