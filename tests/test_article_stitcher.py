import pytest
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.postprocess.article_stitcher import ArticleStitcher

def test_article_stitcher_single_part():
    chunks = [
        {
            "chunk_id": "doc1_art5_p1",
            "doc_id": "doc1",
            "dieu": "Điều 5. Xử phạt",
            "part": 1,
            "n_parts": 1,
            "content": "Khoản 1. Phạt tiền 5 triệu.",
            "name": "Nghị định 10"
        }
    ]
    stitcher = ArticleStitcher(chunks)
    stitched = stitcher.get_full_article("doc1", "Điều 5. Xử phạt")
    assert stitched is not None
    assert stitched["content"] == "Khoản 1. Phạt tiền 5 triệu."

def test_article_stitcher_multiple_sibling_parts():
    chunks = [
        {
            "chunk_id": "doc1_art17_p2",
            "doc_id": "doc1",
            "dieu": "Điều 17",
            "part": 2,
            "n_parts": 3,
            "content": "Khoản 2. Phạt tiền từ 6.000.000 đến 8.000.000 đồng.",
            "name": "Nghị định 90"
        },
        {
            "chunk_id": "doc1_art17_p1",
            "doc_id": "doc1",
            "dieu": "Điều 17",
            "part": 1,
            "n_parts": 3,
            "content": "Khoản 1. Phạt tiền 4.000.000 đồng.",
            "name": "Nghị định 90"
        },
        {
            "chunk_id": "doc1_art17_p3",
            "doc_id": "doc1",
            "dieu": "Điều 17",
            "part": 3,
            "n_parts": 3,
            "content": "Khoản 3. Biện pháp khắc phục hậu quả.",
            "name": "Nghị định 90"
        }
    ]
    stitcher = ArticleStitcher(chunks)
    stitched = stitcher.get_full_article("doc1", "Điều 17")
    assert stitched is not None
    # Verify ordered assembly: part 1 -> part 2 -> part 3
    expected_full = "Khoản 1. Phạt tiền 4.000.000 đồng.\nKhoản 2. Phạt tiền từ 6.000.000 đến 8.000.000 đồng.\nKhoản 3. Biện pháp khắc phục hậu quả."
    assert stitched["content"] == expected_full
