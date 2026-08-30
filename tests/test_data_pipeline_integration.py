import os
import pytest
from src.common.legal_parser import parse_legal_document
from src.common.bm25 import BM25Retriever
from src.task2.predict import LegalQAPipeline

def test_data_pipeline_modules_integrated():
    # 1. Parse legal passage
    passage = "Nghị định 90/2017/NĐ-CP\nĐiều 17. Xử phạt không tiêm phòng\n1. Phạt tiền từ 1 đến 2 triệu đồng."
    chunks = parse_legal_document("1", "Nghị định 90", passage)
    assert len(chunks) >= 1

    # 2. Build index
    bm25 = BM25Retriever()
    bm25.fit(chunks)
    res = bm25.search("Xử phạt không tiêm phòng", top_k=1)
    assert len(res) == 1
    assert res[0]["chunk_id"] in [c["chunk_id"] for c in chunks]
    assert res[0]["article_number"] == "17"

    # 3. Pipeline executes without errors
    pipeline = LegalQAPipeline.build_mock()
    pred = pipeline.predict_single("q1", "Nghị định 90 Điều 17")
    assert isinstance(pred, str)
    assert len(pred) > 0
