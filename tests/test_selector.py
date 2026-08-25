import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.selector.candidate_selector import CandidateSelector

def test_candidate_selector_rules():
    selector = CandidateSelector()
    candidates = {
        "candidate_generate": "Câu trả lời do mô hình sinh",
        "candidate_extract": "Căn cứ Điều 10 mức phạt là 5.000.000 đồng",
        "candidate_snap": "Câu trả lời do mô hình sinh đã snap"
    }
    chosen = selector.select(
        question="Mức phạt tiền là bao nhiêu?",
        candidates=candidates,
        features={"has_penalty_keyword": True, "extractive_quality": 0.9}
    )
    assert chosen in [candidates["candidate_extract"], candidates["candidate_snap"]]

def test_candidate_selector_fallback():
    selector = CandidateSelector()
    candidates = {
        "candidate_generate": "Câu trả lời duy nhất"
    }
    chosen = selector.select(
        question="Bị cáo có quyền gì?",
        candidates=candidates,
        features={}
    )
    assert chosen == "Câu trả lời duy nhất"
