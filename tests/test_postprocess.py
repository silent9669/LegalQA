import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.postprocess.extractive import generate_extractive_answer, extract_penalty_amount, extract_conclusion_sentence
from src.postprocess.source_snap import source_snap_answer

def test_extract_penalty_amount():
    text = "Phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng đối với hành vi vi phạm quy định."
    amt = extract_penalty_amount(text)
    assert amt == "từ 6.000.000 đồng đến 8.000.000 đồng"

def test_extractive_answer_3part_structure():
    evidence = [{
        "name": "Nghị định 90/2017/NĐ-CP",
        "dieu": "Điều 17. Vi phạm quy định chung về kiểm dịch động vật",
        "khoan": "3",
        "content": "3. Phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng đối với hành vi không có Giấy chứng nhận kiểm dịch động vật."
    }]
    ans = generate_extractive_answer("Vận chuyển động vật không có kiểm dịch bị phạt thế nào?", evidence)
    assert "Căn cứ" in ans
    assert "Nghị định 90/2017/NĐ-CP" in ans
    assert "Điều 17." in ans
    assert "Theo đó" in ans
    assert "6.000.000 đồng đến 8.000.000 đồng" in ans

def test_source_snap_amounts():
    gen = "Mức phạt là từ 6 triệu đến 8 triệu đồng."
    evidence = [{"content": "phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng đối với người điều khiển"}]
    snapped = source_snap_answer(gen, evidence)
    assert "6.000.000 đồng đến 8.000.000 đồng" in snapped
