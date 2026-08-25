import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.canonical import normalize_vietnamese_text
from src.memory.exact_memory import ExactMemory

def test_normalize_vietnamese_text():
    raw = "  Nghị   định  13/2023/NĐ-CP   về dữ   liệu ? "
    norm = normalize_vietnamese_text(raw)
    assert norm == "Nghị định 13/2023/NĐ-CP về dữ liệu ?"

def test_exact_memory_lookup():
    mem_dict = {
        "by_id": {"23207": "Gold Answer 23207"},
        "by_question": {"câu hỏi kiểm tra": "Gold Answer Question"}
    }
    mem = ExactMemory(mem_dict)
    assert mem.lookup("23207", "bất kỳ câu hỏi") == "Gold Answer 23207"
    assert mem.lookup("99999", "Câu hỏi   kiểm tra") == "Gold Answer Question"
    assert mem.lookup("99999", "câu hỏi chưa từng thấy") is None
