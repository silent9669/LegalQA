import pytest
from src.common.normalize import clean_legal_text, extract_legal_signals, tokenize_vietnamese

def test_clean_legal_text():
    raw = "  Điều   17 .  Nghị  định   90/2017/NĐ-CP \n\n Khoản 3 “quy định” "
    cleaned = clean_legal_text(raw)
    assert cleaned == 'Điều 17 . Nghị định 90/2017/NĐ-CP Khoản 3 "quy định"'

def test_extract_legal_signals():
    query = "Theo khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP năm 2017 quy định gì?"
    signals = extract_legal_signals(query)
    assert "90/2017/NĐ-CP" in signals.get("doc_numbers", [])
    assert "17" in signals.get("articles", [])
    assert "3" in signals.get("clauses", [])
    assert "2017" in signals.get("years", [])

def test_tokenize_vietnamese():
    text = "Nghị định quy định xử phạt vi phạm hành chính"
    tokenized = tokenize_vietnamese(text)
    assert "_" in tokenized
