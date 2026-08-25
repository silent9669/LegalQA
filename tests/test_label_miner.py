import pytest
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.label_miner import parse_legal_citations, mine_training_labels

def test_parse_citations():
    text = "Về hiệu lực thi hành, căn cứ khoản 1 Điều 43 Nghị định 13/2023/NĐ-CP như sau:"
    citations = parse_legal_citations(text)
    assert len(citations) >= 1
    c = citations[0]
    assert c["document_number"] == "13/2023/NĐ-CP"
    assert c["article"] == "43"
    assert c["clause"] == "1"

def test_mine_training_labels():
    df_qa = pd.DataFrame([
        {"id": "1", "question": "Hỏi về NĐ 13?", "answer": "Theo Điều 43 Nghị định 13/2023/NĐ-CP"}
    ])
    df_chunks = pd.DataFrame([
        {"chunk_id": "c1", "doc_id": "10", "name": "Nghị định 13/2023/NĐ-CP", "dieu": "Điều 43", "content": "Nội dung 43"}
    ])
    df_labels = mine_training_labels(df_qa, df_chunks)
    assert len(df_labels) == 1
    assert "citations" in df_labels.columns
