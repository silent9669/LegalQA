import pytest
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.canonical import build_canonical_qa

def test_build_canonical_qa(tmp_path):
    train_file = tmp_path / "train.json"
    warmup_file = tmp_path / "warmup.json"

    train_data = {
        "1": {"question": "Câu hỏi 1", "answer": "Đáp án 1"},
        "2": {"question": "Câu hỏi 2", "answer": "Đáp án 2"}
    }
    warmup_data = {
        "2": {"question": "Câu hỏi 2", "answer": "Đáp án 2"},
        "3": {"question": "Câu hỏi 3", "answer": "Đáp án 3"}
    }

    train_file.write_text(json.dumps(train_data), encoding="utf-8")
    warmup_file.write_text(json.dumps(warmup_data), encoding="utf-8")

    df_unique, mem_dict = build_canonical_qa(str(train_file), str(warmup_file))

    assert len(df_unique) == 3
    assert "1" in mem_dict["by_id"]
    assert "2" in mem_dict["by_id"]
    assert "3" in mem_dict["by_id"]

    row2 = df_unique[df_unique["id"] == "2"].iloc[0]
    assert set(row2["source_splits"]) == {"train", "warmup"}
