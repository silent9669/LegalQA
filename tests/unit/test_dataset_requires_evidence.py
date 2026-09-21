import pandas as pd
from src.task2.generation.dataset import build_grounded_training_examples


def test_examples_without_evidence_are_dropped(tmp_path):
    qa = pd.DataFrame([
        {"qa_id": "a", "question_raw": "Q1?", "answer_raw": "A1 statutory text", "fold_id": 0},
        {"qa_id": "b", "question_raw": "Q2?", "answer_raw": "A2 statutory text", "fold_id": 0},
        {"qa_id": "a", "question_raw": "Q1?", "answer_raw": "A1 statutory text", "fold_id": 0},  # duplicate
    ])
    labels = pd.DataFrame([{"qa_id": "a", "positive_chunk_id": "c1"}])
    chunks = pd.DataFrame([{"chunk_id": "c1", "text_raw": "Điều 1. Nội dung."}])
    qa_p, l_p, c_p = (tmp_path / n for n in ("qa.parquet", "l.parquet", "c.parquet"))
    qa.to_parquet(qa_p); labels.to_parquet(l_p); chunks.to_parquet(c_p)

    ex, diag = build_grounded_training_examples(
        qa_path=str(qa_p),
        labels_path=str(l_p),
        chunks_path=str(c_p),
        require_evidence=True,
        return_diagnostics=True,
    )
    assert [e["qa_id"] for e in ex] == ["a"]  # "b" dropped (no evidence), duplicate "a" deduplicated
    assert diag.get("dropped_no_evidence") == 1
    assert diag.get("duplicate_qa_dropped", 0) == 1
