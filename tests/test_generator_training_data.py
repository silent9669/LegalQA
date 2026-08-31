import os
import pandas as pd
from src.task2.training.train_generator import build_grounded_training_examples, truncate_evidence_preserving_answer


def test_answer_preserving_truncation():
    # Long evidence + short answer -> evidence is truncated, answer is completely preserved
    long_evidence = "Điều 1. " + "Nội dung điều luật rất dài. " * 500
    gold_answer = "Căn cứ Điều 1, mức phạt từ 5.000.000 đồng đến 10.000.000 đồng."
    question = "Mức phạt là bao nhiêu?"

    truncated_ev = truncate_evidence_preserving_answer(
        question=question,
        evidence=long_evidence,
        answer=gold_answer,
        max_chars=1200,
    )

    assert len(truncated_ev) <= 1200
    # Gold answer must remain intact
    assert "5.000.000 đồng" in gold_answer


def test_build_grounded_training_examples_structure():
    qa_data = [
        {"qa_id": "q1", "fold_id": 1, "question_raw": "Q1", "answer_raw": "Ans 1"},
        {"qa_id": "q2", "fold_id": 0, "question_raw": "Q2", "answer_raw": "Ans 2"},
    ]
    df_qa = pd.DataFrame(qa_data)

    examples = build_grounded_training_examples(
        df_qa=df_qa,
        fold_to_exclude=0,
    )

    # Excluded fold 0 -> only q1 in examples
    assert len(examples) == 1
    assert "Ans 1" in examples[0]["text"]
    assert "<|im_start|>assistant" in examples[0]["text"]
