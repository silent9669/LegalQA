import os
import pandas as pd
from src.task2.training.train_reranker import prepare_reranker_dataset


def test_prepare_reranker_dataset_isolation():
    records = [
        {"qa_id": "q1", "fold_id": 0, "question": "Q1", "positive_text": "Pos 1", "negative_text": "Neg 1", "negative_type": "same_doc"},
        {"qa_id": "q2", "fold_id": 1, "question": "Q2", "positive_text": "Pos 2", "negative_text": "Neg 2", "negative_type": "cross_doc"},
        {"qa_id": "q3", "fold_id": 2, "question": "Q3", "positive_text": "Pos 3", "negative_text": "Neg 3", "negative_type": "same_doc"},
    ]
    df = pd.DataFrame(records)

    # Exclude fold 0
    train_ds, val_ds = prepare_reranker_dataset(df_pairs=df, val_fold=0)

    # Train dataset must not contain fold 0
    train_qids = {ex["qa_id"] for ex in train_ds}
    assert "q1" not in train_qids
    assert "q2" in train_qids
    assert "q3" in train_qids

    if val_ds:
        val_qids = {ex["qa_id"] for ex in val_ds}
        assert "q1" in val_qids
        assert "q2" not in val_qids


def test_prepare_reranker_dataset_subset_limits():
    records = [
        {"qa_id": f"q{i}", "fold_id": i % 2, "question": f"Q{i}", "positive_text": f"Pos {i}", "negative_text": f"Neg {i}"}
        for i in range(20)
    ]
    df = pd.DataFrame(records)

    train_ds, val_ds = prepare_reranker_dataset(
        df_pairs=df,
        val_fold=0,
        max_train_pairs=4,
        max_val_pairs=3,
        seed=42,
    )

    assert len(train_ds) == 4
    assert len(val_ds) == 3
