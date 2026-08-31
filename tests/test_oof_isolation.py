import pandas as pd
import pytest
from scripts.prepare_data import assign_near_duplicate_grouped_folds
from src.task2.qa_memory import QAMemory


def test_near_duplicate_fold_grouping_invariants():
    # Synthetic dataset with exact duplicates, near duplicates, and distinct queries
    records = [
        {"id": "q1", "question_raw": "Thời hạn nộp hồ sơ khai thuế thu nhập cá nhân là bao lâu?", "answer_raw": "A1"},
        {"id": "q2", "question_raw": "Cho hỏi thời hạn nộp hồ sơ khai thuế thu nhập cá nhân là bao lâu?", "answer_raw": "A1"},
        {"id": "q3", "question_raw": "Mức phạt đối với hành vi trốn thuế theo Nghị định 125/2020/NĐ-CP?", "answer_raw": "A2"},
        {"id": "q4", "question_raw": "Nghị định 125/2020/NĐ-CP quy định mức phạt hành vi trốn thuế như thế nào?", "answer_raw": "A2"},
        {"id": "q5", "question_raw": "Điều kiện thành lập doanh nghiệp tư nhân theo Luật Doanh nghiệp 2020?", "answer_raw": "A3"},
        {"id": "q6", "question_raw": "Quy định về thời hiệu xử phạt vi phạm hành chính?", "answer_raw": "A4"},
        {"id": "q7", "question_raw": "Thời hiệu xử phạt vi phạm hành chính là mấy năm?", "answer_raw": "A4"},
        {"id": "q8", "question_raw": "Thủ tục đăng ký kết hôn có yếu tố nước ngoài?", "answer_raw": "A5"},
        {"id": "q9", "question_raw": "Độ tuổi nghỉ hưu của người lao động năm 2024?", "answer_raw": "A6"},
        {"id": "q10", "question_raw": "Quy định về thử việc theo Bộ luật Lao động 2019?", "answer_raw": "A7"},
    ]
    df = pd.DataFrame(records)
    from src.common.normalize import normalize_question
    df["question_norm"] = df["question_raw"].apply(normalize_question)
    df["qa_id"] = df["id"]

    df_grouped = assign_near_duplicate_grouped_folds(df, num_folds=3, seed=42)

    assert "fold_id" in df_grouped.columns
    # Check that q1 and q2 are in the same fold
    f1 = df_grouped.loc[df_grouped["id"] == "q1", "fold_id"].values[0]
    f2 = df_grouped.loc[df_grouped["id"] == "q2", "fold_id"].values[0]
    assert f1 == f2, f"Near duplicates q1 and q2 must have same fold_id, got {f1} vs {f2}"

    # Check that q3 and q4 (sharing doc 125/2020) are in the same fold
    f3 = df_grouped.loc[df_grouped["id"] == "q3", "fold_id"].values[0]
    f4 = df_grouped.loc[df_grouped["id"] == "q4", "fold_id"].values[0]
    assert f3 == f4, f"Near duplicates q3 and q4 must have same fold_id, got {f3} vs {f4}"


def test_sampled_fold_memory_isolation():
    # Verify that isolating an entire fold from memory removes ALL its members, even if evaluating only 1 sample
    records = [
        {"id": "1", "question_raw": "Câu hỏi A1", "answer_raw": "Đáp án A1", "source_split": "train"},
        {"id": "2", "question_raw": "Câu hỏi A2", "answer_raw": "Đáp án A2", "source_split": "train"},
        {"id": "3", "question_raw": "Câu hỏi B1", "answer_raw": "Đáp án B1", "source_split": "train"},
        {"id": "4", "question_raw": "Câu hỏi B2", "answer_raw": "Đáp án B2", "source_split": "train"},
    ]
    memory = QAMemory.from_records(records)

    # Assign fold 0 to {1, 2} and fold 1 to {3, 4}
    fold_0_ids = {"1", "2"}
    fold_0_questions = {"Câu hỏi A1", "Câu hỏi A2"}

    # Filter fold 0 strictly
    isolated_mem = memory.filter_fold(val_qa_ids=fold_0_ids, val_questions=fold_0_questions)

    # Verify that NONE of fold 0 items can be looked up in isolated memory
    assert isolated_mem.lookup_exact("1", "Câu hỏi A1") is None
    assert isolated_mem.lookup_exact("2", "Câu hỏi A2") is None
    assert "1" not in isolated_mem.id_to_answer
    assert "2" not in isolated_mem.id_to_answer

    # Fold 1 items should still be accessible
    assert isolated_mem.lookup_exact("3", "Câu hỏi B1") == "Đáp án B1"
    assert isolated_mem.lookup_exact("4", "Câu hỏi B2") == "Đáp án B2"
