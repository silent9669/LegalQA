from src.task2.qa_memory import QAMemory


def test_qa_memory_exact_lookup():
    data = [
        {"id": "1", "question": "Hành vi trốn thuế bị phạt bao nhiêu?", "answer": "Phạt tiền từ 1 đến 3 lần số tiền trốn thuế."},
        {"id": "2", "question": "Nghị định 90/2017 có hiệu lực khi nào?", "answer": "Có hiệu lực từ ngày 15/09/2017."}
    ]
    mem = QAMemory.from_records(data)

    # Lookup by ID
    assert mem.lookup_exact("1", "random question") == "Phạt tiền từ 1 đến 3 lần số tiền trốn thuế."

    # Lookup by normalized question
    assert mem.lookup_exact("999", "  hành vi  trốn thuế bị phạt bao nhiêu ? ") == "Phạt tiền từ 1 đến 3 lần số tiền trốn thuế."

    # Unhit query
    assert mem.lookup_exact("999", "Câu hỏi chưa từng xuất hiện") is None


def test_qa_memory_conflicts_excluded():
    data = [
        {"id": "1", "question": "Điều 10 quy định gì?", "answer": "Quy định A"},
        {"id": "2", "question": "Điều 10 quy định gì?", "answer": "Quy định B"},
    ]
    mem = QAMemory.from_records(data)
    # Question is ambiguous/conflicting across records -> should not return a single ungrounded answer by question
    assert mem.lookup_exact("999", "Điều 10 quy định gì?") is None
    # But ID lookup still works
    assert mem.lookup_exact("1", "") == "Quy định A"
    assert mem.lookup_exact("2", "") == "Quy định B"


def test_qa_memory_similar_lookup():
    data = [
        {"id": "1", "question": "Theo Nghị định 90/2017/NĐ-CP hành vi không tiêm phòng phạt bao nhiêu?", "answer": "Phạt tiền từ 1.000.000 đến 2.000.000 đồng."},
        {"id": "2", "question": "Quy định về thời hạn nộp thuế thu nhập cá nhân?", "answer": "Thời hạn là ngày 30 của tháng tiếp theo."}
    ]
    mem = QAMemory.from_records(data)

    # Near-duplicate with slight phrasing variation but identical legal signal
    query = "Nghị định 90/2017/NĐ-CP thì hành vi không tiêm phòng bị phạt bao nhiêu?"
    fuzzy_hit = mem.lookup_fuzzy(query, threshold=0.80)
    assert fuzzy_hit is not None
    assert fuzzy_hit["similarity"] >= 0.80
    assert fuzzy_hit["matched_qa_id"] == "1"
    assert "1.000.000" in fuzzy_hit["answer"]


def test_qa_memory_fold_isolation_zero_leakage():
    data = [
        {"id": "train_1", "question": "Hành vi trốn thuế phạt bao nhiêu?", "answer": "Đáp án train 1"},
        {"id": "val_1", "question": "Hành vi buôn lậu thuốc lá xử lý thế nào?", "answer": "Đáp án val buôn lậu"},
    ]
    mem = QAMemory.from_records(data)
    isolated_mem = mem.filter_fold(val_qa_ids={"val_1"}, val_questions={"Hành vi buôn lậu thuốc lá xử lý thế nào?"})

    # Validation record must not be retrievable
    assert isolated_mem.lookup_exact("val_1", "Hành vi buôn lậu thuốc lá xử lý thế nào?") is None
    fuzzy_hit = isolated_mem.lookup_fuzzy("Hành vi buôn lậu thuốc lá xử lý thế nào?", threshold=0.8)
    assert fuzzy_hit is None or fuzzy_hit["matched_qa_id"] != "val_1"

    # Train record remains retrievable
    assert isolated_mem.lookup_exact("train_1", "Hành vi trốn thuế phạt bao nhiêu?") == "Đáp án train 1"
