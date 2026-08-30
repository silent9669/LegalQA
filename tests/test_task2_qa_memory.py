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
