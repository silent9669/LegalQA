from src.task2.source_snap import snap_facts_to_evidence, select_best_answer_candidate

def test_snap_facts_to_evidence():
    evidence = "Nghị định 13/2023/NĐ-CP có hiệu lực thi hành từ ngày 01 tháng 7 năm 2023."
    generated = "Nghị định 13/2023 có hiệu lực từ 1/7/2023."
    snapped = snap_facts_to_evidence(generated, evidence)
    assert "ngày 01 tháng 7 năm 2023" in snapped

def test_select_best_answer_candidate():
    candidates = {
        "focused_extract": "1. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng.",
        "stitched_extract": "Điều 17. Vi phạm quy định\n1. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng.\n2. Buộc tiêu hủy.",
        "generated": "Căn cứ Điều 17 Nghị định 90/2017/NĐ-CP, phạt từ 1 đến 2 triệu đồng.",
        "snapped": "Căn cứ Điều 17 Nghị định 90/2017/NĐ-CP, phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng."
    }
    ans = select_best_answer_candidate(candidates)
    assert "Căn cứ Điều 17" in ans
    assert "1.000.000" in ans
