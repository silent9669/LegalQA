from src.task2.source_snap import (
    apply_strategy_f,
    generate_candidate_ensemble,
    select_best_answer_candidate,
    snap_facts_to_evidence,
)


def test_snap_facts_to_evidence_date_and_clean():
    evidence = "Nghị định 13/2023/NĐ-CP có hiệu lực thi hành từ ngày 01 tháng 7 năm 2023."
    generated = "Nghị định 13/2023 có hiệu lực từ 1/7/2023."
    snapped = snap_facts_to_evidence(generated, evidence)
    assert "ngày 01 tháng 7 năm 2023" in snapped
    assert "ngày ngày" not in snapped


def test_apply_strategy_f_lengths():
    ans = "Căn cứ Điều 17 Nghị định 90/2017/NĐ-CP."
    evidence = "[DOCUMENT] Nghị định 90\n[ARTICLE] Điều 17. Phạt tiền từ 1.000.000 đến 2.000.000 đồng đối với hành vi không tiêm vắc xin phòng bệnh."
    extended_300 = apply_strategy_f(ans, evidence, max_chars=300)
    assert "Trích dẫn quy định:" in extended_300
    assert len(extended_300) <= len(ans) + 350
    assert "[DOCUMENT]" not in extended_300


def test_generate_candidate_ensemble():
    candidates = generate_candidate_ensemble(
        gen_ans="Căn cứ Điều 17 Nghị định 90.",
        evidence="Điều 17. Phạt 1 đến 2 triệu đồng.",
        exact_ans="",
        fuzzy_ans="Căn cứ Điều 17 Nghị định 90, phạt tiền từ 1.000.000 đến 2.000.000 đồng.",
        doc_name="Nghị định 90",
        art_num="17",
    )
    assert "generated" in candidates
    assert "snapped" in candidates
    assert "strategy_f_300" in candidates
    assert "strategy_f_1000" in candidates
    assert "stitched_extract" in candidates


def test_select_best_answer_candidate_exact_and_fuzzy():
    # 1. Exact memory priority
    cand1 = {"exact_memory": "Exact answer from training data", "generated": "Gen answer"}
    assert select_best_answer_candidate(cand1) == "Exact answer from training data"

    # 2. High confidence fuzzy memory
    cand2 = {"fuzzy_memory": "Fuzzy answer", "generated": "Gen answer"}
    assert select_best_answer_candidate(cand2, features={"is_direct_reuse": True}) == "Fuzzy answer"

    # 3. Snapped answer with Strategy F
    cand3 = {
        "snapped": "Căn cứ Điều 17 Nghị định 90/2017/NĐ-CP, phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng.",
        "stitched_extract": "Điều 17. Xử phạt không tiêm phòng."
    }
    selected = select_best_answer_candidate(cand3)
    assert "Căn cứ Điều 17" in selected
