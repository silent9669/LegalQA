from src.task2.candidates import (
    clean_statutory_text,
    build_citation_header,
    snap_facts_to_evidence,
    apply_strategy_f,
    generate_candidate_ensemble,
)


def test_clean_statutory_text():
    raw = "[DOCUMENT] Nghị định 90/2017/NĐ-CP\n[ARTICLE] Điều 17. Vi phạm\n(Hình từ Internet)\nNội dung chính."
    cleaned = clean_statutory_text(raw)
    assert "[DOCUMENT]" not in cleaned
    assert "[ARTICLE]" not in cleaned
    assert "Hình từ Internet" not in cleaned
    assert "Nội dung chính." in cleaned


def test_build_citation_header():
    header = build_citation_header("Nghị định 90/2017/NĐ-CP", "17", "3")
    assert "khoản 3" in header
    assert "Điều 17" in header
    assert "Nghị định 90/2017/NĐ-CP" in header
    assert header.startswith("Căn cứ")


def test_snap_facts_to_evidence():
    evidence = "Quy định áp dụng từ ngày 15 tháng 09 năm 2017 đối với số tiền 2.000.000 đồng."
    gen_ans = "Quy định có hiệu lực từ 15/09/2017."
    snapped = snap_facts_to_evidence(gen_ans, evidence)
    assert "ngày 15 tháng 09 năm 2017" in snapped


def test_generate_candidate_ensemble():
    cands = generate_candidate_ensemble(
        gen_ans="Câu trả lời tóm tắt.",
        evidence="[ARTICLE] Điều 1. Nội dung điều luật chi tiết.",
        exact_ans="",
        fuzzy_ans="Đáp án tương tự",
        doc_name="Luật 01",
        art_num="1",
        clause_num="1",
    )
    assert "focused_extract" in cands
    assert "stitched_extract" in cands
    assert "generated" in cands
    assert "strategy_f_300" in cands
    assert "strategy_f_1000" in cands
    assert "fuzzy_memory" in cands
