"""Unit tests verifying dual-part statutory answer assembly (prose + citation block)."""

from __future__ import annotations

import pytest

from src.task2.candidates import clean_model_prose, generate_candidate_ensemble
from src.task2.production_config import GENERATOR_DEPENDENT_CANDIDATES
from src.task2.selector import CANDIDATE_ORDER


def test_clean_model_prose_strips_preamble_and_markers():
    raw_1 = "Dựa trên các tài liệu được cung cấp, mức phạt từ 4.000.000 đến 5.000.000 đồng."
    assert clean_model_prose(raw_1) == "mức phạt từ 4.000.000 đến 5.000.000 đồng."

    raw_2 = "Căn cứ vào văn bản được cung cấp trên, người vi phạm bị xử phạt..."
    assert clean_model_prose(raw_2) == "người vi phạm bị xử phạt..."

    raw_3 = "```markdown\nTrả lời: Căn cứ quy định pháp luật.\n```<|im_end|>"
    assert clean_model_prose(raw_3) == "Căn cứ quy định pháp luật."


def test_dual_assembled_candidate_structure():
    gen_ans = "Theo quy định tại Điều 17 Nghị định 90/2017/NĐ-CP, người vận chuyển động vật bị phạt từ 6 đến 8 triệu đồng."
    ev_packs = {
        "full_article": (
            "Điều 17. Vi phạm quy định về kiểm dịch động vật\n"
            "1. Phạt tiền từ 4.000.000 đồng đến 5.000.000 đồng...\n"
            "3. Phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng đối với hành vi không có Giấy chứng nhận..."
        )
    }
    cands = generate_candidate_ensemble(
        gen_ans=gen_ans,
        evidence=ev_packs["full_article"],
        doc_name="Nghị định 90/2017/NĐ-CP",
        art_num="17",
        evidence_packs=ev_packs,
    )

    assert "dual_assembled" in cands
    ans = cands["dual_assembled"]
    assert "Trích dẫn quy định:" in ans
    assert "Căn cứ Điều 17 Nghị định 90/2017/NĐ-CP quy định như sau:" in ans
    assert "Phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng" in ans
    assert ans.startswith(gen_ans)


def test_dual_assembled_registered_in_production_config_and_selector():
    assert "dual_assembled" in GENERATOR_DEPENDENT_CANDIDATES
    assert "dual_assembled" in CANDIDATE_ORDER
