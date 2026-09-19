"""Task: query-rewrite regressions (acronym expansion + weighted RRF policy)."""

from __future__ import annotations

from src.common.query_rewrite import (
    expand_acronyms,
    has_legal_reference,
    rewrite_query_for_retrieval,
)


def test_expand_acronyms_appends_once_and_keeps_original():
    out = expand_acronyms("NLĐ có quyền gì về BHXH?")
    assert "NLĐ" in out and "người lao động" in out
    assert "BHXH" in out and "bảo hiểm xã hội" in out
    assert out.count("người lao động") == 1
    # Unknown tokens untouched.
    assert expand_acronyms("mức phạt là gì") == "mức phạt là gì"


def test_has_legal_reference_detection():
    assert has_legal_reference("Theo Điều 17 Nghị định 100/2019?")
    assert has_legal_reference("khoản 2 điểm a")
    assert not has_legal_reference("mức phạt là gì?")


def test_rewrite_is_identity_when_experiment_off():
    assert rewrite_query_for_retrieval("NLĐ?", use_acronyms=False) == "NLĐ?"
    assert "người lao động" in rewrite_query_for_retrieval("NLĐ?", use_acronyms=True)
