"""Task: legal-reference arm regressions (inherited from v10 lexref fix)."""

from __future__ import annotations

from src.common.legal_reference import (
    build_legal_reference_index,
    normalize_article_key,
    parse_doc_key,
    search_legal_references,
)


def test_parse_doc_key_slug_forms():
    assert parse_doc_key("100/2019/NĐ-CP") == "100/2019"
    assert parse_doc_key("Nghi-dinh-100-2019-ND-CP-16923") == "100/2019"
    assert parse_doc_key("Thong-tu-12-2019-TT-BTP-che-do-bao-cao-432844") == "12/2019"
    assert parse_doc_key("Quyet-dinh-405-QD-BNV-2021") == "405/2021"
    assert parse_doc_key("Bo-luat-Dan-su-1995-44-L-CTN") == "44/1995"
    assert parse_doc_key("Bo-luat-Dan-su-1995") is None  # year alone is not a doc key
    assert parse_doc_key("") is None
    assert parse_doc_key(None) is None
    assert parse_doc_key("No numbers here at all") is None


def test_digit_anchored_match_would_miss_slug():
    import re

    slug = "Nghi-dinh-100-2019-ND-CP-16923"
    # The historical bug: re.match anchors at position 0 on a letter-led slug.
    assert re.match(r"(\d{1,4})\s*[/\-]\s*(\d{2,4})", slug.strip()) is None
    # The fix parses it.
    assert parse_doc_key(slug) == "100/2019"


def test_normalize_article_key_rejects_preambles():
    assert normalize_article_key("17") == "17"
    assert normalize_article_key("Điều 17") == "17"
    assert normalize_article_key("preamble_0") is None
    assert normalize_article_key("") is None
    assert normalize_article_key(None) is None


def test_index_keeps_duplicate_chunk_positions_and_reports_empty():
    rows = [
        {"chunk_id": "c", "doc_name": "Nghi-dinh-100-2019-ND-CP", "article_number": "17", "text_raw": "first"},
        {"chunk_id": "c", "doc_name": "Nghi-dinh-100-2019-ND-CP", "article_number": "17", "text_raw": "second"},
        {"chunk_id": "x", "doc_name": "No numbers here at all", "article_number": "3", "text_raw": "other"},
    ]
    index, report = build_legal_reference_index(rows)
    assert report["is_empty"] is False
    assert report["rows_with_doc_key"] == 2
    assert index["doc"]["100/2019"] == [0, 1]
    assert index["doc_article"][("100/2019", "17")] == [0, 1]

    empty_index, empty_report = build_legal_reference_index(
        [{"chunk_id": "x", "doc_name": "No numbers", "article_number": "3", "text_raw": "t"}]
    )
    assert empty_report["is_empty"] is True
    assert search_legal_references(["Nghị định 100/2019 phạt bao nhiêu?"], empty_index, rows) == [[]]


def test_search_prefers_article_then_doc():
    rows = [
        {"chunk_id": "a17", "doc_name": "Nghi-dinh-100-2019-ND-CP", "article_number": "17", "text_raw": "E17"},
        {"chunk_id": "a18", "doc_name": "Nghi-dinh-100-2019-ND-CP", "article_number": "18", "text_raw": "E18"},
        {"chunk_id": "b", "doc_name": "Thong-tu-12-2019-TT-BTP", "article_number": "5", "text_raw": "other"},
    ]
    index, _ = build_legal_reference_index(rows)
    (hits,) = search_legal_references(["Theo Nghị định 100/2019, Điều 17 phạt bao nhiêu?"], index, rows, k=10)
    assert [h["chunk_id"] for h in hits] == ["a17", "a18"]
    (bare,) = search_legal_references(["Mức phạt là gì?"], index, rows)
    assert bare == []
