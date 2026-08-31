import os
from pathlib import Path
import pandas as pd
import pytest
from src.common.legal_parser import parse_legal_document
from src.common.normalize import (
    extract_canonical_doc_keys,
    extract_legal_signals,
    normalize_legal_number,
    normalize_question,
    remove_accents,
    slugify_legal_title,
)
from src.common.evidence import (
    build_corpus_doc_index,
    mine_hard_negatives,
    parse_citations_from_answer,
    resolve_citations_to_chunks,
)
from src.task2.qa_memory import QAMemory


def test_legal_normalization_and_suffix_preservation():
    # Verify suffixes like QH14, NĐ-CP, TT-BTP are preserved
    signals = extract_legal_signals("Căn cứ theo Luật số 42/2017/QH14 và Nghị định 90/2017/NĐ-CP Điều 17 khoản 3")
    assert "42/2017/QH14" in signals["doc_numbers"]
    assert "90/2017/NĐ-CP" in signals["doc_numbers"]
    assert "17" in signals["articles"]
    assert "3" in signals["clauses"]

    # Canonical legal number normalization
    assert normalize_legal_number("02/2021/tt-btp") == "2/2021/TT-BTP"
    assert normalize_legal_number("90/2017/nd-cp") == "90/2017/NĐ-CP"

    # Title slugification and accent stripping
    assert slugify_legal_title("Bộ luật Tố tụng Dân sự 2015-432844") == "bo-luat-to-tung-dan-su-2015"
    assert remove_accents("Đất đai") == "Dat dai"

    # Canonical doc keys extraction
    keys = extract_canonical_doc_keys("Thong-tu-12-2019-TT-BTP-che-do-bao-cao-dinh-ky-432844")
    assert "12/2019/tt-btp" in keys
    assert "12/2019" in keys


def test_hierarchical_legal_parser_and_offsets():
    passage = (
        "Chương I\nQUY ĐỊNH CHUNG\n\n"
        "Điều 1. Phạm vi điều chỉnh\n"
        "1. Thông tư này quy định chế độ báo cáo định kỳ.\n"
        "2. Áp dụng cho mọi cơ quan, tổ chức.\n\n"
        "Điều 2. Đối tượng áp dụng\n"
        "Áp dụng cho toàn bộ cán bộ công chức."
    )
    chunks = parse_legal_document(doc_id="101", doc_name="Thong-tu-12-2019-TT-BTP", passage=passage)
    assert len(chunks) == 3

    # Check clause 1 chunk
    c1 = chunks[0]
    assert c1["chunk_id"] == "doc101_art1_p1"
    assert c1["article_number"] == "1"
    assert c1["clause_number"] == "1"
    assert c1["chapter_number"] == "I"
    assert c1["legal_number"] == "12/2019/TT-BTP"
    assert c1["year"] == "2019"
    assert c1["start_char"] >= 0
    assert c1["end_char"] > c1["start_char"]

    # Verify start_char and end_char accurately bound the text inside passage
    clause_span = passage[c1["start_char"]:c1["end_char"]]
    assert "1. Thông tư này" in clause_span

    # Check full article chunk (Điều 2 has no numbered clauses)
    c3 = chunks[2]
    assert c3["chunk_id"] == "doc101_art2_full"
    assert c3["article_number"] == "2"
    assert c3["clause_number"] is None


def test_citation_parsing_and_resolution():
    answer = "Căn cứ theo quy định tại khoản 2 Điều 1 Thông tư 12/2019/TT-BTP thì áp dụng cho cơ quan."
    citations = parse_citations_from_answer(answer)
    assert len(citations) >= 1
    assert citations[0]["article"] == "1"
    assert citations[0]["clause"] == "2"

    # Create dummy corpus chunks
    passage = "Điều 1. Quy định\n1. Khoản một.\n2. Khoản hai."
    chunks = parse_legal_document("202", "Thong-tu-12-2019-TT-BTP", passage)
    df_chunks = pd.DataFrame(chunks)

    # Resolve citations
    resolved = resolve_citations_to_chunks(citations, df_chunks)
    assert len(resolved) == 1
    assert resolved[0]["positive_chunk_id"] == "doc202_art1_p2"
    assert resolved[0]["positive_article_id"] == "doc202_art1"


def test_hard_negative_mining_and_false_negative_guard():
    passage = (
        "Điều 1. Quy định 1\n1. Khoản 1.1\n2. Khoản 1.2\n\n"
        "Điều 2. Quy định 2\n1. Khoản 2.1"
    )
    chunks = parse_legal_document("303", "Nghi-dinh-90-2017-ND-CP", passage)
    chunks_other = parse_legal_document("404", "Luat-Dat-dai-2013", "Điều 54. Đất đai\nNội dung đất đai.")
    df_chunks = pd.DataFrame(chunks + chunks_other)

    pos_chunk_id = "doc303_art1_p1"
    pos_art_id = "doc303_art1"
    pos_doc = "Nghi-dinh-90-2017-ND-CP"

    negs = mine_hard_negatives(pos_chunk_id, pos_art_id, pos_doc, df_chunks, max_per_type=5)

    # Type A: Same article, wrong clause
    assert "doc303_art1_p2" in negs["same_article_wrong_clause"]
    assert pos_chunk_id not in negs["same_article_wrong_clause"]

    # Type B: Same doc, wrong article
    assert "doc303_art2_p1" in negs["same_doc_wrong_article"]

    # Type C: Different doc
    assert "doc404_art54_full" in negs["different_doc"]


def test_qa_memory_conflict_handling_and_fold_isolation():
    records = [
        {"id": "q1", "question": "Học sinh vay vốn được bao nhiêu?", "answer": "Tối đa 4 triệu đồng/tháng.", "source_split": "train"},
        {"id": "q2", "question": "Học sinh vay vốn được bao nhiêu?", "answer": "Tối đa 4 triệu đồng/tháng.", "source_split": "train"},
        # Conflicting answers for identical normalized question
        {"id": "q3", "question": "Thời hạn án treo là bao lâu?", "answer": "Từ 1 đến 5 năm.", "source_split": "train"},
        {"id": "q4", "question": "Thời hạn án treo là bao lâu?", "answer": "Từ 6 tháng đến 3 năm.", "source_split": "train"},
        # Unique question
        {"id": "q5", "question": "Đất không thu tiền sử dụng đất?", "answer": "Theo Điều 54 Luật Đất đai 2013.", "source_split": "train"},
    ]

    mem = QAMemory.from_records(records)

    # Exact ID lookup works with matching question or empty question
    assert mem.lookup_exact("q1", "Học sinh vay vốn được bao nhiêu?") == "Tối đa 4 triệu đồng/tháng."
    assert mem.lookup_exact("q1", "") == "Tối đa 4 triệu đồng/tháng."
    assert mem.lookup_exact("q3", "Thời hạn án treo là bao lâu?") == "Từ 1 đến 5 năm."
    assert mem.lookup_exact("q4", "Thời hạn án treo là bao lâu?") == "Từ 6 tháng đến 3 năm."

    # ID collision with conflicting question returns None for safety
    assert mem.lookup_exact("q1", "something else") is None

    # Normalized question lookup works for consistent duplicates
    assert mem.lookup_exact(None, "học sinh vay vốn được bao nhiêu?") == "Tối đa 4 triệu đồng/tháng."

    # Normalized question lookup returns None for conflicting duplicate questions!
    assert mem.lookup_exact(None, "thời hạn án treo là bao lâu?") is None

    # Fold filtering test
    filtered_mem = mem.filter_fold(val_qa_ids={"q1", "q2"}, val_questions={"học sinh vay vốn được bao nhiêu?"})
    assert filtered_mem.lookup_exact("q1", "học sinh vay vốn được bao nhiêu?") is None
    assert filtered_mem.lookup_exact(None, "học sinh vay vốn được bao nhiêu?") is None
    assert filtered_mem.lookup_exact("q5", "đất không thu tiền sử dụng đất?") == "Theo Điều 54 Luật Đất đai 2013."


def test_package_kaggle_dataset_with_indexes(tmp_path: Path):
    from scripts.package_kaggle_dataset import package_kaggle_dataset

    # Setup dummy source structure
    src_dir = tmp_path / "src_artifacts"
    data_dir = src_dir / "data"
    data_dir.mkdir(parents=True)

    for name in ["legal_chunks.parquet", "qa_unique.parquet", "known_qa.json", "qa_citations.parquet", "retrieval_labels.parquet", "fold_assignments.parquet"]:
        (data_dir / name).write_text("test_content", encoding="utf-8")

    # Create dummy indexes
    bm25_dir = src_dir / "indexes" / "bm25"
    bm25_dir.mkdir(parents=True)
    (bm25_dir / "bm25_manifest.json").write_text("{}", encoding="utf-8")

    dek21_dir = src_dir / "indexes" / "dek21"
    dek21_dir.mkdir(parents=True)
    (dek21_dir / "dek21_manifest.json").write_text("{}", encoding="utf-8")

    stage_dir = tmp_path / "staged"

    package_kaggle_dataset(
        source_dir=str(src_dir),
        staging_dir=str(stage_dir),
        dataset_title="TestLegalQA",
        dry_run=False,
    )

    # Check staged files
    assert (stage_dir / "legal_chunks.parquet").exists()
    assert (stage_dir / "dataset_manifest.json").exists()
    assert (stage_dir / "indexes" / "bm25" / "bm25_manifest.json").exists()
    assert (stage_dir / "indexes" / "dek21" / "dek21_manifest.json").exists()

