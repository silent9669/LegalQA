from src.common.evidence import parse_citations_from_answer, mine_hard_negatives

def test_parse_citations_from_answer():
    ans = "Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP quy định mức phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng."
    citations = parse_citations_from_answer(ans)
    assert len(citations) >= 1
    assert "90/2017/NĐ-CP" in citations[0]["doc_number"]
    assert citations[0]["article"] == "17"
    assert citations[0]["clause"] == "3"

def test_mine_hard_negatives():
    chunks = [
        {"chunk_id": "doc1_art17_p3", "doc_id": "1", "parent_article_id": "doc1_art17", "clause_number": "3"},
        {"chunk_id": "doc1_art17_p1", "doc_id": "1", "parent_article_id": "doc1_art17", "clause_number": "1"},
        {"chunk_id": "doc1_art16_p1", "doc_id": "1", "parent_article_id": "doc1_art16", "clause_number": "1"},
        {"chunk_id": "doc2_art5_p1", "doc_id": "2", "parent_article_id": "doc2_art5", "clause_number": "1"},
    ]
    negatives = mine_hard_negatives(
        query_info={"doc_id": "1"},
        all_chunks=chunks,
        positive_chunk_id="doc1_art17_p3",
        positive_article_id="doc1_art17"
    )
    assert "doc1_art17_p1" in negatives["same_article_wrong_clause"]
    assert "doc1_art16_p1" in negatives["same_doc_wrong_article"]
