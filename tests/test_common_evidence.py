import pandas as pd
from src.common.evidence import (
    CorpusLookupIndex,
    parse_citations_from_answer,
    mine_hard_negatives,
    mine_retrieval_hard_negatives,
    resolve_citations_to_chunks,
)


def test_parse_citations_from_answer():
    ans = "Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP quy định mức phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng."
    citations = parse_citations_from_answer(ans)
    assert len(citations) >= 1
    assert "90/2017/NĐ-CP" in citations[0]["doc_number"]
    assert citations[0]["article"] == "17"
    assert citations[0]["clause"] == "3"


def test_mine_hard_negatives():
    chunks = [
        {"chunk_id": "doc1_art17_p3", "doc_id": "1", "doc_name": "Nghị định 90/2017", "parent_article_id": "doc1_art17", "clause_number": "3"},
        {"chunk_id": "doc1_art17_p1", "doc_id": "1", "doc_name": "Nghị định 90/2017", "parent_article_id": "doc1_art17", "clause_number": "1"},
        {"chunk_id": "doc1_art16_p1", "doc_id": "1", "doc_name": "Nghị định 90/2017", "parent_article_id": "doc1_art16", "clause_number": "1"},
        {"chunk_id": "doc2_art5_p1", "doc_id": "2", "doc_name": "Nghị định 100/2019", "parent_article_id": "doc2_art5", "clause_number": "1"},
    ]
    df = pd.DataFrame(chunks)
    lookup = CorpusLookupIndex(df)

    negatives = mine_hard_negatives(
        positive_chunk_id="doc1_art17_p3",
        positive_article_id="doc1_art17",
        positive_doc_name="Nghị định 90/2017",
        chunks_df=lookup,
    )
    assert "doc1_art17_p1" in negatives["same_article_wrong_clause"]
    assert "doc1_art16_p1" in negatives["same_doc_wrong_article"]


def test_mine_retrieval_hard_negatives_excludes_positives():
    chunks = [
        {"chunk_id": "pos_1", "doc_name": "Luật Doanh nghiệp", "parent_article_id": "art_10", "text_raw": "T1"},
        {"chunk_id": "neg_same_doc", "doc_name": "Luật Doanh nghiệp", "parent_article_id": "art_12", "text_raw": "T2"},
        {"chunk_id": "neg_cross_doc", "doc_name": "Luật Đầu tư", "parent_article_id": "art_5", "text_raw": "T3"},
    ]
    df = pd.DataFrame(chunks)
    lookup = CorpusLookupIndex(df)

    retrieved = [
        {"chunk_id": "pos_1", "doc_name": "Luật Doanh nghiệp", "parent_article_id": "art_10", "rank": 1, "score": 20.0},
        {"chunk_id": "neg_same_doc", "doc_name": "Luật Doanh nghiệp", "parent_article_id": "art_12", "rank": 2, "score": 15.0},
        {"chunk_id": "neg_cross_doc", "doc_name": "Luật Đầu tư", "parent_article_id": "art_5", "rank": 3, "score": 10.0},
    ]

    negs = mine_retrieval_hard_negatives(
        qa_id="q1",
        question="Điều kiện thành lập doanh nghiệp?",
        positive_chunk_ids={"pos_1"},
        positive_article_ids={"art_10"},
        positive_doc_names={"Luật Doanh nghiệp"},
        retrieved_candidates=retrieved,
        lookup=lookup,
    )

    neg_cids = {n["negative_chunk_id"] for n in negs}
    assert "pos_1" not in neg_cids, "Positive chunk must be strictly excluded from negatives!"
    assert "neg_same_doc" in neg_cids
    assert "neg_cross_doc" in neg_cids
    assert any(n["negative_type"] == "same_doc_wrong_article" for n in negs)
