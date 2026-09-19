"""Inheritance regressions: v10 retrieval experiments (opt-in, defaults unchanged)."""

from __future__ import annotations

import pytest

from src.common.legal_reference import build_legal_reference_index
from src.task2.evidence_packer import diversify_chunks, reorder_lost_in_middle
from src.task2.predict import LegalQAPipeline


def _pipeline():
    return LegalQAPipeline.build_mock()


def test_default_fusion_reproduces_legacy_recipe():
    pipe = _pipeline()
    assert pipe.retrieval_options["use_legal_reference"] is False
    trace = pipe.retrieve_and_rerank("Phạt tiền bao nhiêu?")
    assert trace["lexref_results"] == []
    assert trace["retrieval_meta"]["lexref_fired"] is False
    assert trace["retrieval_meta"]["rrf_weights"] == [0.5, 0.5]
    assert trace["retrieval_meta"]["query_rewritten"] is False


def test_legal_reference_arm_fires_on_doc_query():
    chunks = [
        {"chunk_id": "a17", "doc_name": "Nghi-dinh-100-2019-ND-CP", "parent_article_id": "d1_a17",
         "article_number": "17", "text_raw": "Phạt tiền 800.000 đồng."},
        {"chunk_id": "zzz", "doc_name": "Thong-tu-12-2019-TT-BTP", "parent_article_id": "d2_a5",
         "article_number": "5", "text_raw": "Báo cáo định kỳ."},
    ]
    index, _ = build_legal_reference_index(chunks)
    pipe = _pipeline()
    pipe.legal_index = index
    pipe.legal_rows = chunks
    trace = pipe.retrieve_and_rerank("Theo Nghị định 100/2019 Điều 17 phạt bao nhiêu?", use_legal_reference=True)
    assert trace["retrieval_meta"]["lexref_fired"] is True
    assert trace["lexref_results"][0]["chunk_id"] == "a17"
    assert abs(sum(trace["retrieval_meta"]["rrf_weights"]) - 1.0) < 1e-9


def test_weighted_rrf_and_acronym_rewrite_opt_in():
    pipe = _pipeline()
    trace = pipe.retrieve_and_rerank("NLĐ?", use_acronyms=True, use_weighted_rrf=True)
    assert trace["retrieval_meta"]["query_rewritten"] is True
    # Legacy defaults: weighted lex-query over two arms renormalizes to uniform.
    trace2 = pipe.retrieve_and_rerank("Theo Điều 17?", use_weighted_rrf=True)
    assert trace2["retrieval_meta"]["rrf_weights"] == [0.5, 0.5]
    # Doc-prescribed weights flow from the candidate config.
    doc_pipe = LegalQAPipeline.build_mock(
        retrieval_options={"use_weighted_rrf": True},
        retrieval_weights={"w_bm25_plain": 0.55, "w_dense_plain": 0.45,
                           "w_bm25_lex": 0.40, "w_dense_lex": 0.27, "w_lexref_lex": 0.33},
    )
    trace3 = doc_pipe.retrieve_and_rerank("Theo Điều 17?")
    # Lex row (0.40, 0.27) restricted to two arms, renormalized.
    assert trace3["retrieval_meta"]["rrf_weights"] == pytest.approx([0.40 / 0.67, 0.27 / 0.67])
    # Direct helper: lex triple restricted to present arms, renormalized.
    assert doc_pipe.rrf_arm_weights(["bm25", "dense", "lexref"], True) == pytest.approx([0.40, 0.27, 0.33])
    with pytest.raises(ValueError, match="unknown retrieval weights"):
        LegalQAPipeline.build_mock(retrieval_weights={"nope": 1.0})


def test_unknown_retrieval_option_fails_closed():
    pipe = _pipeline()
    try:
        pipe.retrieve_and_rerank("Q?", no_such_option=True)
    except ValueError as exc:
        assert "unknown retrieval options" in str(exc)
    else:
        raise AssertionError("unknown option must raise")


def test_diversify_caps_per_article_and_lost_in_middle_reorders():
    chunks = [
        {"chunk_id": f"a{i}", "parent_article_id": "art1", "text_raw": f"E{i}"} for i in range(4)
    ] + [{"chunk_id": "b0", "parent_article_id": "art2", "text_raw": "F"}]
    out = diversify_chunks(chunks, max_parts_per_article=2)
    assert [c["chunk_id"] for c in out] == ["a0", "a1", "b0"]
    assert reorder_lost_in_middle(["r1", "r2", "r3", "r4"]) == ["r1", "r3", "r4", "r2"]
    assert reorder_lost_in_middle([]) == []


def test_prepare_seeds_defaults_to_legacy_order():
    from src.task2.evidence_packer import EvidencePacker

    packer = EvidencePacker([])
    seeds = [{"chunk_id": str(i)} for i in range(3)]
    assert packer.prepare_seeds(seeds) == seeds
