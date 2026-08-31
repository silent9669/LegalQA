from src.common.reranker import BGEReranker


def test_bge_reranker_mock():
    reranker = BGEReranker(model_name="mock")
    candidates = [
        {"chunk_id": "c1", "text_raw": "[DOCUMENT] Nghị định 90\n[ARTICLE] Điều 16. Vi phạm khác", "score": 0.5},
        {"chunk_id": "c2", "text_raw": "[DOCUMENT] Nghị định 90\n[ARTICLE] Điều 17. Vi phạm tiêm phòng không tiêm", "score": 0.4}
    ]
    reranked = reranker.rerank("Hành vi không tiêm phòng phạt bao nhiêu?", candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "c2"  # Higher semantic match


def test_reranker_batch_equivalence():
    """Verify rerank_batch produces exact identical result list to individual rerank calls."""
    reranker = BGEReranker(model_name="mock")
    q1 = "Hành vi không tiêm phòng phạt bao nhiêu?"
    c1 = [
        {"chunk_id": "c1", "text_raw": "[DOCUMENT] Nghị định 90\n[ARTICLE] Điều 16. Vi phạm khác", "score": 0.5},
        {"chunk_id": "c2", "text_raw": "[DOCUMENT] Nghị định 90\n[ARTICLE] Điều 17. Vi phạm tiêm phòng không tiêm", "score": 0.4}
    ]

    q2 = "Thời hiệu xử phạt vi phạm hành chính?"
    c2 = [
        {"chunk_id": "c3", "text_raw": "[DOCUMENT] Luật XLVPHC\n[ARTICLE] Điều 6. Thời hiệu xử phạt", "score": 0.6},
        {"chunk_id": "c4", "text_raw": "[DOCUMENT] Nghị định 90\n[ARTICLE] Điều 1. Phạm vi điều chỉnh", "score": 0.3}
    ]

    single_1 = reranker.rerank(q1, c1, top_k=2)
    single_2 = reranker.rerank(q2, c2, top_k=2)

    batch_res = reranker.rerank_batch([q1, q2], [c1, c2], top_k=2)

    assert len(batch_res) == 2
    assert batch_res[0] == single_1
    assert batch_res[1] == single_2
