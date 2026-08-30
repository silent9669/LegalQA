from src.common.reranker import BGEReranker

def test_bge_reranker_mock():
    reranker = BGEReranker(model_name="mock")
    candidates = [
        {"chunk_id": "c1", "text_raw": "[DOCUMENT] Nghị định 90\n[ARTICLE] Điều 16. Vi phạm khác", "score": 0.5},
        {"chunk_id": "c2", "text_raw": "[DOCUMENT] Nghị định 90\n[ARTICLE] Điều 17. Vi phạm tiêm phòng không tiêm", "score": 0.4}
    ]
    reranked = reranker.rerank("Hành vi không tiêm phòng phạt bao nhiêu?", candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "c2" # Higher semantic match
