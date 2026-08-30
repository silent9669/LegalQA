from src.common.bm25 import BM25Retriever

def test_bm25_retriever_with_legal_booster():
    corpus = [
        {"chunk_id": "c1", "text_raw": "[DOCUMENT] Nghị định 90/2017/NĐ-CP\n[ARTICLE] Điều 17. Xử phạt không tiêm vắc xin", "text_norm": "Nghị_định 90/2017/NĐ-CP Điều 17 Xử_phạt không tiêm vắc_xin"},
        {"chunk_id": "c2", "text_raw": "[DOCUMENT] Nghị định 100/2019/NĐ-CP\n[ARTICLE] Điều 5. Vi phạm nồng độ cồn", "text_norm": "Nghị_định 100/2019/NĐ-CP Điều 5 Vi_phạm nồng_độ cồn"},
        {"chunk_id": "c3", "text_raw": "[DOCUMENT] Thông tư 20/2021/TT-BYT\n[ARTICLE] Điều 17. Tiêm chủng cho trẻ em", "text_norm": "Thông_tư 20/2021/TT-BYT Điều 17 Tiêm_chủng cho trẻ_em"}
    ]
    retriever = BM25Retriever()
    retriever.fit(corpus)

    # Query with exact doc number and article
    results = retriever.search("Theo Điều 17 Nghị định 90/2017 xử phạt thế nào?", top_k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "c1"
