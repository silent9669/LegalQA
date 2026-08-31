from src.task2.article_stitcher import ArticleStitcher


def test_article_stitcher_primary_siblings():
    all_chunks = [
        {"chunk_id": "doc1_art17_p1", "parent_article_id": "doc1_art17", "doc_name": "Nghị định 90", "text_raw": "1. Phạt tiền từ 1 đến 2 triệu đồng.", "start_char": 100, "clause_number": "1"},
        {"chunk_id": "doc1_art17_p2", "parent_article_id": "doc1_art17", "doc_name": "Nghị định 90", "text_raw": "2. Phạt tiền từ 2 đến 3 triệu đồng.", "start_char": 150, "clause_number": "2"},
        {"chunk_id": "doc1_art18_p1", "parent_article_id": "doc1_art18", "doc_name": "Nghị định 90", "text_raw": "1. Hành vi vi phạm khác.", "start_char": 200, "clause_number": "1"}
    ]
    stitcher = ArticleStitcher(all_chunks)

    seeds = [{"chunk_id": "doc1_art17_p2", "parent_article_id": "doc1_art17", "rerank_score": 0.9, "text_raw": "2. Phạt tiền từ 2 đến 3 triệu đồng."}]
    stitched = stitcher.stitch(seeds, max_chars=3000)

    assert stitched["parent_article_id"] == "doc1_art17"
    assert "1. Phạt tiền từ 1 đến 2 triệu" in stitched["stitched_text"]
    assert "2. Phạt tiền từ 2 đến 3 triệu" in stitched["stitched_text"]


def test_article_stitcher_multi_seed_budget():
    all_chunks = [
        {"chunk_id": "doc1_art1_p1", "parent_article_id": "doc1_art1", "doc_name": "Luật 01", "text_raw": "Điều 1. Phạm vi điều chỉnh.", "start_char": 10},
        {"chunk_id": "doc1_art2_p1", "parent_article_id": "doc1_art2", "doc_name": "Luật 01", "text_raw": "Điều 2. Đối tượng áp dụng.", "start_char": 50},
    ]
    stitcher = ArticleStitcher(all_chunks)

    seeds = [
        {"chunk_id": "doc1_art1_p1", "parent_article_id": "doc1_art1", "text_raw": "Điều 1. Phạm vi điều chỉnh."},
        {"chunk_id": "doc1_art2_p1", "parent_article_id": "doc1_art2", "text_raw": "Điều 2. Đối tượng áp dụng."},
    ]
    stitched = stitcher.stitch(seeds, max_chars=3500)

    assert "Điều 1" in stitched["stitched_text"]
    assert "Điều 2" in stitched["stitched_text"]
    assert stitched["total_chars"] <= 3500
