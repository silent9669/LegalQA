from src.task2.article_stitcher import ArticleStitcher

def test_article_stitcher():
    all_chunks = [
        {"chunk_id": "doc1_art17_p1", "parent_article_id": "doc1_art17", "doc_name": "Nghị định 90", "text_raw": "1. Phạt tiền từ 1 đến 2 triệu đồng.", "start_char": 100},
        {"chunk_id": "doc1_art17_p2", "parent_article_id": "doc1_art17", "doc_name": "Nghị định 90", "text_raw": "2. Phạt tiền từ 2 đến 3 triệu đồng.", "start_char": 150},
        {"chunk_id": "doc1_art18_p1", "parent_article_id": "doc1_art18", "doc_name": "Nghị định 90", "text_raw": "1. Hành vi vi phạm khác.", "start_char": 200}
    ]
    stitcher = ArticleStitcher(all_chunks)

    seeds = [{"chunk_id": "doc1_art17_p2", "parent_article_id": "doc1_art17", "rerank_score": 0.9}]
    stitched = stitcher.stitch(seeds)

    assert stitched["parent_article_id"] == "doc1_art17"
    assert "1. Phạt tiền từ 1 đến 2 triệu" in stitched["stitched_text"]
    assert "2. Phạt tiền từ 2 đến 3 triệu" in stitched["stitched_text"]
