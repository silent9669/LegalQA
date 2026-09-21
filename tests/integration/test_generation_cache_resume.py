import json
import os
from unittest.mock import MagicMock
from src.task2.predict import LegalQAPipeline, is_degenerate_answer


def test_is_degenerate_answer_rejects_header_only():
    assert is_degenerate_answer("") is True
    assert is_degenerate_answer("   ") is True
    assert is_degenerate_answer("Căn cứ quy định của pháp luật:") is True
    assert is_degenerate_answer("Căn cứ quy định của pháp luật hiện hành:") is True
    assert is_degenerate_answer("Căn cứ theo quy định của pháp luật.") is True
    assert is_degenerate_answer("Căn cứ Điều 12 Luật Doanh nghiệp 2020, doanh nghiệp có quyền tự do kinh doanh.") is False


def test_generation_cache_resume_avoids_repeated_inference(tmp_path):
    cache_file = tmp_path / "gen_raw_cache.jsonl"
    
    mock_memory = MagicMock()
    mock_memory.lookup_exact.return_value = None
    mock_memory.lookup_fuzzy.return_value = None
    
    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [{"chunk_id": "c1", "text_raw": "Điều 1...", "parent_article_id": "art1", "score": 10.0}]
    mock_bm25.search_batch.return_value = [[{"chunk_id": "c1", "text_raw": "Điều 1...", "parent_article_id": "art1", "score": 10.0}]]
    
    mock_dense = MagicMock()
    mock_dense.search_batch.return_value = [[{"chunk_id": "c1", "text_raw": "Điều 1...", "parent_article_id": "art1", "score": 10.0}]]
    
    mock_reranker = MagicMock()
    mock_reranker.rerank_batch.return_value = [[{"chunk_id": "c1", "text_raw": "Điều 1...", "parent_article_id": "art1", "rerank_score": 5.0}]]
    
    mock_packer = MagicMock()
    mock_packer.prepare_seeds.return_value = [{"chunk_id": "c1", "text_raw": "Điều 1...", "parent_article_id": "art1"}]
    mock_packer.pack_evidence.return_value = {"text": "Điều 1 nội dung...", "top_doc_name": "Luật X", "top_article_num": "1"}
    
    mock_gen = MagicMock()
    mock_gen.format_instance_prompt.return_value = "prompt_text"
    mock_gen.generate_batch.return_value = ["Generated response 1"]
    
    mock_selector = MagicMock()
    mock_selector.policy = "fixed_baseline"
    mock_selector.best_fixed_candidate = "dual_assembled"
    mock_selector.select_with_source.return_value = ("Final answer 1", "dual_assembled")
    
    pipeline = LegalQAPipeline(
        memory=mock_memory,
        bm25=mock_bm25,
        dense=mock_dense,
        reranker=mock_reranker,
        packer=mock_packer,
        generator=mock_gen,
        selector=mock_selector,
    )
    
    items = [{"id": "q1", "question": "Question 1"}]
    
    # First call: cache is empty, generator must be called
    res1 = pipeline.predict_batch(items, raw_cache_path=str(cache_file))
    assert mock_gen.generate_batch.call_count == 1
    assert os.path.exists(cache_file)
    
    # Second call with same items: generator must NOT be called again
    res2 = pipeline.predict_batch(items, raw_cache_path=str(cache_file))
    assert mock_gen.generate_batch.call_count == 1  # unchanged!
    assert res1["q1"]["answer"] == res2["q1"]["answer"]
