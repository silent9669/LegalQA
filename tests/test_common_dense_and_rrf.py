from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.common.dense import DenseRetriever
from src.common.dense_dek21 import DEk21Retriever
from src.common.reranker import BGEReranker
from src.common.rrf import reciprocal_rank_fusion


def test_rrf_fusion_deterministic():
    run_bm25 = [
        {"chunk_id": "c1", "score": 10.0},
        {"chunk_id": "c2", "score": 8.0}
    ]
    run_dense = [
        {"chunk_id": "c2", "score": 0.95},
        {"chunk_id": "c3", "score": 0.85}
    ]
    fused = reciprocal_rank_fusion([run_bm25, run_dense], k=60, weights=[0.5, 0.5])
    assert len(fused) == 3
    # c2 appears in both lists, should rank #1
    assert fused[0]["chunk_id"] == "c2"
    assert fused[0]["rank"] == 1
    # Verify deterministic output
    fused2 = reciprocal_rank_fusion([run_bm25, run_dense], k=60, weights=[0.5, 0.5])
    assert [f["chunk_id"] for f in fused] == [f["chunk_id"] for f in fused2]


def test_dek21_retriever_mock(tmp_path: Path):
    retriever = DEk21Retriever(model_name="mock")
    corpus = [
        {"chunk_id": "c1", "text_raw": "Quy định hiệu lực thi hành từ ngày 01 tháng 7 năm 2023"},
        {"chunk_id": "c2", "text_raw": "Xử phạt hành vi không đội mũ bảo hiểm"}
    ]
    retriever.fit_mock(corpus)
    res = retriever.search("Ngày bắt đầu áp dụng", top_k=2)
    assert len(res) == 2

    # Test batched search
    batch_res = retriever.search_batch(["Ngày bắt đầu áp dụng", "Mũ bảo hiểm"], top_k=2)
    assert len(batch_res) == 2
    assert len(batch_res[0]) == 2
    assert len(batch_res[1]) == 2

    # Test save and load index
    index_dir = tmp_path / "dek21_index"
    retriever.save_index(str(index_dir))
    assert (index_dir / "embeddings.npy").exists()
    assert (index_dir / "dek21_manifest.json").exists()

    corpus_file = tmp_path / "legal_chunks.parquet"
    pd.DataFrame(corpus).to_parquet(corpus_file, index=False)

    loaded = DEk21Retriever.load_index(str(index_dir), corpus_path=str(corpus_file), model_name="mock")
    assert loaded.corpus_embeddings is not None
    assert len(loaded.corpus) == 2
    assert loaded.doc_ids == ["c1", "c2"]


def test_dense_expected_model_mismatch_fails(tmp_path: Path):
    retriever = DenseRetriever(model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2")
    corpus = [{"chunk_id": "c1", "text_raw": "T1"}]
    retriever.fit_mock(corpus)
    index_dir = tmp_path / "dense_index"
    retriever.save_index(str(index_dir))

    # Loading with mismatched expected model name in final mode must raise ValueError
    with pytest.raises(ValueError, match="Dense model mismatch"):
        DenseRetriever.load_index(
            str(index_dir),
            expected_model_name="BAAI/bge-m3",
            final_mode=True,
        )


def test_reranker_batch_matches_single():
    reranker = BGEReranker(model_name="mock")
    candidates_1 = [
        {"chunk_id": "c1", "text_raw": "Luật giao thông đường bộ"},
        {"chunk_id": "c2", "text_raw": "Luật đất đai quy định sổ đỏ"},
    ]
    candidates_2 = [
        {"chunk_id": "c3", "text_raw": "Nghị định 90 xử phạt thú y"},
        {"chunk_id": "c4", "text_raw": "Thông tư tiêm phòng vắc xin"},
    ]

    single_1 = reranker.rerank("giao thông", candidates_1, top_k=2)
    single_2 = reranker.rerank("tiêm phòng", candidates_2, top_k=2)

    batch_out = reranker.rerank_batch(["giao thông", "tiêm phòng"], [candidates_1, candidates_2], top_k=2)

    assert len(batch_out) == 2
    assert [c["chunk_id"] for c in batch_out[0]] == [c["chunk_id"] for c in single_1]
    assert [c["chunk_id"] for c in batch_out[1]] == [c["chunk_id"] for c in single_2]
