from pathlib import Path
import numpy as np
import pandas as pd
from src.common.dense_dek21 import DEk21Retriever
from src.common.rrf import reciprocal_rank_fusion


def test_rrf_fusion():
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


def test_dek21_retriever_mock(tmp_path: Path):
    retriever = DEk21Retriever(model_name="mock")
    corpus = [
        {"chunk_id": "c1", "text_raw": "Quy định hiệu lực thi hành từ ngày 01 tháng 7 năm 2023"},
        {"chunk_id": "c2", "text_raw": "Xử phạt hành vi không đội mũ bảo hiểm"}
    ]
    retriever.fit_mock(corpus)
    res = retriever.search("Ngày bắt đầu áp dụng", top_k=2)
    assert len(res) == 2

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
    res_loaded = loaded.search("Ngày bắt đầu áp dụng", top_k=2)
    assert len(res_loaded) == 2
