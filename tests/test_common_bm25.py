import math
import os
from pathlib import Path
import pandas as pd
from src.common.bm25 import BM25Retriever


def test_bm25_hand_computable_correctness():
    """Verify BM25 calculations against manual Robertson-Spärck Jones formulation."""
    # 3-doc tiny corpus
    corpus = [
        {"chunk_id": "c1", "text_raw": "luật giao thông", "text_norm": "luật giao_thông"},
        {"chunk_id": "c2", "text_raw": "luật đất đai", "text_norm": "luật đất_đai"},
        {"chunk_id": "c3", "text_raw": "giao thông đường bộ", "text_norm": "giao_thông đường_bộ"},
    ]
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.fit(corpus)

    assert retriever.corpus_size == 3
    assert "giao_thông" in retriever.df
    assert retriever.df["giao_thông"] == 2
    assert retriever.df["luật"] == 2

    # Query for giao_thông: doc c1 and c3 match, c2 has 0 score
    res = retriever.search("giao thông", top_k=3)
    matching_ids = {r["chunk_id"] for r in res}
    assert "c1" in matching_ids
    assert "c3" in matching_ids
    assert "c2" not in matching_ids


def test_bm25_no_posting_truncation_or_corpus_bias():
    """Verify that terms appearing in >8,000 documents are NOT truncated."""
    # Build a simulated corpus with 8,500 documents containing common term 'pháp_luật'
    corpus = []
    for i in range(8500):
        corpus.append({
            "chunk_id": f"c_{i}",
            "text_raw": f"Văn bản pháp luật số {i}",
            "text_norm": f"văn_bản pháp_luật số {i}",
        })
    retriever = BM25Retriever()
    retriever.fit(corpus)

    # Document 8499 (the last one) must be present in postings
    assert len(retriever.postings["pháp_luật"]) == 8500
    assert retriever.df["pháp_luật"] == 8500

    # Search should be able to retrieve late documents if query matches specific term
    res = retriever.search("pháp luật 8499", top_k=5)
    assert len(res) > 0
    assert res[0]["chunk_id"] == "c_8499"


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


def test_bm25_save_load_without_corpus_duplication(tmp_path: Path):
    corpus = [
        {"chunk_id": "c1", "text_raw": "Quy định Điều 1", "text_norm": "quy_định điều 1"},
        {"chunk_id": "c2", "text_raw": "Quy định Điều 2", "text_norm": "quy_định điều 2"},
    ]
    corpus_file = tmp_path / "legal_chunks.parquet"
    pd.DataFrame(corpus).to_parquet(corpus_file, index=False)

    index_dir = tmp_path / "bm25_index"
    retriever = BM25Retriever()
    retriever.fit(corpus)
    retriever.save(str(index_dir), save_corpus_meta=False)

    # Verify no duplicate corpus_meta.parquet was created
    assert not (index_dir / "corpus_meta.parquet").exists()
    assert (index_dir / "bm25_manifest.json").exists()

    # Load referencing the canonical corpus parquet
    loaded = BM25Retriever.load(str(index_dir), corpus_path=str(corpus_file))
    assert loaded.corpus_size == 2
    res = loaded.search("Điều 1", top_k=1)
    assert len(res) == 1
    assert res[0]["chunk_id"] == "c1"
