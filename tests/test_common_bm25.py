import os
from pathlib import Path
import pandas as pd
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
