"""BM25 rebuild round-trip regressions (order binding + strict reload)."""

from __future__ import annotations

import pytest

from src.common.bm25 import BM25Retriever


def _corpus():
    return [
        {"chunk_id": "a", "text_raw": "Phạt tiền 800.000 đồng theo Nghị định 100.", "text_norm": ""},
        {"chunk_id": "b", "text_raw": "Báo cáo định kỳ hàng quý.", "text_norm": ""},
        {"chunk_id": "a", "text_raw": "Phạt tiền 800.000 đồng theo Nghị định 100.", "text_norm": ""},
    ]


def test_save_load_roundtrip_preserves_order(tmp_path):
    pytest.importorskip("bm25s")
    import pandas as pd

    corpus_path = tmp_path / "chunks.parquet"
    pd.DataFrame(_corpus()).to_parquet(corpus_path)
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.fit(_corpus())
    assert retriever.bm25s_index is not None
    retriever.save(str(tmp_path))
    loaded = BM25Retriever.load(str(tmp_path), corpus_path=str(corpus_path), fail_on_missing_index=True)
    assert loaded.k1 == 1.5 and loaded.b == 0.75
    assert loaded.doc_ids == ["a", "b", "a"]
    hits = loaded.search("phạt tiền", top_k=3)
    assert hits and hits[0]["chunk_id"] == "a"


def test_load_rejects_order_mismatch(tmp_path):
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.fit(_corpus())
    retriever.save(str(tmp_path))
    import pandas as pd

    other = [{"chunk_id": "z", "text_raw": "Hoàn toàn khác."}] * 3
    other_path = tmp_path / "other.parquet"
    pd.DataFrame(other).to_parquet(other_path)
    with pytest.raises(ValueError, match="doc_ids do not match"):
        BM25Retriever.load(str(tmp_path), corpus_path=str(other_path), fail_on_missing_index=True)
