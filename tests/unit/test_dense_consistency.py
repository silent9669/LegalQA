"""Dense self-consistency gate regressions (no encoder weights needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.rebuild_dense_index import check_dense_alignment, ensure_dense_index, require_pinned_revision
from src.common.dense import (
    DenseRetriever,
    duplicate_text_pairs,
    embedding_self_consistency,
)


def _corpus():
    return [
        {"chunk_id": "c", "text_raw": "first"},
        {"chunk_id": "x", "text_raw": "other"},
        {"chunk_id": "c", "text_raw": "first"},
        {"chunk_id": "y", "text_raw": "third"},
    ]


def test_duplicate_pairs_find_identical_texts():
    pairs = duplicate_text_pairs(_corpus())
    assert pairs == [(0, 2)]


def test_aligned_matrix_passes_and_permuted_fails():
    rng = np.random.default_rng(42)
    base = rng.normal(size=(4, 8)).astype(np.float32)
    aligned = base.copy()
    aligned[2] = base[0]  # identical texts share a vector
    pairs = duplicate_text_pairs(_corpus())
    ok = embedding_self_consistency(aligned, pairs)
    assert ok["aligned"] is True and ok["pass_rate"] == 1.0
    foreign = rng.normal(size=(4, 8)).astype(np.float32)
    bad = embedding_self_consistency(foreign, pairs)
    assert bad["aligned"] is False and bad["mean_cosine"] < 0.5


def test_load_index_self_consistency_gate(tmp_path):
    import pandas as pd

    corpus = _corpus()
    corpus_path = tmp_path / "chunks.parquet"
    pd.DataFrame(corpus).to_parquet(corpus_path)

    rng = np.random.default_rng(7)
    vecs = rng.normal(size=(4, 768)).astype(np.float32)
    vecs[2] = vecs[0]
    ret = DenseRetriever(model_name="mock")
    ret.corpus = corpus
    ret.doc_ids = [c["chunk_id"] for c in corpus]
    ret.corpus_embeddings = vecs
    ret.save_index(str(tmp_path), dtype="float32")
    ok = DenseRetriever.load_index(
        str(tmp_path), corpus_path=str(corpus_path),
        final_mode=True, verify_self_consistency=True,
    )
    assert ok.self_consistency_report["aligned"] is True

    # Tamper: roll rows by one (simulates order/content mismatch).
    rolled = np.roll(vecs, 1, axis=0)
    ret2 = DenseRetriever(model_name="mock")
    ret2.corpus = corpus
    ret2.doc_ids = [c["chunk_id"] for c in corpus]
    ret2.corpus_embeddings = rolled
    ret2.save_index(str(tmp_path), dtype="float32")
    with pytest.raises(ValueError, match="self-consistency"):
        DenseRetriever.load_index(
            str(tmp_path), corpus_path=str(corpus_path),
            final_mode=True, verify_self_consistency=True,
        )


def test_require_pinned_revision_rejects_floating():
    assert require_pinned_revision("a" * 40) == "a" * 40
    with pytest.raises(ValueError, match="floating"):
        require_pinned_revision("main")


def test_check_alignment_reports_real_staged_index():
    pytest.importorskip("pandas")
    if not (
        Path("kaggle_dataset/indexes/dek21/embeddings.npy").is_file()
        and Path("kaggle_dataset/legal_chunks.parquet").is_file()
    ):
        pytest.skip("staged dataset index not present (CI checkout has no data)")
    report = check_dense_alignment("kaggle_dataset/indexes/dek21", "kaggle_dataset/legal_chunks.parquet")
    assert report["status"] == "measured"
    assert report["aligned"] is False  # quarantined: mean cosine ~0.0
    assert report["num_pairs"] == 400


def test_ensure_reuses_aligned_fixture(tmp_path):
    import pandas as pd

    corpus = _corpus()
    corpus_path = tmp_path / "chunks.parquet"
    pd.DataFrame(corpus).to_parquet(corpus_path)
    rng = np.random.default_rng(11)
    vecs = rng.normal(size=(4, 768)).astype(np.float32)
    vecs[2] = vecs[0]
    ret = DenseRetriever(model_name="mock")
    ret.corpus = corpus
    ret.doc_ids = [c["chunk_id"] for c in corpus]
    ret.corpus_embeddings = vecs
    index_dir = tmp_path / "index"
    ret.save_index(str(index_dir), dtype="float32")
    report = ensure_dense_index(
        str(index_dir), str(corpus_path), "mock-model", "b" * 40,
        device="cpu", batch_size=2,
    )
    assert report["action"] == "reused" and report["aligned"] is True
