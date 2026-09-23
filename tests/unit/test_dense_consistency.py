"""Dense self-consistency gate regressions (no encoder weights needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.rebuild_dense_index import (
    check_dense_alignment,
    ensure_dense_index,
    require_pinned_revision,
    select_verified_dense_index,
)
from src.common.dense import (
    DenseRetriever,
    duplicate_text_pairs,
    embedding_self_consistency,
    preprocessing_fingerprint,
    verify_dense_manifest_identity,
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


# ----------------------------------------------------------------------
# P0-B: exact dense/index binding (bytes, not family/dim/rows)
# ----------------------------------------------------------------------

def _strict_corpus():
    return [
        {"chunk_id": "c", "text_raw": "first"},
        {"chunk_id": "x", "text_raw": "other"},
        {"chunk_id": "c", "text_raw": "first"},
        {"chunk_id": "y", "text_raw": "third"},
    ]


def _save_strict_index(tmp_path, name, weights_sha, revision="c" * 40, vecs=None, corpus=None):
    import pandas as pd

    corpus = corpus if corpus is not None else _strict_corpus()
    (tmp_path / f"{name}.parquet").parent.mkdir(parents=True, exist_ok=True)
    corpus_path = tmp_path / f"{name}.parquet"
    pd.DataFrame(corpus).to_parquet(corpus_path)
    if vecs is None:
        rng = np.random.default_rng(21)
        vecs = rng.normal(size=(len(corpus), 768)).astype(np.float32)
        vecs[2] = vecs[0]
    ret = DenseRetriever(model_name="runs/20260920-215402/encoder_ft_v2", revision=revision)
    ret.corpus = corpus
    ret.doc_ids = [c["chunk_id"] for c in corpus]
    ret.corpus_embeddings = vecs
    index_dir = tmp_path / name
    ret.save_index(
        str(index_dir), dtype="float32",
        encoder_weights_sha256=weights_sha,
        preprocessing=preprocessing_fingerprint(),
    )
    return index_dir, corpus_path


def _expected(weights_sha, revision="c" * 40):
    return {
        "model_id": "runs/20260920-215402/encoder_ft_v2",
        "revision": revision,
        "encoder_weights_sha256": weights_sha,
        "preprocessing": preprocessing_fingerprint(),
    }


def test_strict_rejects_same_family_different_weights(tmp_path):
    idx, corpus_path = _save_strict_index(tmp_path, "idx", "aa" * 32)
    # Same family, same dim, same rows, valid self-consistency — but the
    # encoder bytes differ: strict reuse must refuse.
    with pytest.raises(ValueError, match="weights mismatch"):
        verify_dense_manifest_identity(
            __import__("json").loads((idx / "dense_manifest.json").read_text(encoding="utf-8")),
            _expected("bb" * 32),
        )
    with pytest.raises(ValueError, match="weights mismatch"):
        DenseRetriever.load_index(
            str(idx), corpus_path=str(corpus_path),
            expected_model_name="runs/20260920-215402/encoder_ft_v2",
            expected_revision="c" * 40,
            expected_encoder_weights_sha256="bb" * 32,
            expected_preprocessing=preprocessing_fingerprint(),
            strict_identity=True,
        )
    sel = select_verified_dense_index([str(idx)], str(corpus_path), _expected("bb" * 32))
    assert sel["action"] == "rebuild_needed"


def test_strict_rejects_legacy_manifest_without_identity(tmp_path):
    import json

    import pandas as pd

    corpus = _strict_corpus()
    corpus_path = tmp_path / "chunks.parquet"
    pd.DataFrame(corpus).to_parquet(corpus_path)
    rng = np.random.default_rng(31)
    vecs = rng.normal(size=(4, 768)).astype(np.float32)
    vecs[2] = vecs[0]
    ret = DenseRetriever(model_name="mock")
    ret.corpus = corpus
    ret.doc_ids = [c["chunk_id"] for c in corpus]
    ret.corpus_embeddings = vecs
    idx = tmp_path / "legacy"
    ret.save_index(str(idx), dtype="float32")
    manifest = json.loads((idx / "dense_manifest.json").read_text(encoding="utf-8"))
    manifest.pop("encoder_weights_sha256", None)
    manifest.pop("preprocessing", None)
    manifest["revision"] = None
    (idx / "dense_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing identity"):
        DenseRetriever.load_index(
            str(idx), corpus_path=str(corpus_path),
            expected_model_name="mock", strict_identity=True,
        )
    sel = select_verified_dense_index([str(idx)], str(corpus_path), _expected("aa" * 32))
    assert sel["action"] == "rebuild_needed"


def test_strict_rejects_corpus_order_and_embeddings_tamper(tmp_path):
    import json

    import pandas as pd

    idx, corpus_path = _save_strict_index(tmp_path, "idx", "aa" * 32)
    manifest = json.loads((idx / "dense_manifest.json").read_text(encoding="utf-8"))

    # Permuted corpus rows fail the order/content binding.
    shuffled = list(reversed(_strict_corpus()))
    pd.DataFrame(shuffled).to_parquet(tmp_path / "shuffled.parquet")
    sel = select_verified_dense_index([str(idx)], str(tmp_path / "shuffled.parquet"), _expected("aa" * 32))
    assert sel["action"] == "rebuild_needed"

    # One flipped embedding byte fails the bytes binding.
    tampered = np.load(str(idx / "embeddings.npy")).copy()
    tampered[0, 0] += 1.0
    np.save(str(idx / "embeddings.npy"), tampered)
    assert manifest["embeddings_sha256"] != __import__("hashlib").sha256(
        (idx / "embeddings.npy").read_bytes()).hexdigest()
    sel2 = select_verified_dense_index([str(idx)], str(corpus_path), _expected("aa" * 32))
    assert sel2["action"] == "rebuild_needed"


def test_strict_rejects_preprocessing_drift(tmp_path):
    import json

    idx, _ = _save_strict_index(tmp_path, "idx", "aa" * 32)
    manifest = json.loads((idx / "dense_manifest.json").read_text(encoding="utf-8"))
    drifted = dict(_expected("aa" * 32))
    drifted["preprocessing"] = preprocessing_fingerprint(vietnamese_tokenized=False)
    with pytest.raises(ValueError, match="preprocessing mismatch"):
        verify_dense_manifest_identity(manifest, drifted)


def test_strict_reuse_passes_on_exact_identity(tmp_path):
    idx, corpus_path = _save_strict_index(tmp_path, "idx", "aa" * 32)
    sel = select_verified_dense_index([str(idx)], str(corpus_path), _expected("aa" * 32))
    assert sel["action"] == "reused"
    loaded = DenseRetriever.load_index(
        str(idx), corpus_path=str(corpus_path),
        expected_model_name="runs/20260920-215402/encoder_ft_v2",
        expected_revision="c" * 40,
        expected_encoder_weights_sha256="aa" * 32,
        expected_preprocessing=preprocessing_fingerprint(),
        strict_identity=True,
        verify_embeddings_hash=True,
    )
    assert loaded.corpus_embeddings.shape == (4, 768)
