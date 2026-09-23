"""Dense self-consistency gate regressions (no encoder weights needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.rebuild_dense_index import (
    _promote_index,
    build_verified_index,
    check_dense_alignment,
    ensure_dense_index,
    require_pinned_revision,
    select_verified_dense_index,
)
from src.common.dense import (
    DenseRetriever,
    build_embedding_row_keys_for_texts,
    compute_embedding_order_hash,
    duplicate_text_pairs,
    embedding_self_consistency,
    hash_encoder_weights_dir,
    preprocessing_fingerprint,
    resolve_encode_texts,
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


# ----------------------------------------------------------------------
# Review blockers: real builder output, text_norm binding, promotion
# ----------------------------------------------------------------------

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

def _mock_encoder_dir(tmp_path, name="encoder", weights=b"encoder-bytes-v1"):
    enc = tmp_path / name
    enc.mkdir(parents=True, exist_ok=True)
    (enc / "model.safetensors").write_bytes(weights)
    (enc / "config.json").write_text('{"model_type": "roberta"}', encoding="utf-8")
    return enc


def _stub_encode_texts(monkeypatch, dim=32):
    import hashlib as _h

    def _fake(self, texts, batch_size=64, show_progress=False, pre_tokenized=False):
        vecs = []
        for text in texts:
            seed = int.from_bytes(_h.sha256(str(text).encode("utf-8")).digest()[:8], "big") % (2 ** 31)
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=(dim,)).astype(np.float32)
            vec /= max(np.linalg.norm(vec), 1e-12)
            vecs.append(vec)
        return np.stack(vecs)

    monkeypatch.setattr(DenseRetriever, "encode_texts", _fake)
    monkeypatch.setattr(DenseRetriever, "_get_dim", lambda self: dim)


def _norm_corpus():
    return [
        {"chunk_id": "c", "text_raw": "first raw", "text_norm": "first norm"},
        {"chunk_id": "x", "text_raw": "other raw", "text_norm": "other norm"},
        {"chunk_id": "c", "text_raw": "first raw", "text_norm": "first norm"},
        {"chunk_id": "y", "text_raw": "third raw", "text_norm": "third norm"},
    ]


def _build_fixture_index(tmp_path, monkeypatch, corpus=None, weights=b"encoder-bytes-v1"):
    import pandas as pd

    corpus = corpus if corpus is not None else _norm_corpus()
    corpus_path = tmp_path / "chunks.parquet"
    pd.DataFrame(corpus).to_parquet(corpus_path)
    enc = _mock_encoder_dir(tmp_path, weights=weights)
    _stub_encode_texts(monkeypatch)
    out = tmp_path / "built"
    manifest = build_verified_index(
        corpus_path=str(corpus_path), out_dir=str(out),
        model_id=str(enc), revision="d" * 40,
        batch_size=4, device="cpu", dtype="float32",
    )
    return out, corpus_path, enc, manifest


def test_builder_output_strict_reselect_without_rebuild_loop(tmp_path, monkeypatch):
    import json

    out, corpus_path, enc, manifest = _build_fixture_index(tmp_path, monkeypatch)
    assert manifest["encoder_weights_sha256"] == hash_encoder_weights_dir(str(enc))
    assert manifest["preprocessing"]["text_field"] == "text_norm"
    saved = json.loads((out / "dense_manifest.json").read_text(encoding="utf-8"))
    assert saved["encoder_weights_sha256"] == manifest["encoder_weights_sha256"]
    assert saved["embedding_order_sha256"] and saved["embeddings_sha256"]
    expected = {
        "model_id": str(enc),
        "revision": "d" * 40,
        "encoder_weights_sha256": manifest["encoder_weights_sha256"],
        "preprocessing": manifest["preprocessing"],
    }
    first = select_verified_dense_index([str(out)], str(corpus_path), expected)
    assert first["action"] == "reused"
    # No rebuild loop: a second selection reuses the same output again.
    second = select_verified_dense_index([str(out)], str(corpus_path), expected)
    assert second["action"] == "reused"
    loaded = DenseRetriever.load_index(
        str(out), corpus_path=str(corpus_path),
        expected_model_name=str(enc),
        expected_revision="d" * 40,
        expected_encoder_weights_sha256=manifest["encoder_weights_sha256"],
        expected_preprocessing=manifest["preprocessing"],
        strict_identity=True,
    )
    assert loaded.corpus_embeddings.shape[0] == 4


def test_builder_binds_text_norm_only_edits(tmp_path, monkeypatch):
    import pandas as pd

    out, corpus_path, enc, manifest = _build_fixture_index(tmp_path, monkeypatch)
    expected = {
        "model_id": str(enc),
        "revision": "d" * 40,
        "encoder_weights_sha256": manifest["encoder_weights_sha256"],
        "preprocessing": manifest["preprocessing"],
    }
    edited = _norm_corpus()
    edited[1] = dict(edited[1], text_norm="other norm EDITED")
    pd.DataFrame(edited).to_parquet(corpus_path)
    sel = select_verified_dense_index([str(out)], str(corpus_path), expected)
    assert sel["action"] == "rebuild_needed"


def test_strict_rejects_missing_embedding_order_sha(tmp_path, monkeypatch):
    import json

    out, corpus_path, enc, manifest = _build_fixture_index(tmp_path, monkeypatch)
    saved_path = out / "dense_manifest.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    del saved["embeddings_sha256"]
    del saved["embedding_order_sha256"]
    saved_path.write_text(json.dumps(saved), encoding="utf-8")
    expected = {
        "model_id": str(enc),
        "revision": "d" * 40,
        "encoder_weights_sha256": manifest["encoder_weights_sha256"],
        "preprocessing": manifest["preprocessing"],
    }
    assert select_verified_dense_index([str(out)], str(corpus_path), expected)["action"] == "rebuild_needed"
    with pytest.raises(ValueError, match="missing identity"):
        DenseRetriever.load_index(
            str(out), corpus_path=str(corpus_path),
            expected_model_name=str(enc), strict_identity=True,
        )


def test_builder_output_rejects_tokenizer_and_weights_drift(tmp_path, monkeypatch):
    out, corpus_path, enc, manifest = _build_fixture_index(tmp_path, monkeypatch)
    base = {
        "model_id": str(enc),
        "revision": "d" * 40,
        "encoder_weights_sha256": manifest["encoder_weights_sha256"],
        "preprocessing": manifest["preprocessing"],
    }
    drifted_tok = dict(base, preprocessing=dict(base["preprocessing"], vietnamese_tokenized=False))
    assert select_verified_dense_index([str(out)], str(corpus_path), drifted_tok)["action"] == "rebuild_needed"
    drifted_w = dict(base, encoder_weights_sha256="ff" * 32)
    assert select_verified_dense_index([str(out)], str(corpus_path), drifted_w)["action"] == "rebuild_needed"


def test_promotion_failure_preserves_old_index(tmp_path):
    old = tmp_path / "index"
    old.mkdir()
    (old / "embeddings.npy").write_bytes(b"old-index-bytes")
    (old / "dense_manifest.json").write_text('{"v": 1}', encoding="utf-8")
    new_tmp = tmp_path / "tmp-new"
    new_tmp.mkdir()
    (new_tmp / "embeddings.npy").write_bytes(b"new-index-bytes")

    _promote_index(new_tmp, old)
    assert (old / "embeddings.npy").read_bytes() == b"new-index-bytes"
    assert not (old.parent / "index.backup").exists()

    old2 = tmp_path / "index2"
    old2.mkdir()
    (old2 / "dense_manifest.json").write_text('{"v": 1}', encoding="utf-8")
    new_tmp2 = tmp_path / "tmp-new2"
    new_tmp2.mkdir()
    (new_tmp2 / "x").write_bytes(b"x")
    import pathlib as _pl

    real_rename = _pl.Path.rename

    def _fail_once(self, target):
        raise OSError("simulated promotion failure")

    _pl.Path.rename = _fail_once
    try:
        with pytest.raises(OSError, match="simulated promotion failure"):
            _promote_index(new_tmp2, old2)
    finally:
        _pl.Path.rename = real_rename
    # Old index survived the failed promotion.
    assert (old2 / "dense_manifest.json").read_text(encoding="utf-8") == '{"v": 1}'


def test_resolve_encode_texts_matches_fit_contract():
    field, texts = resolve_encode_texts(_norm_corpus())
    assert field == "text_norm" and texts[0] == "first norm"
    field_raw, texts_raw = resolve_encode_texts(_strict_corpus())
    assert field_raw == "text_raw" and texts_raw[0] == "first"
