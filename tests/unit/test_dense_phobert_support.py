"""Unit tests for PhoBERT / encoder_ft_v2 support and dense error handling."""

import os
import json
import pytest
from src.common.dense import DenseRetriever


def test_needs_vietnamese_tokenization_detects_ft_and_phobert():
    ret_ft = DenseRetriever(model_name="runs/20260920-215402/encoder_ft_v2")
    assert ret_ft._needs_vietnamese_tokenization() is True

    ret_phobert = DenseRetriever(model_name="vinai/phobert-base")
    assert ret_phobert._needs_vietnamese_tokenization() is True

    ret_dek = DenseRetriever(model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2")
    assert ret_dek._needs_vietnamese_tokenization() is True

    ret_bge = DenseRetriever(model_name="BAAI/bge-m3")
    assert ret_bge._needs_vietnamese_tokenization() is False


def test_needs_vietnamese_tokenization_checks_local_config(tmp_path):
    model_dir = tmp_path / "custom_model"
    model_dir.mkdir()
    cfg_file = model_dir / "config.json"
    cfg_file.write_text(json.dumps({"tokenizer_class": "PhobertTokenizer", "model_type": "roberta"}))

    ret = DenseRetriever(model_name=str(model_dir))
    assert ret._needs_vietnamese_tokenization() is True


def test_encode_texts_fails_loud_when_sentence_transformers_missing(monkeypatch):
    import src.common.dense as dense_mod
    monkeypatch.setattr(dense_mod, "SentenceTransformer", None)

    ret = DenseRetriever(model_name="some-real-model")
    with pytest.raises(RuntimeError, match="sentence_transformers is missing"):
        ret.encode_texts(["văn bản pháp luật"])


def test_mock_model_still_works_for_testing():
    ret = DenseRetriever(model_name="mock")
    embs = ret.encode_texts(["văn bản pháp luật"])
    assert embs.shape == (1, 768)
