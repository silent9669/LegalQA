"""Unit tests verifying order-preserving CUDA OOM-halving backoff in generator and reranker."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.task2.generator import QwenGenerator
from src.common.reranker import BGEReranker


def test_generator_generate_batch_oom_halving_preserves_order():
    gen = QwenGenerator(model_path="mock", device="cpu", runtime="torch")
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()

    calls = []

    def mock_gen_texts(prompts, max_new_tokens):
        calls.append(len(prompts))
        if len(prompts) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate...")
        return [f"ans_{p}" for p in prompts]

    gen._generate_texts = mock_gen_texts
    items = [(f"q{i}", f"ev{i}") for i in range(6)]
    res = gen.generate_batch(items, batch_size=4)
    expected = [f"ans_{gen.format_instance_prompt(q, ev)}" for q, ev in items]
    assert res == expected
    assert len(res) == 6
    assert calls == [4, 2, 2, 2]


def test_reranker_predict_oom_halving_preserves_order():
    reranker = BGEReranker(model_name="mock")
    reranker.model = MagicMock()

    def mock_predict(pairs, batch_size, show_progress_bar=False):
        if batch_size > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate...")
        return [float(i) for i in range(len(pairs))]

    reranker.model.predict = mock_predict
    pairs = [[f"q{i}", f"d{i}"] for i in range(7)]
    scores = reranker._predict_with_oom_backoff(pairs, batch_size=4)
    assert scores == [float(i) for i in range(7)]
