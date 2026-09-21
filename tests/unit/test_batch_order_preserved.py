from unittest.mock import MagicMock
from src.task2.generator import QwenGenerator


def test_length_sorted_generation_returns_original_order(monkeypatch):
    gen = QwenGenerator(runtime="torch")
    gen.model = MagicMock()
    gen.tokenizer = MagicMock()

    # Stub: echo the prompt so order is easily checked
    monkeypatch.setattr(gen, "_generate_texts", lambda batch, mnt: [f"ans::{p}" for p in batch])
    monkeypatch.setattr(gen, "format_instance_prompt", lambda q, ev: q)

    items = [("q" * n, "") for n in (5, 200, 1, 90, 40)]
    out = gen.generate_batch(items, max_new_tokens=8, batch_size=2)
    assert out == [f"ans::{q}" for q, _ in items]   # must match original order exactly
