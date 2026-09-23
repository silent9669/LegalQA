"""Qwen base-revision pinning regressions (review blocker B4, CPU mocks only)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.task2.generator import QwenGenerator, _validate_base_revision


def test_validate_base_revision_accepts_only_immutable():
    assert _validate_base_revision("A" * 40) == "a" * 40
    assert _validate_base_revision(None) is None
    assert _validate_base_revision("") is None
    with pytest.raises(ValueError, match="40-hex"):
        _validate_base_revision("main")
    with pytest.raises(ValueError, match="40-hex"):
        _validate_base_revision("abc123")


def _mock_torch_stack(monkeypatch):
    monkeypatch.setattr("src.task2.generator.torch", MagicMock())
    import src.task2.generator as gen_mod

    gen_mod.torch.cuda.is_available.return_value = False
    gen_mod.torch.backends.mps.is_available.return_value = False


def test_generator_load_pins_hub_revision(tmp_path, monkeypatch):
    _mock_torch_stack(monkeypatch)
    import src.task2.generator as gen_mod

    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<eos>"
    model = MagicMock()
    with patch.object(gen_mod, "AutoTokenizer") as mock_tok, \
        patch.object(gen_mod, "AutoModelForCausalLM") as mock_model:
        mock_tok.from_pretrained.return_value = tok
        mock_model.from_pretrained.return_value = model
        gen = QwenGenerator.load(
            model_path="Qwen/Qwen2.5-3B-Instruct",
            device="cpu",
            runtime="torch",
            base_revision="b" * 40,
        )
    assert gen.base_revision == "b" * 40
    assert mock_tok.from_pretrained.call_args.kwargs.get("revision") == "b" * 40
    assert mock_model.from_pretrained.call_args.kwargs.get("revision") == "b" * 40


def test_generator_load_skips_revision_for_local_weights(tmp_path, monkeypatch):
    _mock_torch_stack(monkeypatch)
    import src.task2.generator as gen_mod

    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<eos>"
    model = MagicMock()
    with patch.object(gen_mod, "AutoTokenizer") as mock_tok, \
        patch.object(gen_mod, "AutoModelForCausalLM") as mock_model:
        mock_tok.from_pretrained.return_value = tok
        mock_model.from_pretrained.return_value = model
        QwenGenerator.load(
            model_path=str(tmp_path),
            device="cpu",
            runtime="torch",
            base_revision="b" * 40,
        )
    assert "revision" not in mock_tok.from_pretrained.call_args.kwargs
    assert "revision" not in mock_model.from_pretrained.call_args.kwargs


def test_generator_load_rejects_floating_revision():
    with pytest.raises(ValueError, match="40-hex"):
        QwenGenerator.load(model_path="Qwen/Qwen2.5-3B-Instruct", base_revision="main")


def test_trainer_threads_pinned_base_revision(tmp_path):
    from src.task2.generation.trainer import train_generator_qlora

    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_model = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.state.global_step = 2
    mock_reloaded = MagicMock()
    mock_reloaded.generate.return_value = "Verified"

    with patch("src.task2.generation.trainer.AutoTokenizer.from_pretrained", return_value=mock_tok) as tok_ctor, \
        patch("src.task2.generation.trainer.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as model_ctor, \
        patch("src.task2.generation.trainer.SFTTrainer", return_value=mock_trainer), \
        patch("src.task2.generation.trainer.build_v16_sft_config", return_value=MagicMock()), \
        patch("src.task2.generation.trainer.build_grounded_training_examples", return_value=[
            {"prompt": "p", "completion": "c", "text": "p c", "qa_id": "1", "total_tokens": 10, "completion_tokens": 5}
        ]), \
        patch("src.task2.generation.trainer.QwenGenerator.load", return_value=mock_reloaded):
        train_generator_qlora(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            qa_path="dummy_qa.parquet",
            labels_path="dummy_labels.parquet",
            chunks_path="dummy_chunks.parquet",
            output_dir=str(tmp_path / "probe_out"),
            max_steps=2,
            probe_mode="worst_case",
            execution_profile="modal_a100",
            device="cpu",
            base_revision="c" * 40,
        )
    assert tok_ctor.call_args.kwargs.get("revision") == "c" * 40
    assert model_ctor.call_args.kwargs.get("revision") == "c" * 40


def test_runner_pinned_base_revision_helper():
    from src.task2.pipeline.runner import _pinned_base_revision

    assert _pinned_base_revision(None) is None
    fake = MagicMock()
    fake.algorithm.models.generator.revision = "AA8E72537993BA99E69DFAAFA59ED015B17504D1"
    assert _pinned_base_revision(fake) == "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    fake.algorithm.models.generator.revision = "main"
    assert _pinned_base_revision(fake) is None
