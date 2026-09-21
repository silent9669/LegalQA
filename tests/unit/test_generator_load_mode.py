import pytest
from src.task2.config.loader import load_resolved_config


def test_modal_a100_declares_bf16_inference():
    cfg = load_resolved_config(
        "configs/task2/algorithm.yaml",
        "configs/task2/runtime/modal_a100.yaml",
    )
    assert cfg.runtime.inference.generator_load_mode == "bfloat16"
    assert cfg.runtime.inference.merge_adapter is True
    assert cfg.runtime.inference.max_new_tokens == 1536
    assert cfg.runtime.inference.best_fixed_candidate == "dual_assembled"


def test_default_load_mode_is_nf4_for_small_gpus():
    cfg = load_resolved_config(
        "configs/task2/algorithm.yaml",
        "configs/task2/runtime/kaggle_t4x2.yaml",
    )
    assert cfg.runtime.inference.generator_load_mode == "nf4"
    assert cfg.runtime.inference.merge_adapter is False


def test_generator_load_rejects_unknown_mode():
    from src.task2.generator import QwenGenerator
    with pytest.raises(ValueError, match="unknown generator load mode"):
        QwenGenerator.load(model_path="Qwen/Qwen2.5-3B-Instruct", load_mode="int3")


def test_generator_load_allows_merged_adapter_when_require_adapter_is_true(monkeypatch):
    from unittest.mock import MagicMock
    from src.task2.generator import QwenGenerator
    import src.task2.generator as gen_mod

    mock_base = MagicMock()
    mock_peft = MagicMock()
    mock_merged = MagicMock()
    mock_peft.merge_and_unload.return_value = mock_merged

    monkeypatch.setattr(gen_mod, "AutoTokenizer", MagicMock())
    monkeypatch.setattr(gen_mod, "AutoModelForCausalLM", MagicMock(from_pretrained=MagicMock(return_value=mock_base)))
    monkeypatch.setattr(gen_mod, "PeftModel", MagicMock(from_pretrained=MagicMock(return_value=mock_peft)))
    monkeypatch.setattr(gen_mod, "is_peft_model", lambda m: m is mock_peft)

    import os
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    gen = QwenGenerator.load(
        model_path="Qwen/Qwen2.5-3B-Instruct",
        adapter_path="/fake/adapter",
        device="cpu",
        runtime="torch",
        load_mode="bfloat16",
        merge_adapter=True,
        require_adapter=True,
    )
    assert gen.model is mock_merged
