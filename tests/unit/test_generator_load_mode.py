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
