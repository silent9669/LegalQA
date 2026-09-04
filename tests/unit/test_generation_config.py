import pytest
from src.task2.generation.config import (
    GeneratorTrainConfig,
    APPROVED_TARGET_MODULES,
    validate_generator_config_for_profile,
)


def test_generator_config_defaults():
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct")
    assert cfg.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.max_seq_len == 2048
    assert cfg.batch_size == 1
    assert cfg.grad_accum == 8
    assert cfg.learning_rate == 1e-4
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.activation_offloading is True
    assert cfg.use_liger_fused_ce is True
    assert cfg.device == "cuda:0"
    assert cfg.target_modules == APPROVED_TARGET_MODULES


def test_generator_config_is_immutable():
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct")
    with pytest.raises(Exception):
        cfg.max_seq_len = 1024  # Immutable dataclass


def test_validate_generator_config_accepts_valid_production():
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct")
    # Should not raise for production or screen profiles
    validate_generator_config_for_profile(cfg, profile="final_train_and_submit")
    validate_generator_config_for_profile(cfg, profile="screen_fold0")
    validate_generator_config_for_profile(cfg, profile="generator_probe_worstcase")
    validate_generator_config_for_profile(cfg, profile="generator_probe_endurance")


def test_validate_generator_config_rejects_altered_seq_len_in_production():
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct", max_seq_len=1024)
    with pytest.raises(ValueError, match="max_seq_len"):
        validate_generator_config_for_profile(cfg, profile="final_train_and_submit")


def test_validate_generator_config_rejects_altered_lora_rank_in_production():
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct", lora_r=8)
    with pytest.raises(ValueError, match="lora_r"):
        validate_generator_config_for_profile(cfg, profile="final_train_and_submit")


def test_validate_generator_config_rejects_disabled_liger_in_production():
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct", use_liger_fused_ce=False)
    with pytest.raises(ValueError, match="use_liger_fused_ce"):
        validate_generator_config_for_profile(cfg, profile="final_train_and_submit")


def test_validate_generator_config_rejects_disabled_activation_offloading():
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct", activation_offloading=False)
    with pytest.raises(ValueError, match="activation_offloading"):
        validate_generator_config_for_profile(cfg, profile="final_train_and_submit")


def test_validate_generator_config_rejects_wrong_device_in_production():
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct", device="cuda:1")
    with pytest.raises(ValueError, match="device"):
        validate_generator_config_for_profile(cfg, profile="final_train_and_submit")
