import os
import tempfile
import pytest
import yaml
from pathlib import Path

from src.task2.config.loader import load_resolved_config
from src.task2.config.schema import ResolvedTask2Config, AlgorithmConfig, RuntimeConfig


def test_authoritative_config_files_exist():
    """Verify that canonical task2 config directory and authoritative YAMLs exist."""
    base_dir = Path("configs/task2")
    algo_path = base_dir / "algorithm.yaml"
    kaggle_path = base_dir / "runtime" / "kaggle_t4x2.yaml"
    colab_t4_path = base_dir / "runtime" / "colab_t4.yaml"
    colab_a100_path = base_dir / "runtime" / "colab_a100.yaml"

    assert algo_path.exists(), f"Missing {algo_path}"
    assert kaggle_path.exists(), f"Missing {kaggle_path}"
    assert colab_t4_path.exists(), f"Missing {colab_t4_path}"
    assert colab_a100_path.exists(), f"Missing {colab_a100_path}"


def test_load_resolved_config_kaggle_t4x2():
    """Test loading and resolving Kaggle T4x2 configuration."""
    cfg = load_resolved_config(
        algorithm_path="configs/task2/algorithm.yaml",
        runtime_path="configs/task2/runtime/kaggle_t4x2.yaml",
    )
    assert isinstance(cfg, ResolvedTask2Config)
    assert cfg.algorithm.models.generator.id == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.runtime.profile_name == "kaggle_t4x2"
    assert cfg.runtime.required_gpu_count == 2
    assert cfg.runtime.generator_runtime.per_device_train_batch_size == 1
    assert cfg.runtime.generator_runtime.gradient_accumulation_steps == 8
    assert cfg.algorithm.generator.effective_batch_size == 8
    assert cfg.algorithm.generator.lora_r == 16
    assert cfg.algorithm.generator.learning_rate == 1.0e-4

    # Hashes must be non-empty hex strings
    assert len(cfg.algorithm_sha256) == 64
    assert len(cfg.runtime_sha256) == 64
    assert len(cfg.bundle_sha256) == 64


def test_load_resolved_config_colab_a100():
    """Test loading and resolving Colab A100 configuration."""
    cfg = load_resolved_config(
        algorithm_path="configs/task2/algorithm.yaml",
        runtime_path="configs/task2/runtime/colab_a100.yaml",
    )
    assert cfg.runtime.profile_name == "colab_a100"
    assert cfg.runtime.required_gpu_count == 1
    assert cfg.runtime.generator_runtime.compute_dtype == "bfloat16"
    assert cfg.runtime.generator_runtime.per_device_train_batch_size == 4
    assert cfg.runtime.generator_runtime.gradient_accumulation_steps == 2
    assert cfg.algorithm.generator.effective_batch_size == 8

    # Final A100 training must have val_fold is None
    assert cfg.algorithm.final_training.val_fold is None
    assert cfg.algorithm.final_training.training_scope == "all_allowed_train"


def test_runtime_cannot_override_protected_algorithm_fields():
    """Verify that runtime profiles cannot override score-affecting algorithm settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        algo_file = Path(tmpdir) / "algo.yaml"
        runtime_file = Path(tmpdir) / "runtime.yaml"

        algo_content = {
            "schema_version": 1,
            "seed": 42,
            "models": {
                "generator": {"id": "Qwen/Qwen2.5-3B-Instruct", "revision_policy": "exact_commit"},
                "reranker": {"id": "BAAI/bge-reranker-v2-m3", "revision_policy": "exact_commit"},
                "dense": {"id": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", "revision_policy": "exact_commit"},
            },
            "generator": {
                "max_seq_len": 2048,
                "quantization": "4bit_nf4",
                "double_quant": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
                "learning_rate": 1.0e-4,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.05,
                "effective_batch_size": 8,
                "num_train_epochs": 3,
                "completion_only_loss": True,
                "use_liger_fused_ce": True,
                "gradient_checkpointing": True,
            },
            "final_training": {"training_scope": "all_allowed_train", "val_fold": None},
            "evaluation": {"primary_metric": "whitespace_meteor", "secondary_metric": "rouge_l"},
        }

        # Malicious runtime trying to override lora_r
        runtime_content = {
            "profile_name": "test_override",
            "required_gpu_count": 1,
            "required_gpu_name_contains": "T4",
            "devices": {"generator": "cuda:0", "retrieval": "cuda:0"},
            "generator_runtime": {
                "compute_dtype": "float16",
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "activation_offloading": True,
            },
            "lora_r": 64,  # ILLEGAL OVERRIDE
        }

        algo_file.write_text(yaml.dump(algo_content))
        runtime_file.write_text(yaml.dump(runtime_content))

        with pytest.raises(ValueError, match="Protected field or unknown runtime key: lora_r"):
            load_resolved_config(algo_file, runtime_file)


def test_effective_batch_size_invariant():
    """Verify that runtime per-device batch * grad_accum must match effective_batch_size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        algo_file = Path(tmpdir) / "algo.yaml"
        runtime_file = Path(tmpdir) / "runtime.yaml"

        algo_content = {
            "schema_version": 1,
            "seed": 42,
            "models": {
                "generator": {"id": "Qwen/Qwen2.5-3B-Instruct", "revision_policy": "exact_commit"},
                "reranker": {"id": "BAAI/bge-reranker-v2-m3", "revision_policy": "exact_commit"},
                "dense": {"id": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", "revision_policy": "exact_commit"},
            },
            "generator": {
                "max_seq_len": 2048,
                "quantization": "4bit_nf4",
                "double_quant": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
                "learning_rate": 1.0e-4,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.05,
                "effective_batch_size": 8,
                "num_train_epochs": 3,
                "completion_only_loss": True,
                "use_liger_fused_ce": True,
                "gradient_checkpointing": True,
            },
            "final_training": {"training_scope": "all_allowed_train", "val_fold": None},
            "evaluation": {"primary_metric": "whitespace_meteor", "secondary_metric": "rouge_l"},
        }

        # Runtime has per_device=2, grad_accum=2 -> effective=4 != 8
        runtime_content = {
            "profile_name": "test_mismatch",
            "required_gpu_count": 1,
            "required_gpu_name_contains": "T4",
            "devices": {"generator": "cuda:0", "retrieval": "cuda:0"},
            "generator_runtime": {
                "compute_dtype": "float16",
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "activation_offloading": True,
            },
        }

        algo_file.write_text(yaml.dump(algo_content))
        runtime_file.write_text(yaml.dump(runtime_content))

        with pytest.raises(ValueError, match="Effective batch size mismatch"):
            load_resolved_config(algo_file, runtime_file)


def test_unknown_keys_fail_closed():
    """Verify that unknown keys in algorithm YAML are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        algo_file = Path(tmpdir) / "algo.yaml"
        runtime_file = Path(tmpdir) / "runtime.yaml"

        algo_content = {
            "schema_version": 1,
            "seed": 42,
            "unexpected_secret_tuning_param": 999,  # UNKNOWN KEY
            "models": {
                "generator": {"id": "Qwen/Qwen2.5-3B-Instruct", "revision_policy": "exact_commit"},
                "reranker": {"id": "BAAI/bge-reranker-v2-m3", "revision_policy": "exact_commit"},
                "dense": {"id": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", "revision_policy": "exact_commit"},
            },
            "generator": {
                "max_seq_len": 2048,
                "quantization": "4bit_nf4",
                "double_quant": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
                "learning_rate": 1.0e-4,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.05,
                "effective_batch_size": 8,
                "num_train_epochs": 3,
                "completion_only_loss": True,
                "use_liger_fused_ce": True,
                "gradient_checkpointing": True,
            },
            "final_training": {"training_scope": "all_allowed_train", "val_fold": None},
            "evaluation": {"primary_metric": "whitespace_meteor", "secondary_metric": "rouge_l"},
        }

        runtime_content = {
            "profile_name": "test_ok",
            "required_gpu_count": 1,
            "required_gpu_name_contains": "T4",
            "devices": {"generator": "cuda:0", "retrieval": "cuda:0"},
            "generator_runtime": {
                "compute_dtype": "float16",
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "activation_offloading": True,
            },
        }

        algo_file.write_text(yaml.dump(algo_content))
        runtime_file.write_text(yaml.dump(runtime_content))

        with pytest.raises(ValueError, match="Unknown algorithm key"):
            load_resolved_config(algo_file, runtime_file)
