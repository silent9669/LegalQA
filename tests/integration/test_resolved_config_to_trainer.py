import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import yaml

from src.task2.config.loader import load_resolved_config
from src.task2.generation.trainer import train_generator_qlora


def test_resolved_config_propagates_to_trainer_and_lora(tmp_path):
    """Verify that YAML LoRA rank, LR, epochs, batch, and dtype reach trainer and PEFT."""
    algo_dict = {
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
            "lora_r": 24,
            "lora_alpha": 48,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "learning_rate": 2.5e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "effective_batch_size": 8,
            "num_train_epochs": 4,
            "completion_only_loss": True,
            "use_liger_fused_ce": True,
            "gradient_checkpointing": True,
        },
        "final_training": {"training_scope": "all_allowed_train", "val_fold": None},
        "evaluation": {"primary_metric": "whitespace_meteor", "secondary_metric": "rouge_l"},
    }

    runtime_dict = {
        "profile_name": "colab_a100",
        "required_gpu_count": 1,
        "required_gpu_name_contains": "A100",
        "devices": {"generator": "cuda:0", "retrieval": "cuda:0"},
        "generator_runtime": {
            "compute_dtype": "bfloat16",
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "activation_offloading": False,
        },
        "a100_micro_probe": {"optimizer_steps": 2},
        "production": {"full_train": True},
    }

    algo_file = tmp_path / "algo.yaml"
    runtime_file = tmp_path / "runtime.yaml"
    algo_file.write_text(yaml.dump(algo_dict))
    runtime_file.write_text(yaml.dump(runtime_dict))

    resolved_cfg = load_resolved_config(algo_file, runtime_file)

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_model = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.state.global_step = 2
    mock_trainer.args.n_gpu = 1

    captured_peft = {}
    captured_sft_args = {}

    def fake_sft_trainer(model, args, train_dataset, peft_config, **kwargs):
        captured_peft["config"] = peft_config
        captured_sft_args["args"] = args
        return mock_trainer

    with patch("src.task2.generation.trainer.AutoTokenizer") as mock_tok, \
         patch("src.task2.generation.trainer.AutoModelForCausalLM") as mock_causal, \
         patch("src.task2.generation.trainer.SFTTrainer", side_effect=fake_sft_trainer), \
         patch("src.task2.generation.trainer.build_grounded_training_examples", return_value=[
             {"prompt": "p", "completion": "c", "text": "p c", "qa_id": "1", "total_tokens": 100, "completion_tokens": 20}
         ]) as mock_build_examples, \
         patch("src.task2.generation.trainer.QwenGenerator.load") as mock_load:

        mock_tok.from_pretrained.return_value = mock_tokenizer
        mock_causal.from_pretrained.return_value = mock_model
        mock_reloaded = MagicMock()
        mock_reloaded.generate.return_value = "Mock response"
        mock_load.return_value = mock_reloaded

        output_dir = str(tmp_path / "output")

        res = train_generator_qlora(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            qa_path="dummy_qa.parquet",
            labels_path="dummy_labels.parquet",
            chunks_path="dummy_chunks.parquet",
            output_dir=output_dir,
            resolved_config=resolved_cfg,
            max_steps=2,
            device="cpu",
        )

        # 1. Assert PEFT config values came directly from authoritative YAML
        peft_cfg = captured_peft["config"]
        assert peft_cfg.r == 24
        assert peft_cfg.lora_alpha == 48
        assert peft_cfg.lora_dropout == 0.05
        assert set(peft_cfg.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}

        # 2. Assert SFTTrainer args came directly from authoritative YAML + runtime
        sft_args = captured_sft_args["args"]
        assert sft_args.learning_rate == 2.5e-4
        assert sft_args.num_train_epochs == 4
        assert sft_args.per_device_train_batch_size == 4
        assert sft_args.gradient_accumulation_steps == 2

        # 3. Assert val_fold was passed as None (all allowed train data)
        mock_build_examples.assert_called_once()
        _, call_kwargs = mock_build_examples.call_args
        assert call_kwargs.get("fold_to_exclude") is None
