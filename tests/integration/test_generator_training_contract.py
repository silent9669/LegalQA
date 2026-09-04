from unittest.mock import MagicMock, patch
import pytest

from src.task2.generation.config import GeneratorTrainConfig
from src.task2.generation.trainer import (
    build_v16_sft_config,
    enforce_single_gpu_trainer_args,
    train_generator_qlora,
)


def test_build_v16_sft_config_parameters():
    cfg = GeneratorTrainConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        max_seq_len=2048,
        activation_offloading=True,
        use_liger_fused_ce=True,
    )

    class DummySFTConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    with patch("src.task2.generation.trainer.SFTConfig", DummySFTConfig):
        with patch("inspect.signature") as mock_sig:
            mock_param_names = {
                "completion_only_loss",
                "loss_type",
                "activation_offloading",
                "max_length",
                "use_liger_kernel",
                "liger_kernel_config",
                "output_dir",
            }
            mock_params = {name: MagicMock() for name in mock_param_names}
            mock_sig.return_value.parameters = mock_params

            sft_args = build_v16_sft_config(cfg, output_dir="/tmp/dummy")
            call_kwargs = sft_args.kwargs

            assert call_kwargs["completion_only_loss"] is True
            assert call_kwargs["activation_offloading"] is True
            assert call_kwargs["max_length"] == 2048
            assert call_kwargs.get("use_liger_kernel") is True
            assert call_kwargs.get("liger_kernel_config", {}).get("fused_linear_cross_entropy") is True
            # TRL chunked_nll must NOT be present
            assert call_kwargs.get("loss_type") != "chunked_nll"


def test_enforce_single_gpu_trainer_args_cuda0_enforcement():
    mock_args = MagicMock()
    mock_args.device = "cuda:0"
    mock_args._n_gpu = 2
    mock_args.n_gpu = 1

    # cuda:0 is valid and forces _n_gpu=1
    enforce_single_gpu_trainer_args(mock_args, "cuda:0")
    assert mock_args._n_gpu == 1

    # Other cuda devices must be rejected
    with pytest.raises(RuntimeError, match="cuda:0"):
        enforce_single_gpu_trainer_args(mock_args, "cuda:1")


def test_train_generator_qlora_dry_mock_pipeline(tmp_path):
    output_dir = str(tmp_path / "checkpoint")
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct")

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_model = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.state.global_step = 3
    mock_trainer.args.n_gpu = 1

    with patch("src.task2.generation.trainer.AutoTokenizer") as mock_auto_tok, \
         patch("src.task2.generation.trainer.AutoModelForCausalLM") as mock_auto_model, \
         patch("src.task2.generation.trainer.SFTTrainer", return_value=mock_trainer), \
         patch("src.task2.generation.trainer.build_v16_sft_config", return_value=MagicMock()), \
         patch("src.task2.generation.trainer.build_grounded_training_examples", return_value=[
             {"prompt": "p", "completion": "c", "text": "p c", "qa_id": "1", "total_tokens": 100, "completion_tokens": 20}
         ]), \
         patch("src.task2.generation.trainer.QwenGenerator.load") as mock_load:

        mock_auto_tok.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_reloaded = MagicMock()
        mock_reloaded.generate.return_value = "Non-empty model response"
        mock_load.return_value = mock_reloaded

        res = train_generator_qlora(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            qa_path="dummy_qa.parquet",
            labels_path="dummy_labels.parquet",
            chunks_path="dummy_chunks.parquet",
            output_dir=output_dir,
            config=cfg,
            max_steps=3,
            probe_mode="worst_case",
            device="cpu",  # testing on CPU mock
        )

        assert res["runtime_api_version"] == 16
        assert res["backend"] == "liger_fused_linear_ce"
        assert res["liger_version"] == "0.8.2"
        assert res["strict_reload"] == "pass"
