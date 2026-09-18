import os
import pytest
from unittest.mock import patch, MagicMock
from src.task2.generation.config import GeneratorTrainConfig
from src.task2.generation.trainer import train_generator_qlora


@pytest.fixture(autouse=True)
def mock_secret_scan():
    with patch("src.task2.generation.trainer.assert_no_secrets_in_workspace"):
        yield


def test_screen_fold0_rejects_altered_max_seq_len():
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct", max_seq_len=1024, device="cuda:0")
    with pytest.raises(ValueError, match="requires max_seq_len in"):
        train_generator_qlora(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            qa_path="mock",
            labels_path="mock",
            chunks_path="mock",
            output_dir="mock",
            config=cfg,
            execution_profile="screen_fold0",
        )


def test_final_train_rejects_disabled_activation_offloading():
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct", activation_offloading=False, device="cuda:0")
    with pytest.raises(ValueError, match="requires activation_offloading=True"):
        train_generator_qlora(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            qa_path="mock",
            labels_path="mock",
            chunks_path="mock",
            output_dir="mock",
            config=cfg,
            execution_profile="final_train_and_submit",
        )


def test_final_train_rejects_disabled_liger():
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct", use_liger_fused_ce=False, device="cuda:0")
    with pytest.raises(ValueError, match="requires use_liger_fused_ce=True"):
        train_generator_qlora(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            qa_path="mock",
            labels_path="mock",
            chunks_path="mock",
            output_dir="mock",
            config=cfg,
            execution_profile="final_train_and_submit",
        )


def test_final_train_rejects_wrong_device():
    cfg = GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct", device="cuda:1")
    with pytest.raises(ValueError, match="requires generator on device='cuda:0'"):
        train_generator_qlora(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            qa_path="mock",
            labels_path="mock",
            chunks_path="mock",
            output_dir="mock",
            config=cfg,
            execution_profile="final_train_and_submit",
        )


def test_standard_profile_does_not_enforce_strict_checks(tmp_path):
    cfg = GeneratorTrainConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        max_seq_len=512,
        device="cpu",
        activation_offloading=False,
        use_liger_fused_ce=False,
    )
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]

    mock_model = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.state.global_step = 1
    mock_trainer.args.n_gpu = 0

    mock_reloaded = MagicMock()
    mock_reloaded.generate.return_value = "Response"

    with patch("src.task2.generation.trainer.AutoTokenizer") as mock_auto_tok, \
         patch("src.task2.generation.trainer.AutoModelForCausalLM") as mock_auto_model, \
         patch("src.task2.generation.trainer.SFTTrainer", return_value=mock_trainer), \
         patch("src.task2.generation.trainer.build_v16_sft_config", return_value=MagicMock()), \
         patch("src.task2.generation.trainer.build_grounded_training_examples", return_value=[
             {"prompt": "p", "completion": "c", "text": "p c", "qa_id": "1", "total_tokens": 10, "completion_tokens": 5}
         ]), \
         patch("src.task2.generation.trainer.QwenGenerator.load", return_value=mock_reloaded):

        mock_auto_tok.from_pretrained.return_value = mock_tok
        mock_auto_model.from_pretrained.return_value = mock_model

        res = train_generator_qlora(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            qa_path="dummy_qa.parquet",
            labels_path="dummy_labels.parquet",
            chunks_path="dummy_chunks.parquet",
            output_dir=str(tmp_path / "ckpt"),
            config=cfg,
            execution_profile="standard",
            device="cpu",
        )
        assert res["runtime_api_version"] == 16
        assert res["strict_reload"] == "pass"
