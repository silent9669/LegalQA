from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

from src.task2.config.loader import load_resolved_config
from src.task2.pipeline.profiles import resolve_execution_profile
from src.task2.generation.trainer import train_generator_qlora


def test_algorithm_yaml_declares_all_data_final_training():
    """Verify that algorithm.yaml explicitly specifies all_allowed_train with val_fold=None."""
    cfg = load_resolved_config(
        algorithm_path="configs/task2/algorithm.yaml",
        runtime_path="configs/task2/runtime/colab_a100.yaml",
    )
    assert cfg.algorithm.final_training.training_scope == "all_allowed_train"
    assert cfg.algorithm.final_training.val_fold is None


def test_colab_a100_execution_profile_has_no_val_fold():
    """Verify that colab_a100 profile executes with val_fold=None."""
    prof = resolve_execution_profile("colab_a100")
    assert prof.val_fold is None


def test_trainer_excludes_no_folds_when_val_fold_is_none(tmp_path):
    """Verify that train_generator_qlora passes fold_to_exclude=None to dataset builder."""
    cfg = load_resolved_config(
        algorithm_path="configs/task2/algorithm.yaml",
        runtime_path="configs/task2/runtime/colab_a100.yaml",
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_model = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.state.global_step = 2

    with patch("src.task2.generation.trainer.AutoTokenizer") as mock_tok, \
         patch("src.task2.generation.trainer.AutoModelForCausalLM") as mock_causal, \
         patch("src.task2.generation.trainer.SFTTrainer", return_value=mock_trainer), \
         patch("src.task2.generation.trainer.build_grounded_training_examples", return_value=[
             {"prompt": "p", "completion": "c", "text": "p c", "qa_id": "1", "total_tokens": 100, "completion_tokens": 20}
         ]) as mock_build, \
         patch("src.task2.generation.trainer.QwenGenerator.load") as mock_load:

        mock_tok.from_pretrained.return_value = mock_tokenizer
        mock_causal.from_pretrained.return_value = mock_model
        mock_reloaded = MagicMock()
        mock_reloaded.generate.return_value = "Non-empty"
        mock_load.return_value = mock_reloaded

        train_generator_qlora(
            model_name_or_path=cfg.algorithm.models.generator.id,
            qa_path="dummy_qa.parquet",
            labels_path="dummy_labels.parquet",
            chunks_path="dummy_chunks.parquet",
            output_dir=str(tmp_path / "out"),
            resolved_config=cfg,
            max_steps=2,
            device="cpu",
        )

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert kwargs.get("fold_to_exclude") is None, (
            f"Final A100 training must not exclude fold 0; expected fold_to_exclude=None, got {kwargs.get('fold_to_exclude')}"
        )
