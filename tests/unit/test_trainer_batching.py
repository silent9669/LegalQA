from src.task2.generation.config import GeneratorTrainConfig
from src.task2.generation.trainer import build_v16_sft_config


def test_sft_config_groups_by_length():
    args = build_v16_sft_config(
        GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct", device="cpu"),
        output_dir="/tmp/x",
        per_device_train_batch_size=4,
    )
    is_grouped = getattr(args, "group_by_length", False) or getattr(args, "train_sampling_strategy", None) == "group_by_length"
    assert is_grouped, "length grouping avoids padding every example to 2048"
    assert getattr(args, "packing", False) is False, "packing would break completion-only masking"
