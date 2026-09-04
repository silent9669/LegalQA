"""Generator training configuration and strict production validation for V16."""

from dataclasses import dataclass, field
from typing import List, Tuple

APPROVED_TARGET_MODULES: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class GeneratorTrainConfig:
    """Immutable configuration contract for QLoRA generator training."""

    model_id: str
    max_seq_len: int = 2048
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = APPROVED_TARGET_MODULES
    activation_offloading: bool = True
    use_liger_fused_ce: bool = True
    device: str = "cuda:0"
    quantization: str = "4bit_nf4"
    double_quant: bool = True
    compute_dtype: str = "float16"
    optimizer: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    completion_only_loss: bool = True
    trainer_n_gpu: int = 1


def validate_generator_config_for_profile(config: GeneratorTrainConfig, profile: str) -> None:
    """Validate generator configuration against strict production requirements.

    Production profiles ('final_train_and_submit', 'screen_fold0', probes) forbid speculative
    capacity reduction or changing approved training hyperparameters.
    """
    strict_profiles = {
        "final_train_and_submit",
        "screen_fold0",
        "generator_probe_worstcase",
        "generator_probe_endurance",
    }

    if profile in strict_profiles:
        if config.max_seq_len != 2048:
            raise ValueError(
                f"Production profile '{profile}' requires max_seq_len=2048, got {config.max_seq_len}"
            )
        if config.lora_r != 16:
            raise ValueError(
                f"Production profile '{profile}' requires lora_r=16, got {config.lora_r}"
            )
        if config.lora_alpha != 32:
            raise ValueError(
                f"Production profile '{profile}' requires lora_alpha=32, got {config.lora_alpha}"
            )
        if config.batch_size != 1:
            raise ValueError(
                f"Production profile '{profile}' requires batch_size=1, got {config.batch_size}"
            )
        if config.grad_accum != 8:
            raise ValueError(
                f"Production profile '{profile}' requires grad_accum=8, got {config.grad_accum}"
            )
        if not config.activation_offloading:
            raise ValueError(
                f"Production profile '{profile}' requires activation_offloading=True to prevent OOM"
            )
        if not config.use_liger_fused_ce:
            raise ValueError(
                f"Production profile '{profile}' requires use_liger_fused_ce=True (fused-linear CE)"
            )
        if config.device != "cuda:0":
            raise ValueError(
                f"Production profile '{profile}' requires generator on device='cuda:0', got {config.device}"
            )
        if config.trainer_n_gpu != 1:
            raise ValueError(
                f"Production profile '{profile}' requires trainer_n_gpu=1, got {config.trainer_n_gpu}"
            )
