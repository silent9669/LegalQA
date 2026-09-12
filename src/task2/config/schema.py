"""Typed schemas for Task 2 algorithm and runtime configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelEntry:
    id: str
    revision_policy: str = "exact_commit"
    revision: Optional[str] = None


@dataclass(frozen=True)
class ModelsConfig:
    generator: ModelEntry
    reranker: ModelEntry
    dense: ModelEntry


@dataclass(frozen=True)
class GeneratorAlgorithmConfig:
    max_seq_len: int = 2048
    quantization: str = "4bit_nf4"
    double_quant: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    learning_rate: float = 1.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    effective_batch_size: int = 8
    num_train_epochs: int = 3
    completion_only_loss: bool = True
    use_liger_fused_ce: bool = True
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class FinalTrainingConfig:
    training_scope: str = "all_allowed_train"
    val_fold: Optional[int] = None


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str = "whitespace_meteor"
    secondary_metric: str = "rouge_l"


@dataclass(frozen=True)
class AlgorithmConfig:
    schema_version: int
    seed: int
    models: ModelsConfig
    generator: GeneratorAlgorithmConfig
    final_training: FinalTrainingConfig
    evaluation: EvaluationConfig

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratorRuntimeConfig:
    compute_dtype: str = "float16"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    activation_offloading: bool = True


@dataclass(frozen=True)
class SmokeConfig:
    worstcase_steps: Optional[int] = None
    endurance_steps: Optional[int] = None
    optimizer_steps: Optional[int] = None
    mini_eval_samples: Optional[int] = None


@dataclass(frozen=True)
class MicroProbeConfig:
    optimizer_steps: int = 2


@dataclass(frozen=True)
class ProductionRuntimeConfig:
    full_train: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    profile_name: str
    required_gpu_count: int
    required_gpu_name_contains: str
    devices: Dict[str, str]
    generator_runtime: GeneratorRuntimeConfig
    smoke: Optional[SmokeConfig] = None
    a100_micro_probe: Optional[MicroProbeConfig] = None
    production: Optional[ProductionRuntimeConfig] = None
    outputs: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedTask2Config:
    algorithm: AlgorithmConfig
    runtime: RuntimeConfig
    candidate_id: Optional[str] = None
    algorithm_sha256: str = ""
    runtime_sha256: str = ""
    bundle_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm.to_dict(),
            "runtime": self.runtime.to_dict(),
            "candidate_id": self.candidate_id,
            "algorithm_sha256": self.algorithm_sha256,
            "runtime_sha256": self.runtime_sha256,
            "bundle_sha256": self.bundle_sha256,
        }
