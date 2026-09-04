"""QLoRA fine-tuning module with selective Liger fused-linear cross entropy (V16)."""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

try:
    from peft import LoraConfig
except ImportError:
    class LoraConfig:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    SFTConfig = None
    SFTTrainer = None

try:
    from datasets import Dataset as HFDataset
except ImportError:
    class HFDataset:  # type: ignore
        @classmethod
        def from_list(cls, data: list) -> Any:
            return data

from src.common.security import assert_no_secrets_in_workspace
from src.task2.generator import QwenGenerator
from src.task2.generation.config import GeneratorTrainConfig, validate_generator_config_for_profile
from src.task2.generation.dataset import (
    SFTExample,
    build_grounded_training_examples,
    select_worst_case_probe,
)
from src.task2.generation.memory import (
    cleanup_cuda_stage,
    snapshot_cuda_memory,
    TrainerMemoryCallback,
)
from src.task2.generation.liger_backend import (
    REQUIRED_LIGER_VERSION,
    build_liger_training_kwargs,
    validate_liger_environment,
    assert_loss_type_compatible,
)

logger = logging.getLogger(__name__)


def enforce_single_gpu_trainer_args(args: Any, device: str) -> None:
    """Enforce generator single-GPU training policy and prevent DataParallel allocation on secondary GPUs."""
    dev = str(device)
    if not dev.startswith("cuda"):
        return

    if dev != "cuda:0":
        raise RuntimeError(
            f"LegalQA QLoRA generator must strictly target cuda:0, got {dev!r}."
        )

    # Force TrainingArguments._setup_devices to run first
    _ = getattr(args, "device", None)

    if hasattr(args, "_n_gpu"):
        args._n_gpu = 1

    if hasattr(args, "n_gpu") and int(args.n_gpu) != 1:
        raise RuntimeError(
            f"Failed to force single-GPU QLoRA Trainer policy; Trainer reports n_gpu={args.n_gpu}."
        )


def build_v16_sft_config(config: GeneratorTrainConfig, **kwargs: Any) -> Any:
    """Construct SFTConfig requiring completion_only_loss, activation_offloading, and selective Liger fused-linear CE."""
    if SFTConfig is None:
        raise RuntimeError("TRL SFTConfig is not available in the current environment.")

    sig = inspect.signature(SFTConfig)
    if "completion_only_loss" not in sig.parameters:
        raise RuntimeError(
            "Installed TRL SFTConfig does not support completion_only_loss parameter. "
            "Refusing to train with changed/corrupted loss semantics."
        )

    config_kwargs = dict(kwargs)
    config_kwargs["completion_only_loss"] = config.completion_only_loss

    if "activation_offloading" in sig.parameters:
        config_kwargs["activation_offloading"] = config.activation_offloading

    # CRITICAL V16 rule: DO NOT set loss_type="chunked_nll" when use_liger_fused_ce is True
    if config.use_liger_fused_ce:
        if "loss_type" in config_kwargs:
            del config_kwargs["loss_type"]
        config_kwargs.update(build_liger_training_kwargs(enabled=True))
    elif "loss_type" in sig.parameters:
        config_kwargs["loss_type"] = "nll"

    # Set sequence length
    if "max_length" in sig.parameters:
        config_kwargs["max_length"] = config.max_seq_len
    elif "max_seq_length" in sig.parameters:
        config_kwargs["max_seq_length"] = config.max_seq_len
    else:
        config_kwargs["max_length"] = config.max_seq_len

    return SFTConfig(**config_kwargs)


def train_generator_qlora(
    *,
    model_name_or_path: str,
    qa_path: str,
    labels_path: str,
    chunks_path: str,
    output_dir: str,
    config: Optional[GeneratorTrainConfig] = None,
    val_fold: Optional[int] = 0,
    max_steps: Optional[int] = None,
    max_train_examples: Optional[int] = None,
    probe_mode: Optional[str] = None,
    device: str = "cuda:0",
    epochs: int = 1,
    fail_on_error: bool = True,
    seed: int = 42,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Train Qwen2.5-3B-Instruct with 4-bit NF4 QLoRA, selective Liger fused-linear CE, and strict validation (V16)."""
    assert_no_secrets_in_workspace(Path.cwd())

    if config is None:
        config = GeneratorTrainConfig(model_id=model_name_or_path, device=device)

    # 1. Validate configuration for the active execution profile
    profile_name = "generator_probe_worstcase" if probe_mode == "worst_case" else (
        "generator_probe_endurance" if probe_mode == "endurance" else "standard"
    )
    if probe_mode in ("worst_case", "endurance"):
        validate_generator_config_for_profile(config, profile=profile_name)

    # 2. Validate Liger environment if active
    if config.use_liger_fused_ce and device.startswith("cuda"):
        validate_liger_environment(strict=True)

    # 3. Clean up prior CUDA stage memory
    cleanup_cuda_stage(devices=(0, 1))

    # 4. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. Build SFT dataset
    print(f"\nBuilding SFT training examples from {qa_path} (excluding val_fold={val_fold})...")
    examples = build_grounded_training_examples(
        qa_path=qa_path,
        labels_path=labels_path,
        chunks_path=chunks_path,
        fold_to_exclude=val_fold,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        max_train_examples=max_train_examples,
        seed=seed,
    )

    if probe_mode == "worst_case":
        print(f"Applying worst-case probe selector (top total & completion lengths)...")
        examples = select_worst_case_probe(examples, n_total=12, n_completion=12)
        print(f"Selected {len(examples)} worst-case examples for probe.")

    if not examples:
        raise RuntimeError("No valid SFT examples constructed for training.")

    train_dataset = HFDataset.from_list(examples)

    # 6. Load base model with 4-bit NF4 quantization
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }

    if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = {"": device}
    else:
        model_kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    # 7. Configure LoRA adapter
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.target_modules),
    )

    # 8. Configure SFT training arguments
    sft_kwargs: Dict[str, Any] = {
        "output_dir": os.path.join(output_dir, "runs"),
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.grad_accum,
        "gradient_checkpointing": config.gradient_checkpointing,
        "optim": config.optimizer,
        "num_train_epochs": epochs,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": "none",
        "fp16": config.compute_dtype == "float16" and device.startswith("cuda"),
    }
    if max_steps is not None:
        sft_kwargs["max_steps"] = max_steps

    sft_args = build_v16_sft_config(config, **sft_kwargs)
    enforce_single_gpu_trainer_args(sft_args, device)

    # 9. Construct SFTTrainer with TrainerMemoryCallback
    memory_callback = TrainerMemoryCallback(log_every_n_steps=50)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[memory_callback],
    )

    if hasattr(trainer, "args") and hasattr(trainer.args, "n_gpu") and int(trainer.args.n_gpu) != 1 and device.startswith("cuda"):
        raise RuntimeError(
            f"FINAL_PIPELINE_ERROR: SFTTrainer altered single-GPU policy; n_gpu={trainer.args.n_gpu}"
        )

    # 10. Execute Training
    train_start = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    train_elapsed = time.perf_counter() - train_start

    optimizer_steps = (
        int(trainer.state.global_step)
        if hasattr(trainer, "state") and hasattr(trainer.state, "global_step")
        else (max_steps or 0)
    )
    seconds_per_step = round(train_elapsed / max(1, optimizer_steps), 2)

    # 11. Record VRAM telemetry
    vram_snap = snapshot_cuda_memory("after_training", devices=(0, 1))
    d0_stats = vram_snap.get("devices", {}).get(0, {})
    peak_vram_mb = d0_stats.get("max_allocated_mb", 0.0)
    peak_reserved_mb = d0_stats.get("reserved_mb", 0.0)

    # 12. Save Adapter
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(trainer, "model") and hasattr(trainer.model, "save_pretrained"):
        trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 13. Strict Reload and Generation Smoke Verification
    print(f"\nExecuting strict reload smoke verification for adapter at {output_dir}...")
    cleanup_cuda_stage(trainer, model, devices=(0, 1))

    reload_status = "pass"
    try:
        reloaded = QwenGenerator.load(
            model_path=model_name_or_path,
            adapter_path=output_dir,
            require_adapter=True,
            device=device,
        )
        sample_out = reloaded.generate(
            "Căn cứ Điều 1 Luật Dân sự, hãy cho biết hợp đồng là gì?",
            max_new_tokens=32,
        )
        if not sample_out or not sample_out.strip():
            raise RuntimeError("Reloaded model generated empty response.")
        print(f"Strict reload sample output: {sample_out[:100]}...")
        cleanup_cuda_stage(reloaded, devices=(0, 1))
    except Exception as e:
        reload_status = f"fail: {e}"
        msg = f"QLoRA adapter saved but failed strict reload verification: {e}"
        if fail_on_error:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}") from e
        logger.error(msg)

    # 14. Write and Return Provenance Manifest
    manifest = {
        "runtime_api_version": 16,
        "backend": "liger_fused_linear_ce",
        "liger_version": REQUIRED_LIGER_VERSION,
        "model": model_name_or_path,
        "max_seq_len": config.max_seq_len,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "activation_offloading": config.activation_offloading,
        "trainer_n_gpu": 1,
        "probe_mode": probe_mode,
        "dataset_size": len(train_dataset),
        "optimizer_steps": optimizer_steps,
        "peak_vram_mb": peak_vram_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "seconds_per_optimizer_step": seconds_per_step,
        "strict_reload": reload_status,
    }

    manifest_path = os.path.join(output_dir, "generator_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerator training complete. Manifest saved to {manifest_path}")
    return manifest
