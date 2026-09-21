"""QLoRA fine-tuning module with selective Liger fused-linear cross entropy (V16)."""

from __future__ import annotations

import gc
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
from src.common.env_loader import load_environment
from src.task2.config.schema import ResolvedTask2Config
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

    # CRITICAL V16 rule: loss_type must be "nll", NEVER "chunked_nll" when Liger is active
    if "loss_type" in sig.parameters:
        config_kwargs["loss_type"] = "nll"

    if config.use_liger_fused_ce:
        config_kwargs.update(build_liger_training_kwargs(enabled=True))
    assert_loss_type_compatible(config_kwargs.get("loss_type"), config.use_liger_fused_ce)

    # CPU/GPU precision guards to prevent SFTConfig/TrainingArguments validation errors on CPU CI
    if "bf16" in sig.parameters and "bf16" not in config_kwargs:
        config_kwargs["bf16"] = (config.compute_dtype == "bfloat16" and config.device.startswith("cuda"))
    if "fp16" in sig.parameters and "fp16" not in config_kwargs:
        config_kwargs["fp16"] = (config.compute_dtype == "float16" and config.device.startswith("cuda"))
    if torch is not None and not torch.cuda.is_available() and "use_cpu" in sig.parameters:
        config_kwargs.setdefault("use_cpu", True)

    # Length grouping to avoid padding waste at max_seq_len
    if "train_sampling_strategy" in sig.parameters and "train_sampling_strategy" not in config_kwargs:
        config_kwargs["train_sampling_strategy"] = "group_by_length"
    elif "group_by_length" in sig.parameters and "group_by_length" not in config_kwargs:
        config_kwargs["group_by_length"] = True
    if "packing" in sig.parameters and "packing" not in config_kwargs:
        config_kwargs["packing"] = False

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
    resolved_config: Optional[ResolvedTask2Config] = None,
    val_fold: Optional[int] = 0,
    max_steps: Optional[int] = None,
    max_train_examples: Optional[int] = None,
    probe_mode: Optional[str] = None,
    device: str = "cuda:0",
    epochs: int = 1,
    fail_on_error: bool = True,
    seed: int = 42,
    resume_from_checkpoint: Optional[str] = None,
    execution_profile: Optional[str] = None,
    require_evidence: Optional[bool] = None,
) -> Dict[str, Any]:
    """Train Qwen2.5-3B-Instruct with 4-bit NF4 QLoRA, selective Liger fused-linear CE, and strict validation (V16)."""
    assert_no_secrets_in_workspace(Path.cwd())

    if resolved_config is not None:
        algo = resolved_config.algorithm
        rt = resolved_config.runtime
        target_dev = device if device != "cuda:0" else rt.devices.get("generator", device)
        device = target_dev
        from src.task2.training.context_builder import recipe_epochs, recipe_to_train_config

        config = recipe_to_train_config(resolved_config, device=target_dev)
        epochs = recipe_epochs(resolved_config)
        seed = algo.seed
        if val_fold == 0 and algo.final_training.val_fold is None and (execution_profile == "final_train_and_submit" or rt.production):
            val_fold = None
        elif resolved_config.algorithm.final_training.val_fold is None and "final" in (execution_profile or ""):
            val_fold = None
        elif algo.final_training.val_fold is None and val_fold == 0 and not probe_mode:
            val_fold = None

    if config is None:
        config = GeneratorTrainConfig(model_id=model_name_or_path, device=device)

    # 1. Validate configuration for the active execution profile
    profile_name = execution_profile or (
        resolved_config.runtime.profile_name if resolved_config is not None else (
            "generator_probe_worstcase" if probe_mode == "worst_case" else (
                "generator_probe_endurance" if probe_mode == "endurance" else "standard"
            )
        )
    )
    strict_profiles = {
        "final_train_and_submit",
        "screen_fold0",
        "generator_probe_worstcase",
        "generator_probe_endurance",
    }
    if profile_name in strict_profiles:
        validate_generator_config_for_profile(config, profile=profile_name)

    # 2. Validate Liger environment if active (fail loudly before model load, no fallback)
    if config.use_liger_fused_ce and device.startswith("cuda"):
        liger_status = validate_liger_environment(strict=True)
        assert liger_status.version == REQUIRED_LIGER_VERSION, f"Expected liger-kernel {REQUIRED_LIGER_VERSION}"
        assert liger_status.qwen2_patch_available, "Expected Qwen2 Liger patch available"
        assert liger_status.fused_linear_ce, "Expected Liger fused-linear CE available"
        assert config.use_liger_fused_ce is True, "Expected use_liger_fused_ce=True"
        assert config.trainer_n_gpu == 1, "Expected trainer_n_gpu=1"
        assert device == "cuda:0", f"Expected generator on cuda:0, got {device}"
        print(f"Liger-Kernel: {liger_status.version}")
        print("Qwen2 Liger patch: PASS")
        print("Liger fused-linear CE: PASS")
        print("use_liger_kernel=True")
        print("fused_linear_cross_entropy=True")
        print("loss_type=nll")
        print(f"target={device}")
        print(f"trainer_n_gpu={config.trainer_n_gpu}")

    # 3. Clean up prior CUDA stage memory and restore generator device context
    cleanup_cuda_stage(devices=(0, 1))
    if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 4. Load environment credentials and tokenizer
    load_environment()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        tok_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        **tok_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. Build SFT dataset
    print(f"\nBuilding SFT training examples from {qa_path} (excluding val_fold={val_fold})...")
    require_ev = (val_fold is None and probe_mode is None) if require_evidence is None else bool(require_evidence)
    raw_res = build_grounded_training_examples(
        qa_path=qa_path,
        labels_path=labels_path,
        chunks_path=chunks_path,
        fold_to_exclude=val_fold,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        max_train_examples=max_train_examples,
        seed=seed,
        require_evidence=require_ev,
        return_diagnostics=True,
    )
    if isinstance(raw_res, tuple) and len(raw_res) == 2:
        examples, diag = raw_res
    else:
        examples, diag = raw_res, {}
    if diag.get("dropped_no_evidence"):
        print(f"[+] Grounded SFT: dropped {diag['dropped_no_evidence']} evidence-free examples (kept {len(examples)}).")
    if diag.get("duplicate_qa_dropped"):
        print(f"[+] Grounded SFT: deduplicated {diag['duplicate_qa_dropped']} repeated QA IDs.")

    if probe_mode == "worst_case":
        print(f"Applying worst-case probe selector (top total & completion lengths)...")
        examples = select_worst_case_probe(examples, n_total=12, n_completion=12)
        print(f"Selected {len(examples)} worst-case examples for probe.")

    if not examples:
        raise RuntimeError("No valid SFT examples constructed for training.")

    train_dataset = HFDataset.from_list(examples)

    # 6. Load base model with 4-bit NF4 quantization and Liger Kernel patching
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if config.use_liger_fused_ce and device.startswith("cuda") and torch is not None and torch.cuda.is_available():
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            apply_liger_kernel_to_qwen2(
                rope=True,
                cross_entropy=False,
                fused_linear_cross_entropy=True,
                rms_norm=True,
                swiglu=True,
            )
            print("Successfully applied Liger Kernel monkey-patch to Qwen2 (fused-linear CE, RoPE, RMSNorm, SwiGLU).")
        except Exception as e:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: Failed to apply Liger Kernel to Qwen2: {e}") from e

    if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
        target_torch_dtype = torch.bfloat16 if config.compute_dtype == "bfloat16" else torch.float16
        if config.quantization == "4bit_nf4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=config.double_quant,
                bnb_4bit_compute_dtype=target_torch_dtype,
            )
            model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = {"": device}
        model_kwargs["dtype"] = target_torch_dtype
        model_kwargs["torch_dtype"] = target_torch_dtype
        model_kwargs["attn_implementation"] = "sdpa"
    else:
        model_kwargs["device_map"] = {"": device}

    if hf_token:
        model_kwargs["token"] = hf_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Prepare model for 4-bit k-bit training with gradient checkpointing
    if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            print("Successfully prepared model for k-bit training with gradient checkpointing (non-reentrant).")
        except Exception as e:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

    # Empty cache after base model loading
    if torch is not None and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

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
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": config.optimizer,
        "num_train_epochs": epochs,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 1,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": "none",
        "fp16": config.compute_dtype == "float16" and device.startswith("cuda"),
        "bf16": config.compute_dtype == "bfloat16" and device.startswith("cuda"),
    }
    if SFTConfig is not None:
        _sft_sig = inspect.signature(SFTConfig)
        if "train_sampling_strategy" in _sft_sig.parameters:
            sft_kwargs["train_sampling_strategy"] = "group_by_length"
        elif "group_by_length" in _sft_sig.parameters:
            sft_kwargs["group_by_length"] = True
        if "packing" in _sft_sig.parameters:
            sft_kwargs["packing"] = False
    if max_steps is not None:
        sft_kwargs["max_steps"] = max_steps

    # Optimize gradient accumulation for smoke/probe mode to avoid multi-batch gradient accumulation VRAM retention
    grad_accum_eff = (
        1 if (probe_mode or (execution_profile and "smoke" in execution_profile))
        else config.grad_accum
    )
    sft_kwargs["gradient_accumulation_steps"] = grad_accum_eff

    sft_args = build_v16_sft_config(config, **sft_kwargs)
    enforce_single_gpu_trainer_args(sft_args, device)

    # 9. Construct SFTTrainer with TrainerMemoryCallback
    log_freq = 1 if probe_mode else 25
    empty_freq = 1 if probe_mode else None
    memory_callback = TrainerMemoryCallback(log_every_n_steps=log_freq, empty_cache_every_n_steps=empty_freq)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[memory_callback],
    )

    # Verify completion-only loss masking
    if config.completion_only_loss and hasattr(trainer, "data_collator") and len(train_dataset) > 0:
        try:
            probe_batch = trainer.data_collator([train_dataset[0]])
            if "labels" in probe_batch:
                p_labels = probe_batch["labels"][0]
                if hasattr(p_labels, "numpy"):
                    p_labels = p_labels.cpu().numpy()
                elif hasattr(p_labels, "tolist"):
                    p_labels = p_labels.tolist()
                masked = sum(1 for tok in p_labels if tok == -100)
                if masked == 0 and len(p_labels) > 0:
                    raise RuntimeError(
                        "completion_only_loss=True but collator masked 0 prompt tokens to -100; "
                        "loss would leak onto prompt/evidence text."
                    )
                print(f"[+] completion-only loss verified: {masked} prompt tokens masked to -100.")
        except Exception as e:
            if "masked 0 prompt tokens" in str(e):
                raise
            logger.warning("Completion-only loss collator probe skipped: %s", e)

    # Ensure lora_dropout layers do not allocate intermediate dropout tensors on wide intermediate projections
    if hasattr(trainer, "model"):
        import torch.nn as nn
        for name, module in trainer.model.named_modules():
            if hasattr(module, "lora_dropout"):
                ld = getattr(module, "lora_dropout")
                if isinstance(ld, (dict, nn.ModuleDict)):
                    for k in list(ld.keys()):
                        ld[k] = nn.Identity()

        # Enforce float32 on all trainable parameters when training with fp16
        # PyTorch AMP GradScaler strictly requires float32 master weights/gradients for safe unscaling.
        if getattr(sft_args, "fp16", False):
            for name, param in trainer.model.named_parameters():
                if param.requires_grad and param.dtype != torch.float32:
                    param.data = param.data.to(torch.float32)

    if hasattr(trainer, "args") and hasattr(trainer.args, "n_gpu") and int(trainer.args.n_gpu) != 1 and device.startswith("cuda"):
        raise RuntimeError(
            f"FINAL_PIPELINE_ERROR: SFTTrainer altered single-GPU policy; n_gpu={trainer.args.n_gpu}"
        )

    # 10. Execute Training
    if torch is not None and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

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
    del trainer
    del model
    gc.collect()
    cleanup_cuda_stage(devices=(0, 1))

    reload_status = "pass"
    try:
        reloaded = QwenGenerator.load(
            model_path=model_name_or_path,
            adapter_path=output_dir,
            require_adapter=True,
            device=device,
        )
        sample_out = reloaded.generate(
            question="Căn cứ Điều 1 Luật Dân sự, hãy cho biết hợp đồng là gì?",
            evidence="Hợp đồng là sự thỏa thuận giữa các bên về việc xác lập, thay đổi hoặc chấm dứt quyền, nghĩa vụ dân sự.",
            max_new_tokens=32,
        )
        if not sample_out or not sample_out.strip():
            raise RuntimeError("Reloaded model generated empty response.")
        print(f"Strict reload sample output: {sample_out[:100]}...")
        cleanup_cuda_stage(reloaded, devices=(0, 1))
        del reloaded
        gc.collect()
        cleanup_cuda_stage(devices=(0, 1))
    except Exception as e:
        reload_status = f"fail: {e}"
        msg = f"QLoRA adapter saved but failed strict reload verification: {e}"
        if fail_on_error:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}") from e
        logger.error(msg)

    # 14. Write and Return Provenance Manifest
    is_full_scope = probe_mode is None and val_fold is None
    manifest = {
        "runtime_api_version": 16,
        "backend": "liger_fused_linear_ce",
        "liger_version": REQUIRED_LIGER_VERSION,
        "model": model_name_or_path,
        "base_model_id": model_name_or_path,
        "execution_profile": profile_name,
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
        "is_final_checkpoint": bool(is_full_scope),
        "smoke_only": bool(probe_mode is not None),
        "training_scope": "all_allowed_task2_data" if is_full_scope else (
            f"fold_excluded_{val_fold}" if val_fold is not None else "probe_subset"
        ),
        "val_fold": val_fold,
        "val_fold_excluded": val_fold,
    }

    manifest_path = os.path.join(output_dir, "generator_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerator training complete. Manifest saved to {manifest_path}")
    return manifest
