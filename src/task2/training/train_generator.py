"""Fine-tuning module for Qwen2.5 with modern TRL QLoRA SFT on LegalQA Dual-T4 GPUs."""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.common.security import assert_no_secrets_in_workspace
from src.task2.generator import QwenGenerator, format_qwen_chat_prompt
from src.task2.generation.dataset import (
    SFTExample,
    truncate_evidence_preserving_answer,
    build_sft_example_token_aware,
    build_grounded_training_examples,
    select_worst_case_probe,
)


def run_seq_len_diagnostic(
    qa_path: str = "artifacts/task2/data/qa_unique.parquet",
    labels_path: str = "artifacts/task2/data/retrieval_labels.parquet",
    chunks_path: str = "artifacts/task2/data/legal_chunks.parquet",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    seq_lens: List[int] = [2048, 3072],
) -> Dict[int, Dict[str, Any]]:
    """Actionable sequence length diagnostic comparing truncation and drop rates at different lengths (Task 9)."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = None

    results = {}
    print("\n=== Running Actionable SFT Sequence-Length Diagnostic (Tokenizer-Derived) ===")
    for length in seq_lens:
        examples, diag = build_grounded_training_examples(
            qa_path=qa_path,
            labels_path=labels_path,
            chunks_path=chunks_path,
            tokenizer=tokenizer,
            max_seq_len=length,
            return_diagnostics=True,
        )
        results[length] = {
            "max_seq_len": length,
            "num_examples": len(examples),
            "dropped_count": diag["dropped_count"],
            "drop_rate": round(diag["drop_rate"], 4),
            "evidence_truncated_rate": round(diag["evidence_truncated_rate"], 4),
            "p50_tokens": round(diag["p50_tokens"], 1),
            "p90_tokens": round(diag["p90_tokens"], 1),
            "p95_tokens": round(diag["p95_tokens"], 1),
            "p99_tokens": round(diag["p99_tokens"], 1),
            "max_tokens": diag["max_tokens"],
        }
        print(
            f" - Max Seq Len {length:4d}: {len(examples)} kept, {diag['dropped_count']} dropped ({diag['drop_rate']*100:.1f}%), "
            f"P95={diag['p95_tokens']:.0f}, Max={diag['max_tokens']}"
        )
    return results


def enforce_single_gpu_trainer_args(args: Any, device: str) -> None:
    """Force HF Trainer to keep LegalQA QLoRA on its dedicated generator GPU.

    Kaggle exposes two T4s in one process. Transformers Trainer otherwise uses
    args.n_gpu > 1 to activate nn.DataParallel, which violates LegalQA's
    generator=cuda:0 / retrieval-reranker=cuda:1 hardware split.
    """
    dev = str(device)

    if not dev.startswith("cuda"):
        return

    if dev != "cuda:0":
        raise RuntimeError(
            f"LegalQA QLoRA generator must target cuda:0, got {dev!r}."
        )

    # Force TrainingArguments._setup_devices to run first. On Kaggle this
    # normally records _n_gpu=2 because both T4s are intentionally visible.
    _ = args.device

    if not hasattr(args, "_n_gpu"):
        raise RuntimeError(
            "Transformers TrainingArguments no longer exposes internal "
            "_n_gpu after device setup; refusing to risk DataParallel."
        )

    args._n_gpu = 1

    if int(args.n_gpu) != 1:
        raise RuntimeError(
            f"Failed to force single-GPU QLoRA Trainer policy; "
            f"Trainer reports n_gpu={args.n_gpu}."
        )


def validate_ce_chunk_size(value: int) -> int:
    """Validate that LegalQA CE chunk size is a positive power of two <= 256."""
    value = int(value)
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(
            "LegalQA CE chunk size must be a positive power of two, "
            f"got {value}."
        )
    if value > 256:
        raise ValueError(
            f"LegalQA T4 CE chunk size must be <=256, got {value}."
        )
    return value


def inspect_and_guard_trl_chunk_size(target_chunk_size: int = 32) -> Dict[str, Any]:
    """Inspect TRL chunked LM head chunk size and safely cap to target if > target in TRL 1.12.0.

    Guarded strictly by:
    - TRL version == 1.12.0 (or starts with 1.12.)
    - trl.trainer.sft_trainer._CHUNKED_LM_HEAD_CHUNK_SIZE exists
    - current value > target_chunk_size
    """
    target_chunk_size = validate_ce_chunk_size(target_chunk_size)
    info: Dict[str, Any] = {
        "trl_version": "unknown",
        "chunk_size_attr_present": False,
        "original_chunk_size": None,
        "modified_chunk_size": None,
        "action": "none",
    }
    try:
        import trl
        trl_ver = str(getattr(trl, "__version__", "unknown"))
        info["trl_version"] = trl_ver

        import trl.trainer.sft_trainer as sft_module
        if hasattr(sft_module, "_CHUNKED_LM_HEAD_CHUNK_SIZE"):
            info["chunk_size_attr_present"] = True
            orig_val = getattr(sft_module, "_CHUNKED_LM_HEAD_CHUNK_SIZE")
            info["original_chunk_size"] = orig_val
            info["modified_chunk_size"] = orig_val

            if (trl_ver == "1.12.0" or trl_ver.startswith("1.12.")) and isinstance(orig_val, int) and orig_val > target_chunk_size:
                setattr(sft_module, "_CHUNKED_LM_HEAD_CHUNK_SIZE", target_chunk_size)
                info["modified_chunk_size"] = target_chunk_size
                info["action"] = "capped"
                print(
                    f"[TRL Memory Guard] Capped _CHUNKED_LM_HEAD_CHUNK_SIZE from {orig_val} to {target_chunk_size} "
                    f"for TRL {trl_ver} to prevent GPU0 OOM on large vocab models."
                )
            else:
                info["action"] = f"retained {orig_val}"
                print(f"[TRL Memory Guard] _CHUNKED_LM_HEAD_CHUNK_SIZE={orig_val} (action: {info['action']}).")
        else:
            info["action"] = "attr_not_found"
    except Exception as e:
        info["action"] = f"inspection_error: {e}"
        print(f"[TRL Memory Guard] Note: Could not inspect/modify TRL chunk size: {e}")

    return info


def log_cuda_memory_diagnostics(stage_label: str, device: str = "cuda:0") -> Dict[str, Any]:
    """Log non-secret GPU memory diagnostics for a given training stage."""
    diag: Dict[str, Any] = {"stage": stage_label, "device": device}
    try:
        import torch
        if not torch.cuda.is_available() or not device.startswith("cuda"):
            return diag

        dev_idx = int(device.split(":")[-1]) if ":" in device else 0
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
        alloc_bytes = torch.cuda.memory_allocated(dev_idx)
        res_bytes = torch.cuda.memory_reserved(dev_idx)

        diag.update({
            "free_mb": round(free_bytes / (1024 * 1024), 2),
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "allocated_mb": round(alloc_bytes / (1024 * 1024), 2),
            "reserved_mb": round(res_bytes / (1024 * 1024), 2),
        })
        print(
            f"[VRAM Diagnostics - {stage_label}] Device {device}: "
            f"Allocated: {diag['allocated_mb']:.1f} MiB | "
            f"Reserved: {diag['reserved_mb']:.1f} MiB | "
            f"Free: {diag['free_mb']:.1f} MiB | "
            f"Total: {diag['total_mb']:.1f} MiB"
        )
    except Exception as e:
        print(f"[VRAM Diagnostics - {stage_label}] Memory query notice: {e}")

    return diag


def build_sft_config(max_seq_len: int, **kwargs) -> Any:
    """Build SFTConfig requiring modern completion_only_loss, loss_type, and activation_offloading."""
    import trl
    from trl import SFTConfig

    sig = inspect.signature(SFTConfig)
    if "completion_only_loss" not in sig.parameters:
        raise RuntimeError(
            "Installed TRL SFTConfig does not support completion_only_loss parameter. "
            "Refusing to train with changed/corrupted loss semantics."
        )

    config_kwargs = dict(kwargs)
    config_kwargs["completion_only_loss"] = True

    # Memory-efficient chunked loss and activation offloading for T4 stability
    if "loss_type" in sig.parameters:
        config_kwargs["loss_type"] = "chunked_nll"
    if "activation_offloading" in sig.parameters:
        config_kwargs["activation_offloading"] = True

    if "max_length" in sig.parameters:
        config_kwargs["max_length"] = max_seq_len
    elif "max_seq_length" in sig.parameters:
        config_kwargs["max_seq_length"] = max_seq_len
    else:
        config_kwargs["max_length"] = max_seq_len

    return SFTConfig(**config_kwargs)


def run_qlora_training(
    model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    qa_path: str = "artifacts/task2/data/qa_unique.parquet",
    labels_path: str = "artifacts/task2/data/retrieval_labels.parquet",
    chunks_path: str = "artifacts/task2/data/legal_chunks.parquet",
    output_dir: str = "artifacts/task2/checkpoints/generator/hf_adapter",
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 1e-4,
    max_seq_len: int = 2048,
    val_fold: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    max_steps: Optional[int] = None,
    max_train_examples: Optional[int] = None,
    is_final_checkpoint: Optional[bool] = None,
    fail_on_error: bool = True,
    ce_chunk_size: int = 32,
) -> Dict[str, Any]:
    """Execute QLoRA fine-tuning on GPU 0 using modern TRL prompt-completion SFT with strict reload verification."""
    print(f"=== Starting QLoRA Generator Fine-Tuning (Base: {base_model_id} | Path: {model_name_or_path}) ===")
    assert_no_secrets_in_workspace(Path.cwd())

    try:
        import torch
        from datasets import Dataset as HFDataset
        import peft
        from peft import LoraConfig, PeftModel
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import trl
        from trl import SFTConfig, SFTTrainer
        import bitsandbytes
    except ImportError as e:
        msg = f"Required training packages not installed: {e}"
        if fail_on_error:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}")
        return {"status": "skipped", "reason": "missing_dependencies"}

    # Non-secret diagnostics & TRL chunk-size inspection and guard
    print(f"TRL version: {getattr(trl, '__version__', 'unknown')}")
    print(f"Transformers version: {getattr(transformers, '__version__', 'unknown')}")
    print(f"PEFT version: {getattr(peft, '__version__', 'unknown')}")
    print(f"bitsandbytes version: {getattr(bitsandbytes, '__version__', 'unknown')}")
    ce_chunk_size = validate_ce_chunk_size(ce_chunk_size)
    chunk_info = inspect_and_guard_trl_chunk_size(target_chunk_size=ce_chunk_size)
    print(
        "LegalQA QLoRA CE Chunk Policy: "
        f"requested={ce_chunk_size} | "
        f"effective={chunk_info['modified_chunk_size']}"
    )
    if chunk_info.get("chunk_size_attr_present") and chunk_info.get("modified_chunk_size") != ce_chunk_size:
        raise RuntimeError(
            "FINAL_PIPELINE_ERROR: TRL CE chunk size guard failed: "
            f"requested={ce_chunk_size} but effective={chunk_info.get('modified_chunk_size')}."
        )

    if not torch.cuda.is_available() and device is None:
        msg = "CUDA not available for QLoRA GPU training."
        if fail_on_error:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}")
        return {"status": "skipped", "reason": "no_cuda"}

    dev = device or "cuda:0"
    print(f"QLoRA Training targeted on device: {dev}")

    # Track VRAM on target GPU
    if torch.cuda.is_available() and dev.startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats(dev)
        except Exception:
            pass

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    examples = build_grounded_training_examples(
        qa_path=qa_path,
        labels_path=labels_path,
        chunks_path=chunks_path,
        fold_to_exclude=val_fold,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_train_examples=max_train_examples,
    )
    if not examples:
        msg = f"No SFT training examples generated. Check {qa_path} and {labels_path}."
        if fail_on_error:
            raise FileNotFoundError(f"FINAL_PIPELINE_ERROR: {msg}")
        return {"status": "skipped", "reason": "no_data"}

    dataset = HFDataset.from_list(examples)
    print(f"Training dataset ready: {len(dataset)} examples (val_fold={val_fold}, max_steps={max_steps}).")

    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map={"": dev} if dev.startswith("cuda") else "auto",
        token=token,
    )
    model.config.use_cache = False
    log_cuda_memory_diagnostics("After 4-bit Base Model Load", dev)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Modern TRL prompt-completion configuration (Step 1.6)
    sft_kwargs: Dict[str, Any] = {
        "output_dir": os.path.join(output_dir, "runs"),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": "none",
        "fp16": not use_bf16,
        "bf16": use_bf16,
    }
    if max_steps is not None:
        sft_kwargs["max_steps"] = max_steps

    sft_args = build_sft_config(max_seq_len=max_seq_len, **sft_kwargs)

    enforce_single_gpu_trainer_args(sft_args, dev)

    visible_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        "QLoRA Trainer GPU Policy: "
        f"target={dev} | visible_cuda={visible_cuda} | "
        f"trainer_n_gpu={sft_args.n_gpu}"
    )

    # Modern SFTTrainer automatically handles prompt/completion columns
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    log_cuda_memory_diagnostics("After SFTTrainer Construction", dev)

    if int(trainer.args.n_gpu) != 1 and dev.startswith("cuda"):
        raise RuntimeError(
            "FINAL_PIPELINE_ERROR: SFTTrainer changed the generator GPU policy; "
            f"expected n_gpu=1, got {trainer.args.n_gpu} (would enable DataParallel)."
        )

    log_cuda_memory_diagnostics("Immediately Before trainer.train()", dev)
    train_started = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    train_elapsed = time.perf_counter() - train_started
    log_cuda_memory_diagnostics("After Training Completion", dev)

    # Compute step telemetry
    optimizer_steps = int(trainer.state.global_step) if hasattr(trainer, "state") and hasattr(trainer.state, "global_step") else (max_steps or 0)
    seconds_per_step = round(train_elapsed / max(1, optimizer_steps), 2)
    train_elapsed_sec = round(train_elapsed, 2)

    if optimizer_steps > 0 and len(dataset) > 0:
        estimated_total_optimizer_steps = math.ceil(
            len(dataset) / (batch_size * grad_accum)
        )
        estimated_generator_hours = round(
            (seconds_per_step * estimated_total_optimizer_steps) / 3600,
            2
        )
        print(
            f"[Generator Telemetry] Completed {optimizer_steps} optimizer steps in {train_elapsed_sec:.1f}s "
            f"({seconds_per_step:.2f}s/step). "
            f"Estimated full epoch ({len(dataset)} examples, {estimated_total_optimizer_steps} steps): "
            f"~{estimated_generator_hours:.2f} hours."
        )

    # Measure peak VRAM allocated and reserved on GPU
    peak_vram_mb = 0.0
    peak_reserved_mb = 0.0
    free_vram_mb = 0.0
    if torch.cuda.is_available() and dev.startswith("cuda"):
        try:
            peak_bytes = torch.cuda.max_memory_allocated(dev)
            peak_vram_mb = round(peak_bytes / (1024 * 1024), 2)
            peak_res_bytes = torch.cuda.max_memory_reserved(dev)
            peak_reserved_mb = round(peak_res_bytes / (1024 * 1024), 2)
            dev_idx = int(dev.split(":")[-1]) if ":" in dev else 0
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
            free_vram_mb = round(free_bytes / (1024 * 1024), 2)
            print(
                f"QLoRA Training Peak VRAM on {dev}: {peak_vram_mb:.2f} MB "
                f"(Reserved: {peak_reserved_mb:.2f} MB, Free: {free_vram_mb:.2f} MB)"
            )
        except Exception:
            pass

    # Count exact trainable adapter parameters
    adapter_trainable_params = int(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))
    print(f"Trained PEFT Adapter Parameters: {adapter_trainable_params:,}")

    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Determine finality and scope
    if is_final_checkpoint is None:
        is_final = (val_fold is None and max_steps is None)
    else:
        is_final = is_final_checkpoint

    training_scope = "all_allowed_task2_data" if val_fold is None else f"folds_excluding_{val_fold}"
    if max_steps is not None:
        training_scope = f"smoke_subset_{max_steps}_steps"

    manifest = {
        "base_model_id": base_model_id,
        "base_model": base_model_id,
        "resolved_model_path": model_name_or_path,
        "epochs": epochs,
        "batch_size_per_device": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "val_fold_excluded": val_fold,
        "training_scope": training_scope,
        "is_final_checkpoint": is_final,
        "smoke_only": max_steps is not None,
        "dataset_size": len(dataset),
        "adapter_trainable_params": adapter_trainable_params,
        "ce_chunk_size": ce_chunk_size,
        "train_elapsed_seconds": train_elapsed_sec,
        "optimizer_steps": optimizer_steps,
        "seconds_per_optimizer_step": seconds_per_step,
        "peak_vram_mb": peak_vram_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "free_vram_mb": free_vram_mb,
    }
    with open(os.path.join(output_dir, "generator_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(output_dir, "training_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"QLoRA Training complete (is_final={is_final}, scope={training_scope}). Adapter saved to {output_dir}")

    # Strict Reload Smoke Verification with require_adapter=True
    print("Executing strict reload smoke verification...")
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        reload_gen = QwenGenerator.load(
            model_path=model_name_or_path,
            adapter_path=output_dir,
            device=dev,
            runtime="torch",
            fail_on_fallback=True,
            final_mode=True,
            require_adapter=True,
        )
        if reload_gen.runtime != "torch" or reload_gen.model is None or reload_gen.tokenizer is None:
            raise RuntimeError(f"Reloaded generator is not running in neural torch mode (runtime={reload_gen.runtime})")

        test_out = reload_gen.generate("Quy định xử phạt vi phạm hành chính?", "Điều 1. Phạt tiền từ 1 đến 2 triệu đồng.")
        if not test_out or not test_out.strip():
            raise RuntimeError("Smoke test generated empty text.")
        print(f"Smoke test generation output sample: {test_out[:100]}...")
        del reload_gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        msg = f"QLoRA checkpoint saved but failed strict reload smoke test: {e}"
        if fail_on_error:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}") from e
        print(f"Warning during reload smoke test: {msg}", file=sys.stderr)

    return {"status": "completed", "output_dir": output_dir, "manifest": manifest}
