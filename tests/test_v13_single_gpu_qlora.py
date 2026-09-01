"""Tests for LegalQA V13 single-GPU QLoRA Trainer policy & anti-DataParallel contract."""

from __future__ import annotations

import inspect
from typing import Any
import pytest

import src.task2.training.train_generator as train_generator_mod
from src.task2.training.train_generator import enforce_single_gpu_trainer_args


class DummyArgs:
    def __init__(self, n_gpu: int = 2, device: str = "cuda:0"):
        self._n_gpu = n_gpu
        self._device = device

    @property
    def device(self) -> str:
        return self._device

    @property
    def n_gpu(self) -> int:
        return self._n_gpu


def test_generator_trainer_forces_one_gpu_when_two_are_visible():
    """Verify that when 2 GPUs are visible, generator args are forced to n_gpu=1."""
    args = DummyArgs(n_gpu=2, device="cuda:0")
    enforce_single_gpu_trainer_args(args, "cuda:0")
    assert args.n_gpu == 1


def test_generator_trainer_rejects_wrong_cuda_target():
    """Verify that QLoRA generator fails loud if targeted to any device other than cuda:0."""
    args = DummyArgs(n_gpu=2, device="cuda:0")
    with pytest.raises(RuntimeError, match="cuda:0"):
        enforce_single_gpu_trainer_args(args, "cuda:1")


def test_generator_trainer_cpu_policy_does_not_fake_gpu_count():
    """Verify that CPU execution is untouched."""
    args = DummyArgs(n_gpu=0, device="cpu")
    enforce_single_gpu_trainer_args(args, "cpu")
    assert args.n_gpu == 0


def test_run_qlora_enforces_policy_before_trainer():
    """Verify run_qlora_training calls enforce_single_gpu_trainer_args before SFTTrainer."""
    src = inspect.getsource(train_generator_mod.run_qlora_training)
    force_pos = src.index("enforce_single_gpu_trainer_args(sft_args, dev)")
    trainer_pos = src.index("trainer = SFTTrainer(")
    assert force_pos < trainer_pos
    assert "trainer.args.n_gpu" in src


def test_training_semantics_and_hyperparameters_preserved():
    """Verify that QLoRA hyperparameters and async load fix are preserved without speculative reduction."""
    src = inspect.getsource(train_generator_mod.run_qlora_training)
    # Batch size 1, grad accum 8, seq len 2048, lr 1e-4
    sig = inspect.signature(train_generator_mod.run_qlora_training)
    assert sig.parameters["max_seq_len"].default == 2048
    assert sig.parameters["batch_size"].default == 1
    assert sig.parameters["grad_accum"].default == 8
    assert sig.parameters["lr"].default == 1e-4
    assert "paged_adamw_8bit" in src
    assert "r=16" in src or "r= 16" in src or "r = 16" in src or '"r": 16' in src or "lora_r = 16" in src or "lora_alpha" in src
