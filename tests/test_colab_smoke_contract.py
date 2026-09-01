"""Tests for LegalQA Colab smoke runner contract, runtime lock, and operating rules."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_colab_lock_never_replaces_torch_cuda():
    """Task 1: Verify requirements-colab-smoke.txt pins exact ML libraries and excludes Torch/CUDA."""
    txt = Path("requirements-colab-smoke.txt").read_text()
    forbidden = [
        "\ntorch==",
        "\ntorch>=",
        "\ntorchvision",
        "\ntorchaudio",
        "\ntriton",
        "\ncuda-",
        "\nnvidia-",
    ]
    assert all(x not in "\n" + txt for x in forbidden)
    assert "transformers==5.0.0" in txt
    assert "trl==1.12.0" in txt
    assert "bitsandbytes==0.50.2" in txt
    assert "peft==0.19.1" in txt
    assert "accelerate==1.13.0" in txt


def test_colab_smoke_runner_source_contract():
    """Task 3: Verify scripts/run_colab_smoke.py enforces required runtime flags and safety settings."""
    runner_path = Path("scripts/run_colab_smoke.py")
    assert runner_path.exists(), "scripts/run_colab_smoke.py must exist"
    src = runner_path.read_text()

    # Required patterns
    assert 'os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"' in src
    assert 'device="cuda:0"' in src
    assert "max_seq_len=2048" in src
    assert "grad_accum=8" in src
    assert "fail_on_error=True" in src
    assert "Tesla T4" in src or "T4" in src

    # Forbidden anti-patterns
    forbidden_terms = [
        "kaggle kernels push",
        "kaggle kernels output",
        "kaggle kernels status",
        "CUDA_VISIBLE_DEVICES",
        "ALLOW_SINGLE_GPU_SMOKE = True",
    ]
    for term in forbidden_terms:
        assert term not in src, f"Forbidden anti-pattern found in Colab runner: {term!r}"


def test_colab_smoke_runner_cli_help():
    """Task 3: Verify scripts/run_colab_smoke.py CLI options parse correctly."""
    res = subprocess.run(
        [sys.executable, "scripts/run_colab_smoke.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--data-root" in res.stdout
    assert "--component" in res.stdout
    assert "--mode" in res.stdout
    assert "--model-name" in res.stdout
    assert "--output-dir" in res.stdout


def test_operating_rules_reserve_kaggle_for_user():
    """Task 5: Verify docs/OPERATING_RULES.md strictly prohibits agent Kaggle notebook execution."""
    rules_path = Path("docs/OPERATING_RULES.md")
    assert rules_path.exists(), "docs/OPERATING_RULES.md must exist"
    txt = rules_path.read_text()
    assert "Coding agents must never push or execute Kaggle notebooks." in txt
