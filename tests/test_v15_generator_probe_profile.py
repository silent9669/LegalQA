"""Tests for LegalQA V16 generator probe profiles and pipeline decoupling."""

from __future__ import annotations

import json
import re
from pathlib import Path
import pytest

from src.task2.pipeline.profiles import resolve_execution_profile


def notebook_cells() -> list[dict]:
    nb = json.loads(Path("kaggle_kernel/legalqa_gpu_pipeline.ipynb").read_text())
    return nb["cells"]


def notebook_source() -> str:
    parts = []
    for cell in notebook_cells():
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        parts.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(parts)


def test_notebook_default_profile_is_generator_probe_worstcase():
    src = notebook_source()
    assert 'EXECUTION_PROFILE = "generator_probe_worstcase"' in src
    assert "ALLOW_SINGLE_GPU_SMOKE = False" in src
    assert "ALLOW_UNVALIDATED_FINAL = False" in src


def test_profile_worstcase_probe_semantics():
    prof = resolve_execution_profile("generator_probe_worstcase")
    assert prof.name == "generator_probe_worstcase"
    assert prof.run_reranker_training is False
    assert prof.run_generator_training is True
    assert prof.run_dev_evaluation is False
    assert prof.run_public_inference is False
    assert prof.val_fold == 0
    assert prof.probe_selection == "worst_case"
    assert prof.max_generator_steps == 3
    assert prof.max_generator_examples is None


def test_profile_endurance_probe_semantics():
    prof = resolve_execution_profile("generator_probe_endurance")
    assert prof.name == "generator_probe_endurance"
    assert prof.run_reranker_training is False
    assert prof.run_generator_training is True
    assert prof.run_dev_evaluation is False
    assert prof.run_public_inference is False
    assert prof.val_fold == 0
    assert prof.probe_selection == "endurance"
    assert prof.max_generator_steps == 30
    assert prof.max_generator_examples is None


def test_v16_pipeline_runner_wires_generator_probe():
    runner_src = Path("src/task2/pipeline/runner.py").read_text(encoding="utf-8")
    assert "train_generator_qlora(" in runner_src
    assert "probe_mode=profile.probe_selection" in runner_src
    assert "use_liger_fused_ce=True" in runner_src
