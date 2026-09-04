"""Tests for LegalQA V13 real Kaggle screen gate and Protocol-8 auto-promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.task2.production_config import (
    load_production_selection,
    validate_production_selection_for_profile,
)


def notebook_cells():
    nb = json.loads(Path("kaggle_kernel/legalqa_gpu_pipeline.ipynb").read_text(encoding="utf-8"))
    return nb["cells"]


def notebook_source():
    nb = json.loads(Path("kaggle_kernel/legalqa_gpu_pipeline.ipynb").read_text(encoding="utf-8"))
    return "\n".join(
        "".join(c.get("source", "")) if isinstance(c.get("source"), list) else str(c.get("source", ""))
        for c in nb["cells"]
        if c.get("cell_type") == "code"
    )


def test_notebook_cell1_strict_screen_fold0_profile():
    """Task 1: Verify Cell 1 commits generator_probe or screen_fold0 with ALLOW_UNVALIDATED_FINAL=False."""
    src = notebook_source()
    assert ('EXECUTION_PROFILE = "generator_probe_worstcase"' in src or 'EXECUTION_PROFILE = "screen_fold0"' in src)
    assert "ALLOW_SINGLE_GPU_SMOKE = False" in src
    assert "ALLOW_UNVALIDATED_FINAL = False" in src
    assert 'ALLOW_UNVALIDATED_FINAL = True' not in src
    assert 'os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"' in src
    assert "REQUIRED_RUNTIME_API_VERSION = 16" in src


def test_notebook_cell10_contains_auto_promotion_and_handoff():
    """Task 2, 3 & 4: Verify runner validates promotion_report, promotes config, and zips handoff."""
    runner_src = Path("src/task2/pipeline/runner.py").read_text(encoding="utf-8")
    assert "promote_production_selection(" in runner_src
    assert "promoted_production_selection.yaml" in runner_src
    assert "screen_handoff.zip" in runner_src
    assert "screen_protocol_version" in runner_src
    assert "SCREEN_PROMOTION_ERROR" in runner_src
    assert "screen_handoff" in runner_src


def test_unvalidated_config_rejected_for_final_profile():
    """Task 6: Verify UNVALIDATED config cannot enter final_train_and_submit when allow_unvalidated_final=False."""
    cfg = load_production_selection("configs/production_selection.yaml")
    if cfg.status == "UNVALIDATED":
        with pytest.raises(RuntimeError, match="UNVALIDATED"):
            validate_production_selection_for_profile(
                cfg,
                "final_train_and_submit",
                allow_unvalidated_final=False,
            )


def test_notebook_screen_fold0_semantics_preserved():
    """Task 5: Verify profiles set full 250 evaluation queries and all training components for screen_fold0."""
    from src.task2.pipeline.profiles import resolve_execution_profile
    prof = resolve_execution_profile("screen_fold0")
    assert prof.dev_eval_size == 250
    assert prof.val_fold == 0
    assert prof.run_reranker_training is True
    assert prof.run_generator_training is True
    assert prof.run_dev_evaluation is True
