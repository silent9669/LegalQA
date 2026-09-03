"""Tests for LegalQA V15 generator_probe profile and notebook CE chunk binding."""

from __future__ import annotations

import json
import re
from pathlib import Path
import pytest


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


def test_notebook_default_profile_is_generator_probe():
    src = notebook_source()
    assert 'EXECUTION_PROFILE = "generator_probe"' in src
    assert "ALLOW_SINGLE_GPU_SMOKE = False" in src
    assert "ALLOW_UNVALIDATED_FINAL = False" in src


def test_notebook_defines_qlora_ce_chunk_size_32():
    src = notebook_source()
    assert re.search(r"QLORA_CE_CHUNK_SIZE\s*=\s*32", src)


def test_notebook_generator_probe_profile_semantics():
    src = notebook_source()
    # Ensure generator_probe branch exists in Cell 3
    assert 'elif EXECUTION_PROFILE == "generator_probe":' in src or 'if EXECUTION_PROFILE == "generator_probe":' in src

    probe_block_match = re.search(
        r'EXECUTION_PROFILE == "generator_probe":\s*(.*?)(?=\nelif|\nelse:|\n[A-Z0-9_]+\s*=)',
        src,
        re.DOTALL,
    )
    assert probe_block_match is not None, "generator_probe profile block not found"
    probe_block = probe_block_match.group(1)

    assert "RUN_RERANKER_TRAINING = False" in probe_block
    assert "RUN_GENERATOR_TRAINING = True" in probe_block
    assert "RUN_DEV_EVALUATION = False" in probe_block
    assert "RUN_PUBLIC_INFERENCE = False" in probe_block
    assert "REUSE_EXISTING_CHECKPOINTS = False" in probe_block
    assert "TRAIN_VAL_FOLD = 0" in probe_block
    assert "MAX_GENERATOR_STEPS = 3" in probe_block
    assert "MAX_GENERATOR_EXAMPLES = None" in probe_block
    assert "DEV_EVAL_SIZE = None" in probe_block


def test_notebook_passes_ce_chunk_size_to_run_qlora_training():
    src = notebook_source()
    # Check Cell 8 invocation
    assert re.search(r"run_qlora_training\([^)]*ce_chunk_size\s*=\s*QLORA_CE_CHUNK_SIZE", src, re.DOTALL)
    assert "QLORA_CE_CHUNK_SIZE:" in src
