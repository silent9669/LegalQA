from __future__ import annotations

import json
import re
from pathlib import Path
import pytest


def notebook_cells() -> list[dict]:
    nb = json.loads(Path("kaggle_kernel/legalqa_gpu_pipeline.ipynb").read_text(encoding="utf-8"))
    return nb["cells"]


def notebook_source() -> str:
    parts = []
    for cell in notebook_cells():
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        parts.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(parts)


def test_v16_notebook_owns_api16_literal():
    src = notebook_source()
    assert re.search(r"REQUIRED_RUNTIME_API_VERSION\s*=\s*16\b", src), (
        "Notebook must define literal REQUIRED_RUNTIME_API_VERSION = 16"
    )


def test_v16_notebook_committed_profile_is_worstcase_probe():
    src = notebook_source()
    assert 'EXECUTION_PROFILE = "generator_probe_worstcase"' in src, (
        "Committed EXECUTION_PROFILE must be generator_probe_worstcase"
    )


def test_v16_notebook_retains_async_load_guard():
    src = notebook_source()
    assert 'os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"' in src


def test_v16_notebook_retains_strict_dual_t4_guard():
    src = notebook_source()
    assert "ALLOW_SINGLE_GPU_SMOKE = False" in src
    assert "gpu_count < 2 and not ALLOW_SINGLE_GPU_SMOKE" in src


def test_v16_notebook_calls_strict_path_resolvers():
    src = notebook_source()
    assert re.search(r"resolve_packaged_code_root\(\s*[\"']/kaggle/input[\"']\s*,\s*strict=True\s*\)", src)
    assert re.search(r"resolve_runtime_paths\(\s*[\"']/kaggle/input[\"']\s*,\s*strict=True\s*,\s*allow_remote_model_download=False\s*,?\s*\)", src)


def test_v16_notebook_calls_pipeline_runner():
    src = notebook_source()
    assert "from src.task2.pipeline.runner import run_pipeline" in src
    assert "run_pipeline(" in src
