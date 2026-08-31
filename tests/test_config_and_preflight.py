import os
import sys
import yaml
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from scripts.preflight_kaggle import run_preflight_checks
from scripts.audit_parameters import audit_parameter_budget, verify_config_consistency
from src.task2.path_resolver import find_runtime_roots, find_qwen_model_dir, resolve_runtime_paths


def test_canonical_pipeline_config():
    config_path = "configs/pipeline.yaml"
    assert os.path.exists(config_path), "configs/pipeline.yaml must exist"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("seed") == 42
    paths = cfg.get("paths", {})
    assert paths.get("data_dir") == "artifacts/task2/data"
    assert paths.get("index_dir") == "artifacts/task2/indexes"
    assert paths.get("checkpoint_dir") == "artifacts/task2/checkpoints"

    retrieval = cfg.get("retrieval", {})
    assert retrieval.get("sparse", {}).get("method") == "bm25s"
    assert "stack_a_model" in retrieval.get("dense", {})
    assert "stack_b_model" in retrieval.get("dense", {})

    gen = cfg.get("generation", {})
    assert "stack_a_model" in gen
    assert "stack_b_model" in gen

    final = cfg.get("final", {})
    assert final.get("fail_on_missing_index") is True
    assert final.get("fail_on_model_fallback") is True


def test_models_config_and_budget():
    models_path = "configs/models.yaml"
    assert os.path.exists(models_path), "configs/models.yaml must exist"
    with open(models_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("{") or models_path.endswith(".json"):
        import json
        cfg = json.loads(content)
    else:
        cfg = yaml.safe_load(content)

    assert "stacks" in cfg or "recommended_stack" in cfg

    audit_a = audit_parameter_budget(models_path, stack="stack_a")
    assert audit_a["is_compliant"] is True
    assert audit_a["total_learned_parameters"] < 4000000000

    audit_b = audit_parameter_budget(models_path, stack="stack_b")
    assert audit_b["is_compliant"] is True
    assert audit_b["total_learned_parameters"] < 4000000000


def test_preflight_checks_basic():
    res = run_preflight_checks(
        pipeline_config_path="configs/pipeline.yaml",
        models_config_path="configs/models.yaml",
        require_cuda=False,
        check_dataset_files=False,
        check_indexes=False,
    )
    assert res["passed"] is True
    assert len(res["errors"]) == 0


def test_preflight_index_checks(tmp_path: Path):
    # Setup dummy dense index with wrong dtype to ensure preflight catches it (P0-12, P0-13)
    dek21_dir = tmp_path / "dek21"
    dek21_dir.mkdir()

    # Create dummy FP32 array (should fail because expected is float16)
    emb = np.zeros((10, 768), dtype=np.float32)
    np.save(dek21_dir / "embeddings.npy", emb)

    import json
    with open(dek21_dir / "dense_manifest.json", "w") as f:
        json.dump({
            "model_id": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
            "dtype": "float32",  # Wrong dtype
            "dim": 768,
            "corpus_rows": 10,
        }, f)

    res = run_preflight_checks(
        pipeline_config_path="configs/pipeline.yaml",
        models_config_path="configs/models.yaml",
        require_cuda=False,
        check_dataset_files=False,
        check_indexes=True,
        bm25_dir=None,
        dek21_dir=str(dek21_dir),
    )
    assert res["passed"] is False
    assert any("Dense dtype must be 'float16'" in e for e in res["errors"])


def test_path_resolver(tmp_path: Path):
    kaggle_input = tmp_path / "kaggle_input"
    dataset_dir = kaggle_input / "legalqa-dataset"
    dataset_dir.mkdir(parents=True)

    (dataset_dir / "dataset_manifest.json").write_text('{"title": "LegalQA"}', encoding="utf-8")
    (dataset_dir / "legal_chunks.parquet").write_text('dummy', encoding="utf-8")

    roots = find_runtime_roots(str(kaggle_input))
    assert len(roots) == 1
    assert roots[0] == str(dataset_dir)
