import os
import yaml
from pathlib import Path
from scripts.preflight_kaggle import run_preflight_checks
from scripts.audit_parameters import audit_parameter_budget, verify_config_consistency


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
    # Preflight in test/cpu mode should check workspace files and schema
    res = run_preflight_checks(
        pipeline_config_path="configs/pipeline.yaml",
        models_config_path="configs/models.yaml",
        require_cuda=False,
        check_dataset_files=False,
    )
    assert res["passed"] is True
    assert len(res["errors"]) == 0
