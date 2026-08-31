import json
import os
from pathlib import Path
import pytest

from src.task2.production_config import (
    ProductionSelection,
    load_production_selection,
    policy_requires_generator,
    validate_production_selection_for_profile,
)
from src.task2.checkpoint_manifest import (
    assert_final_checkpoint,
    load_generator_manifest,
    load_reranker_manifest,
)


def test_load_production_selection():
    cfg = load_production_selection("configs/production_selection.yaml")
    assert cfg.schema_version == 3
    assert cfg.status in ("UNVALIDATED", "PROMOTED")
    assert cfg.stack == "stack_a"
    assert cfg.reranker_base_model == "BAAI/bge-reranker-v2-m3"
    assert cfg.generator_base_model == "Qwen/Qwen2.5-3B-Instruct"


def test_policy_requires_generator():
    assert policy_requires_generator("learned", None) is True
    assert policy_requires_generator("fixed_baseline", "generated") is True
    assert policy_requires_generator("fixed_baseline", "strategy_f_1000") is True
    assert policy_requires_generator("fixed_baseline", "stitched_extract") is False
    assert policy_requires_generator("fixed_baseline", "pack_focused") is False
    assert policy_requires_generator("fixed_baseline", "pack_top2_relevance") is False


def test_validate_production_selection_unvalidated_guard():
    cfg = load_production_selection("configs/production_selection.yaml")
    # By default, configs/production_selection.yaml has status UNVALIDATED
    cfg.status = "UNVALIDATED"
    # Should be fine for smoke_only and screen_fold0
    validate_production_selection_for_profile(cfg, "smoke_only")
    validate_production_selection_for_profile(cfg, "screen_fold0")

    # Should raise for final_train_and_submit without override
    with pytest.raises(RuntimeError, match="status is 'UNVALIDATED'"):
        validate_production_selection_for_profile(cfg, "final_train_and_submit", allow_unvalidated_final=False)

    # Should pass when allow_unvalidated_final=True
    validate_production_selection_for_profile(cfg, "final_train_and_submit", allow_unvalidated_final=True)

    # Should pass when status is PROMOTED with protocol 7
    cfg.status = "PROMOTED"
    cfg.raw_config["screen_protocol_version"] = 7
    validate_production_selection_for_profile(cfg, "final_train_and_submit", allow_unvalidated_final=False)


def test_assert_final_checkpoint(tmp_path: Path):
    ckpt_dir = tmp_path / "reranker_final"
    ckpt_dir.mkdir()

    # 1. Missing manifest
    with pytest.raises(FileNotFoundError):
        assert_final_checkpoint(str(ckpt_dir), "BAAI/bge-reranker-v2-m3", "reranker")

    # 2. Non-final checkpoint
    with open(ckpt_dir / "reranker_manifest.json", "w") as f:
        json.dump({
            "is_final_checkpoint": False,
            "training_scope": "fold_0_held_out",
            "base_model": "BAAI/bge-reranker-v2-m3",
        }, f)

    with pytest.raises(ValueError, match="NOT marked as is_final_checkpoint=true"):
        assert_final_checkpoint(str(ckpt_dir), "BAAI/bge-reranker-v2-m3", "reranker")

    # 3. Smoke checkpoint
    with open(ckpt_dir / "reranker_manifest.json", "w") as f:
        json.dump({
            "is_final_checkpoint": True,
            "training_scope": "all_allowed_task2_data",
            "smoke_only": True,
            "base_model": "BAAI/bge-reranker-v2-m3",
        }, f)

    with pytest.raises(ValueError, match="is a smoke checkpoint"):
        assert_final_checkpoint(str(ckpt_dir), "BAAI/bge-reranker-v2-m3", "reranker")

    # 4. Valid final checkpoint
    with open(ckpt_dir / "reranker_manifest.json", "w") as f:
        json.dump({
            "is_final_checkpoint": True,
            "training_scope": "all_allowed_task2_data",
            "smoke_only": False,
            "base_model": "BAAI/bge-reranker-v2-m3",
            "val_fold": None,
        }, f)

    manifest = assert_final_checkpoint(str(ckpt_dir), "BAAI/bge-reranker-v2-m3", "reranker")
    assert manifest["is_final_checkpoint"] is True
    assert manifest["training_scope"] == "all_allowed_task2_data"
