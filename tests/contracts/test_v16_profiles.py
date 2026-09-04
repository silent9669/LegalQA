import pytest
from src.task2.pipeline.profiles import (
    ExecutionProfile,
    resolve_execution_profile,
    VALID_V16_PROFILES,
)
from src.task2.production_config import ProductionSelection


def test_valid_v16_profiles_set():
    assert "generator_probe_worstcase" in VALID_V16_PROFILES
    assert "generator_probe_endurance" in VALID_V16_PROFILES
    assert "screen_fold0" in VALID_V16_PROFILES
    assert "final_train_and_submit" in VALID_V16_PROFILES


def test_resolve_generator_probe_worstcase():
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


def test_resolve_generator_probe_endurance():
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


def test_resolve_screen_fold0():
    prof = resolve_execution_profile("screen_fold0")
    assert prof.name == "screen_fold0"
    assert prof.run_reranker_training is True
    assert prof.run_generator_training is True
    assert prof.run_dev_evaluation is True
    assert prof.run_public_inference is False
    assert prof.val_fold == 0
    assert prof.probe_selection is None
    assert prof.max_generator_steps is None
    assert prof.dev_eval_size == 250


def test_resolve_final_train_and_submit_promoted():
    mock_prod = ProductionSelection(
        schema_version=2,
        status="PROMOTED",
        source_screen_manifest="screen.json",
        source_screen_sha256="12345",
        stack="stack_a",
        use_task_tuned_reranker=True,
        reranker_base_model="BAAI/bge-reranker-v2-m3",
        reranker_checkpoint="checkpoints/reranker",
        use_qlora=True,
        generator_base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_path="checkpoints/generator",
        max_new_tokens=384,
        candidate_policy="fixed_baseline",
        best_fixed_candidate="generated",
        selector_checkpoint=None,
        primary_evidence_pack="multi_seed",
        raw_config={"screen_protocol_version": 8},
    )

    prof = resolve_execution_profile("final_train_and_submit", production_cfg=mock_prod)
    assert prof.name == "final_train_and_submit"
    assert prof.run_reranker_training is True
    assert prof.run_generator_training is True
    assert prof.run_dev_evaluation is False
    assert prof.run_public_inference is True
    assert prof.val_fold is None
    assert prof.requires_generator is True


def test_resolve_final_train_and_submit_rejects_unvalidated():
    mock_prod = ProductionSelection(
        schema_version=2,
        status="UNVALIDATED",
        source_screen_manifest=None,
        source_screen_sha256=None,
        stack="stack_a",
        use_task_tuned_reranker=True,
        reranker_base_model="BAAI/bge-reranker-v2-m3",
        reranker_checkpoint="checkpoints/reranker",
        use_qlora=True,
        generator_base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_path=None,
        max_new_tokens=384,
        candidate_policy="fixed_baseline",
        best_fixed_candidate="stitched_extract",
        selector_checkpoint=None,
        primary_evidence_pack="multi_seed",
        raw_config={},
    )

    with pytest.raises(RuntimeError, match="PROMOTED"):
        resolve_execution_profile(
            "final_train_and_submit",
            production_cfg=mock_prod,
            allow_unvalidated_final=False,
        )


def test_resolve_unknown_profile_raises():
    with pytest.raises(ValueError, match="Unknown execution profile"):
        resolve_execution_profile("invalid_profile_name")
