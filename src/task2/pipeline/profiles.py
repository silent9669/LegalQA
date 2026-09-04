"""Execution profile definitions and resolution logic for LegalQA V16."""

from dataclasses import dataclass
from typing import Optional, Set

from src.task2.production_config import (
    ProductionSelection,
    validate_production_selection_for_profile,
)

VALID_V16_PROFILES: Set[str] = {
    "generator_probe_worstcase",
    "generator_probe_endurance",
    "screen_fold0",
    "final_train_and_submit",
    # Backward compatibility aliases
    "generator_probe",
    "smoke_only",
    "reuse_final_checkpoints_and_submit",
}


@dataclass(frozen=True)
class ExecutionProfile:
    """Immutable resolution of pipeline execution switches and limits."""

    name: str
    run_reranker_training: bool
    run_generator_training: bool
    run_dev_evaluation: bool
    run_public_inference: bool
    reuse_existing_checkpoints: bool
    val_fold: Optional[int]
    probe_selection: Optional[str]  # "worst_case", "endurance", or None
    max_generator_steps: Optional[int]
    max_generator_examples: Optional[int]
    max_reranker_steps: Optional[int]
    max_reranker_pairs: Optional[int]
    max_reranker_val_pairs: Optional[int]
    dev_eval_size: Optional[int]
    requires_generator: bool


def resolve_execution_profile(
    profile_name: str,
    production_cfg: Optional[ProductionSelection] = None,
    allow_unvalidated_final: bool = False,
    smoke_max_steps: int = 5,
    smoke_max_reranker_pairs: int = 200,
    smoke_max_qa_examples: int = 20,
    smoke_eval_queries: int = 10,
) -> ExecutionProfile:
    """Resolve and strictly validate execution parameters for the chosen profile."""
    prof = str(profile_name).strip()
    if prof not in VALID_V16_PROFILES:
        raise ValueError(
            f"Unknown execution profile '{prof}'. Valid profiles: {sorted(VALID_V16_PROFILES)}"
        )

    # 1. generator_probe_worstcase (or alias generator_probe)
    if prof in ("generator_probe_worstcase", "generator_probe"):
        return ExecutionProfile(
            name="generator_probe_worstcase",
            run_reranker_training=False,
            run_generator_training=True,
            run_dev_evaluation=False,
            run_public_inference=False,
            reuse_existing_checkpoints=False,
            val_fold=0,
            probe_selection="worst_case",
            max_generator_steps=3,
            max_generator_examples=None,
            max_reranker_steps=None,
            max_reranker_pairs=None,
            max_reranker_val_pairs=None,
            dev_eval_size=None,
            requires_generator=True,
        )

    # 2. generator_probe_endurance
    if prof == "generator_probe_endurance":
        return ExecutionProfile(
            name="generator_probe_endurance",
            run_reranker_training=False,
            run_generator_training=True,
            run_dev_evaluation=False,
            run_public_inference=False,
            reuse_existing_checkpoints=False,
            val_fold=0,
            probe_selection="endurance",
            max_generator_steps=30,
            max_generator_examples=None,
            max_reranker_steps=None,
            max_reranker_pairs=None,
            max_reranker_val_pairs=None,
            dev_eval_size=None,
            requires_generator=True,
        )

    # 3. screen_fold0
    if prof == "screen_fold0":
        return ExecutionProfile(
            name="screen_fold0",
            run_reranker_training=True,
            run_generator_training=True,
            run_dev_evaluation=True,
            run_public_inference=False,
            reuse_existing_checkpoints=False,
            val_fold=0,
            probe_selection=None,
            max_generator_steps=None,
            max_generator_examples=None,
            max_reranker_steps=None,
            max_reranker_pairs=None,
            max_reranker_val_pairs=None,
            dev_eval_size=250,
            requires_generator=True,
        )

    # 4. smoke_only (convenience debugging profile)
    if prof == "smoke_only":
        return ExecutionProfile(
            name="smoke_only",
            run_reranker_training=True,
            run_generator_training=True,
            run_dev_evaluation=True,
            run_public_inference=False,
            reuse_existing_checkpoints=False,
            val_fold=0,
            probe_selection=None,
            max_generator_steps=smoke_max_steps,
            max_generator_examples=smoke_max_qa_examples,
            max_reranker_steps=smoke_max_steps,
            max_reranker_pairs=smoke_max_reranker_pairs,
            max_reranker_val_pairs=smoke_max_reranker_pairs,
            dev_eval_size=smoke_eval_queries,
            requires_generator=True,
        )

    # Production Profiles: validate production_selection eligibility
    if production_cfg is None:
        raise ValueError(f"Profile '{prof}' requires production_cfg to be provided.")

    validate_production_selection_for_profile(
        production_cfg,
        prof,
        allow_unvalidated_final=allow_unvalidated_final,
    )
    req_gen = production_cfg.requires_generator

    # 5. final_train_and_submit
    if prof == "final_train_and_submit":
        return ExecutionProfile(
            name="final_train_and_submit",
            run_reranker_training=production_cfg.use_task_tuned_reranker,
            run_generator_training=(production_cfg.use_qlora and req_gen),
            run_dev_evaluation=False,
            run_public_inference=True,
            reuse_existing_checkpoints=False,
            val_fold=None,  # All allowed data
            probe_selection=None,
            max_generator_steps=None,
            max_generator_examples=None,
            max_reranker_steps=None,
            max_reranker_pairs=None,
            max_reranker_val_pairs=None,
            dev_eval_size=None,
            requires_generator=req_gen,
        )

    # 6. reuse_final_checkpoints_and_submit
    if prof == "reuse_final_checkpoints_and_submit":
        return ExecutionProfile(
            name="reuse_final_checkpoints_and_submit",
            run_reranker_training=False,
            run_generator_training=False,
            run_dev_evaluation=False,
            run_public_inference=True,
            reuse_existing_checkpoints=True,
            val_fold=None,
            probe_selection=None,
            max_generator_steps=None,
            max_generator_examples=None,
            max_reranker_steps=None,
            max_reranker_pairs=None,
            max_reranker_val_pairs=None,
            dev_eval_size=None,
            requires_generator=req_gen,
        )

    raise ValueError(f"Unhandled profile: {prof}")
