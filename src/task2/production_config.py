"""Authoritative production selection configuration loader and validator."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml


@dataclass
class ProductionSelection:
    schema_version: int
    status: str  # "UNVALIDATED" or "PROMOTED"
    source_screen_manifest: Optional[str]
    source_screen_sha256: Optional[str]
    stack: str
    use_task_tuned_reranker: bool
    reranker_base_model: str
    reranker_checkpoint: str
    use_qlora: bool
    generator_base_model: str
    adapter_path: Optional[str]
    max_new_tokens: int
    candidate_policy: str
    best_fixed_candidate: Optional[str]
    selector_checkpoint: Optional[str]
    primary_evidence_pack: str
    raw_config: Dict[str, Any]

    @property
    def requires_generator(self) -> bool:
        """Derive whether the current candidate selection policy requires generator output."""
        return policy_requires_generator(self.candidate_policy, self.best_fixed_candidate)


def policy_requires_generator(candidate_policy: str, best_fixed_candidate: Optional[str] = None) -> bool:
    """Check if the candidate policy or fixed choice requires Qwen generator output."""
    if candidate_policy in ("learned", "meta_selector"):
        return True
    if candidate_policy in ("fixed_baseline", "fixed"):
        if best_fixed_candidate in ("generated", "strategy_f_1000", "strategy_f_1500", "strategy_f_600", "strategy_f_300"):
            return True
        return False
    return False


def load_production_selection(config_path: str = "configs/production_selection.yaml") -> ProductionSelection:
    """Load and parse production selection YAML into a typed ProductionSelection dataclass."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Production selection config not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in {config_path}: expected dictionary")

    schema_v = data.get("schema_version", 1)
    if schema_v < 2:
        raise ValueError(f"Unsupported schema_version {schema_v} in {config_path}; require >= 2")

    status = data.get("status", "UNVALIDATED")
    if status not in ("UNVALIDATED", "PROMOTED"):
        raise ValueError(f"Invalid status '{status}' in {config_path}; expected UNVALIDATED or PROMOTED")

    reranker_cfg = data.get("reranker", {})
    generator_cfg = data.get("generator", {})
    policy_cfg = data.get("candidate_policy", {})
    evidence_cfg = data.get("evidence", {})

    return ProductionSelection(
        schema_version=schema_v,
        status=status,
        source_screen_manifest=data.get("source_screen_manifest"),
        source_screen_sha256=data.get("source_screen_sha256"),
        stack=data.get("stack", "stack_a"),
        use_task_tuned_reranker=bool(reranker_cfg.get("use_task_tuned", True)),
        reranker_base_model=reranker_cfg.get("base_model", "BAAI/bge-reranker-v2-m3"),
        reranker_checkpoint=reranker_cfg.get("checkpoint", "checkpoints/reranker/best"),
        use_qlora=bool(generator_cfg.get("use_qlora", True)),
        generator_base_model=generator_cfg.get("base_model", "Qwen/Qwen2.5-3B-Instruct"),
        adapter_path=generator_cfg.get("adapter_path", "checkpoints/generator/hf_adapter"),
        max_new_tokens=int(generator_cfg.get("max_new_tokens", 384)),
        candidate_policy=policy_cfg.get("type", "fixed_baseline"),
        best_fixed_candidate=policy_cfg.get("best_fixed_candidate", "stitched_extract"),
        selector_checkpoint=policy_cfg.get("selector_checkpoint"),
        primary_evidence_pack=evidence_cfg.get("primary_pack", "multi_seed_2500_chars"),
        raw_config=data,
    )


def validate_production_selection_for_profile(
    config: ProductionSelection,
    profile: str,
    allow_unvalidated_final: bool = False,
) -> None:
    """Validate that the production configuration is eligible for the chosen execution profile."""
    if profile in ("final_train_and_submit", "reuse_final_checkpoints_and_submit"):
        if config.status == "UNVALIDATED" and not allow_unvalidated_final:
            raise RuntimeError(
                f"Production config status is 'UNVALIDATED'. "
                f"Running profile '{profile}' requires a validated 'PROMOTED' config resulting from screen_fold0, "
                f"or setting ALLOW_UNVALIDATED_FINAL=True for emergency override."
            )
