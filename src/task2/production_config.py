"""Authoritative production selection configuration loader and validator (V8)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from src.common.hashing import sha256_file

GENERATOR_DEPENDENT_CANDIDATES = {
    "generated",
    "snapped",
    "strategy_f_300",
    "strategy_f_600",
    "strategy_f_1000",
    "strategy_f_1500",
}


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
    provenance: Optional[Dict[str, Any]] = None

    @property
    def requires_generator(self) -> bool:
        """Derive whether the current candidate selection policy requires generator output."""
        return policy_requires_generator(self.candidate_policy, self.best_fixed_candidate)


def policy_requires_generator(candidate_policy: str, best_fixed_candidate: Optional[str] = None) -> bool:
    """Check if the candidate policy or fixed choice requires Qwen generator output."""
    p = str(candidate_policy).lower().strip()
    if p in ("learned", "learned_model", "meta_selector"):
        return True
    if p in ("fixed_baseline", "fixed", "direct_candidate"):
        cand = str(best_fixed_candidate).lower().strip() if best_fixed_candidate else ""
        return cand in GENERATOR_DEPENDENT_CANDIDATES
    return False


def get_default_production_selection() -> ProductionSelection:
    """Return canonical default production selection when running smoke or unconfigured pipelines."""
    return ProductionSelection(
        schema_version=1,
        status="PROMOTED",
        source_screen_manifest=None,
        source_screen_sha256=None,
        stack="stack_a",
        use_task_tuned_reranker=False,
        reranker_base_model="BAAI/bge-reranker-v2-m3",
        reranker_checkpoint="BAAI/bge-reranker-v2-m3",
        use_qlora=True,
        generator_base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_path=None,
        max_new_tokens=384,
        candidate_policy="fixed_baseline",
        best_fixed_candidate="generated",
        selector_checkpoint=None,
        primary_evidence_pack="hybrid",
        raw_config={},
    )


def load_production_selection(config_path: Optional[str] = None) -> ProductionSelection:
    """Load and parse production selection YAML into a typed ProductionSelection dataclass.

    Enforces valid policy types ('fixed_baseline', 'learned_model', 'direct_candidate')
    and rejects overloading candidate names as policy types.
    """
    if config_path is None:
        return get_default_production_selection()

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

    policy_type = policy_cfg.get("type", "fixed_baseline")
    if policy_type in GENERATOR_DEPENDENT_CANDIDATES or policy_type in ("stitched_extract", "focused_extract"):
        raise ValueError(
            f"Invalid candidate_policy type '{policy_type}' in {config_path}. "
            f"Policy type must be 'fixed_baseline', 'learned_model', or 'direct_candidate'. "
            f"Set 'type: fixed_baseline' and 'best_fixed_candidate: {policy_type}' instead."
        )

    best_fixed = policy_cfg.get("best_fixed_candidate", "stitched_extract")

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
        candidate_policy=policy_type,
        best_fixed_candidate=best_fixed,
        selector_checkpoint=policy_cfg.get("selector_checkpoint"),
        primary_evidence_pack=evidence_cfg.get("primary_pack", "multi_seed_2500_chars"),
        raw_config=data,
        provenance=data.get("provenance"),
    )


def verify_promotion_provenance(
    config: ProductionSelection,
    search_roots: Optional[Sequence[str | Path]] = None,
) -> None:
    """Cryptographically revalidate promotion report and screen manifests before final/reuse execution."""
    prov = config.provenance or {}
    report_rel_path = prov.get("promotion_report_path") or config.source_screen_manifest
    expected_report_sha = prov.get("promotion_report_sha256") or config.source_screen_sha256

    if not report_rel_path or not expected_report_sha:
        raise RuntimeError("Production config is missing promotion report provenance path or SHA256.")

    # Locate report file
    roots = [Path(r) for r in (search_roots or [".", "/kaggle/input", "/kaggle/working"])]
    report_path: Optional[Path] = None
    if Path(report_rel_path).is_file():
        report_path = Path(report_rel_path)
    else:
        for r in roots:
            cand = r / report_rel_path
            if cand.is_file():
                report_path = cand
                break
            cand_direct = r / Path(report_rel_path).name
            if cand_direct.is_file():
                report_path = cand_direct
                break
            for sub in r.glob("**/promotion_report.json"):
                if sub.is_file():
                    report_path = sub
                    break
            if report_path:
                break

    if not report_path:
        raise FileNotFoundError(f"Promotion report file not found: {report_rel_path}")

    actual_report_sha = sha256_file(report_path)
    if actual_report_sha != expected_report_sha:
        raise RuntimeError(
            f"Promotion report SHA256 mismatch for {report_path}: "
            f"actual {actual_report_sha} != expected {expected_report_sha}"
        )

    # Validate screen_run_manifest if specified
    manifest_rel_path = prov.get("screen_run_manifest_path")
    expected_manifest_sha = prov.get("screen_run_manifest_sha256")
    if manifest_rel_path and expected_manifest_sha:
        manifest_path: Optional[Path] = None
        if Path(manifest_rel_path).is_file():
            manifest_path = Path(manifest_rel_path)
        else:
            for r in roots:
                cand = r / manifest_rel_path
                if cand.is_file():
                    manifest_path = cand
                    break
                cand_direct = r / Path(manifest_rel_path).name
                if cand_direct.is_file():
                    manifest_path = cand_direct
                    break
                for sub in r.glob("**/screen_run_manifest.json"):
                    if sub.is_file():
                        manifest_path = sub
                        break
                if manifest_path:
                    break
        if manifest_path:
            actual_m_sha = sha256_file(manifest_path)
            if actual_m_sha != expected_manifest_sha:
                raise RuntimeError(
                    f"Screen run manifest SHA256 mismatch for {manifest_path}: "
                    f"actual {actual_m_sha} != expected {expected_manifest_sha}"
                )

    # Validate protocol version and API version
    protocol_v = prov.get("screen_protocol_version") or config.raw_config.get("screen_protocol_version", 1)
    if int(protocol_v) < 8:
        raise RuntimeError(f"Promotion provenance requires screen_protocol_version >= 8, got {protocol_v}")

    api_v = prov.get("runtime_api_version")
    if api_v and int(api_v) != 16:
        raise RuntimeError(f"Promotion provenance requires runtime_api_version=16, got {api_v}")


def validate_production_selection_for_profile(
    config: ProductionSelection,
    profile: str,
    allow_unvalidated_final: bool = False,
    verify_provenance: bool = False,
) -> None:
    """Validate that the production configuration is eligible for the chosen execution profile (Protocol 8)."""
    if profile in ("final_train_and_submit", "reuse_final_checkpoints_and_submit"):
        if config.status == "UNVALIDATED" and not allow_unvalidated_final:
            raise RuntimeError(
                f"Production config status is 'UNVALIDATED'. "
                f"Running profile '{profile}' requires a validated 'PROMOTED' config resulting from screen_fold0, "
                f"or setting ALLOW_UNVALIDATED_FINAL=True for emergency override."
            )
        if config.status == "PROMOTED" and not allow_unvalidated_final:
            protocol_v = config.raw_config.get("screen_protocol_version", 1)
            if protocol_v < 8:
                raise RuntimeError(
                    f"Promoted config uses screen_protocol_version={protocol_v} < 8. "
                    f"Profile '{profile}' requires screening under Protocol 8 (staged component consistency and provenance)."
                )
            if verify_provenance:
                verify_promotion_provenance(config)
