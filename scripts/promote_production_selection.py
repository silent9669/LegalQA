"""Deterministic production selection promoter from screen_fold0 evaluation report (V8)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.hashing import sha256_file
from src.common.security import assert_no_secrets_in_workspace
from src.task2.evaluation import GENERATOR_DEPENDENT_FAMILIES


def promote_production_selection(
    report_path: str = "artifacts/task2/evaluations/promotion_report.json",
    config_path: str = "configs/production_selection.yaml",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Read measured promotion_report.json and freeze an authoritative PROMOTED configuration (Protocol 8)."""
    assert_no_secrets_in_workspace(Path.cwd())
    output_path = output_path or config_path

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Promotion report not found at: {report_path}")

    report_sha256 = sha256_file(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # 1. Step 8.1: Validate Protocol 8 and required report schema keys
    protocol_v = report.get("screen_protocol_version", 1)
    if protocol_v < 8:
        raise ValueError(
            f"Promotion report at {report_path} uses obsolete screen_protocol_version {protocol_v}. "
            f"Require >= 8 (staged component-consistent screening and provenance)."
        )

    required_keys = [
        "screen_protocol_version",
        "held_out_fold",
        "sample_ids_sha256",
        "sample_size",
        "evaluated_systems",
        "selected_reranker",
        "selected_generator",
        "candidate_policy",
        "final_measured_system_key",
        "overall_deployable_winner",
        "overall_deployable_meteor",
    ]
    for k in required_keys:
        if k not in report:
            raise ValueError(f"Promotion report at {report_path} is missing required key: '{k}'")

    eval_systems = report["evaluated_systems"]
    expected_sample_sha = report["sample_ids_sha256"]
    expected_sample_size = report["sample_size"]

    # 2. Step 8.2: Require identical evaluation set across all systems
    for sys_key, sys_summary in eval_systems.items():
        if isinstance(sys_summary, dict) and "sample_ids_sha256" in sys_summary:
            if sys_summary["sample_ids_sha256"] != expected_sample_sha:
                raise ValueError(
                    f"Sample IDs hash mismatch in evaluated system '{sys_key}': "
                    f"found {sys_summary['sample_ids_sha256']} != expected {expected_sample_sha}."
                )
            if sys_summary.get("sample_size") != expected_sample_size:
                raise ValueError(
                    f"Sample size mismatch in evaluated system '{sys_key}': "
                    f"found {sys_summary.get('sample_size')} != expected {expected_sample_size}."
                )

    selected_reranker = report["selected_reranker"]
    selected_generator = report["selected_generator"]
    final_measured_key = report["final_measured_system_key"]
    cand_policy = report["candidate_policy"]
    best_fixed = cand_policy.get("best_fixed_candidate", "stitched_extract")

    if final_measured_key not in eval_systems:
        raise ValueError(f"final_measured_system_key '{final_measured_key}' not found in evaluated_systems.")

    final_summary = eval_systems[final_measured_key]

    # 3. Step 8.3 & 8.4: Cross-check component consistency
    use_tuned_rerank = bool(selected_reranker.get("use_task_tuned", False))
    use_qlora = bool(selected_generator.get("use_qlora", False))

    if not use_qlora and final_measured_key == "R_SELECTED_G1":
        raise ValueError(
            "Component inconsistency in promotion report: selected_generator has use_qlora=False, "
            "but final_measured_system_key is 'R_SELECTED_G1'."
        )

    if not use_tuned_rerank and final_measured_key == "R1G0":
        raise ValueError(
            "Component inconsistency in promotion report: selected_reranker has use_task_tuned=False, "
            "but final_measured_system_key is 'R1G0'."
        )

    # Verify reranker checkpoint identity
    if isinstance(final_summary, dict) and "reranker_checkpoint" in final_summary:
        if final_summary["reranker_checkpoint"] != selected_reranker.get("checkpoint"):
            raise ValueError(
                f"Reranker checkpoint mismatch in final measured system '{final_measured_key}': "
                f"found '{final_summary['reranker_checkpoint']}' != selected '{selected_reranker.get('checkpoint')}'."
            )

    # Verify generator adapter identity
    if use_qlora and isinstance(final_summary, dict) and "adapter_path" in final_summary:
        if final_summary["adapter_path"] != selected_generator.get("adapter"):
            raise ValueError(
                f"Generator adapter mismatch in final measured system '{final_measured_key}': "
                f"found '{final_summary['adapter_path']}' != selected '{selected_generator.get('adapter')}'."
            )

    # 4. Step 8.5: Verify winning candidate exists in final summary and scores match
    if isinstance(final_summary, dict) and "candidate_family_meteors" in final_summary:
        cand_scores = final_summary["candidate_family_meteors"]
        if best_fixed not in cand_scores:
            raise ValueError(
                f"Winning candidate '{best_fixed}' is not present in candidate_family_meteors of {final_measured_key}."
            )
        measured_score = float(cand_scores[best_fixed])
        expected_score = float(report["overall_deployable_meteor"])
        if abs(measured_score - expected_score) > 1e-4:
            raise ValueError(
                f"Candidate score mismatch for '{best_fixed}': "
                f"measured={measured_score:.4f} != overall_deployable_meteor={expected_score:.4f}."
            )

    # Read base config for retrieval/evidence infrastructure settings
    base_config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f) or {}

    promoted_config: Dict[str, Any] = {
        "schema_version": 3,
        "screen_protocol_version": 8,
        "status": "PROMOTED",
        "source_screen_manifest": report_path,
        "source_screen_sha256": report_sha256,
        "stack": base_config.get("stack", "stack_a"),
        "retrieval": base_config.get("retrieval", {
            "sparse": {"method": "bm25s", "top_k": 50},
            "dense": {
                "model": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
                "dtype": "float16",
                "device": "cuda:1",
                "top_k": 50,
            },
            "fusion": {"method": "rrf", "rrf_k": 60},
        }),
        "reranker": {
            "base_model": "BAAI/bge-reranker-v2-m3",
            "use_task_tuned": use_tuned_rerank,
            "checkpoint": selected_reranker.get("checkpoint", "checkpoints/reranker/best"),
            "top_k": 8,
            "device": "cuda:1",
        },
        "evidence": {
            "primary_pack": "multi_seed_2500_chars",
            "max_chars": 3500,
        },
        "generator": {
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "use_qlora": use_qlora,
            "adapter_path": selected_generator.get("adapter", "checkpoints/generator/hf_adapter"),
            "max_new_tokens": 384,
            "device": "cuda:0",
        },
        "candidate_policy": {
            "type": cand_policy.get("type", "fixed_baseline"),
            "best_fixed_candidate": best_fixed,
            "selector_checkpoint": cand_policy.get("selector_checkpoint"),
            "guardrail_enabled": True,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(promoted_config, f, sort_keys=False, indent=2)

    print(f"\n================ PRODUCTION PROMOTION SUCCESS (Protocol 8) ================")
    print(f"Status:                      PROMOTED")
    print(f"Screen Protocol Version:     8")
    print(f"Source Report:               {report_path}")
    print(f"Report SHA256:               {report_sha256[:16]}...")
    print(f"Final Measured System:       {final_measured_key}")
    print(f"Use Task-Tuned Reranker:     {use_tuned_rerank}")
    print(f"Use QLoRA Generator:         {use_qlora}")
    print(f"Candidate Selection Policy:  fixed_baseline (best_fixed='{best_fixed}')")
    print(f"Saved Authoritative Config:  {output_path}")
    print("============================================================================")

    return promoted_config


def main():
    parser = argparse.ArgumentParser(description="Promote production configuration from screen report")
    parser.add_argument("--report", default="artifacts/task2/evaluations/promotion_report.json")
    parser.add_argument("--config", default="configs/production_selection.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    promote_production_selection(
        report_path=args.report,
        config_path=args.config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
