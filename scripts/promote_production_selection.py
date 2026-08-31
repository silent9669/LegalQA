"""Deterministic production selection promoter from screen_fold0 evaluation report (V7)."""

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
    """Read measured promotion_report.json and freeze an authoritative PROMOTED configuration (Task 5)."""
    assert_no_secrets_in_workspace(Path.cwd())
    output_path = output_path or config_path

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Promotion report not found at: {report_path}")

    report_sha256 = sha256_file(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # 1. Step 5.1: Validate Protocol 7 and required report schema keys
    protocol_v = report.get("screen_protocol_version", 1)
    if protocol_v < 7:
        raise ValueError(
            f"Promotion report at {report_path} uses obsolete screen_protocol_version {protocol_v}. "
            f"Require >= 7 (staged component-consistent screening)."
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
    ]
    for k in required_keys:
        if k not in report:
            raise ValueError(f"Promotion report at {report_path} is missing required key: '{k}'")

    selected_reranker = report["selected_reranker"]
    selected_generator = report["selected_generator"]
    final_measured_key = report["final_measured_system_key"]
    cand_policy = report["candidate_policy"]
    best_fixed = cand_policy.get("best_fixed_candidate", "stitched_extract")

    # 2. Step 5.2: Cross-check component consistency
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

    if best_fixed in GENERATOR_DEPENDENT_FAMILIES and not use_qlora and final_measured_key == "R_SELECTED_G1":
        raise ValueError(
            f"Component inconsistency in promotion report: winning candidate '{best_fixed}' "
            f"is generator-dependent from a rejected QLoRA system."
        )

    # 3. Read base config for retrieval/evidence infrastructure settings
    base_config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f) or {}

    promoted_config: Dict[str, Any] = {
        "schema_version": 3,
        "screen_protocol_version": 7,
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

    print(f"\n================ PRODUCTION PROMOTION SUCCESS (Protocol 7) ================")
    print(f"Status:                      PROMOTED")
    print(f"Screen Protocol Version:     7")
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
