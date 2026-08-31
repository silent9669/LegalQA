"""Deterministic production selection promoter from screen_fold0 evaluation report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.hashing import sha256_file
from src.common.security import assert_no_secrets_in_workspace


def promote_production_selection(
    report_path: str = "artifacts/task2/evaluations/promotion_report.json",
    config_path: str = "configs/production_selection.yaml",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Read measured promotion_report.json and freeze an authoritative PROMOTED configuration."""
    assert_no_secrets_in_workspace(Path.cwd())
    output_path = output_path or config_path

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Promotion report not found at: {report_path}")

    report_sha256 = sha256_file(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Validate report completeness
    required_keys = [
        "held_out_fold",
        "sample_ids_sha256",
        "recommended_use_task_tuned_reranker",
        "recommended_use_qlora",
        "candidate_policy",
    ]
    for k in required_keys:
        if k not in report:
            raise ValueError(f"Promotion report at {report_path} is missing required key: {k}")

    # Read base config if present
    base_config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f) or {}

    use_task_tuned_reranker = bool(report["recommended_use_task_tuned_reranker"])
    use_qlora = bool(report["recommended_use_qlora"])
    cand_policy = report["candidate_policy"]
    policy_type = cand_policy.get("type", "fixed_baseline")
    best_fixed = cand_policy.get("best_fixed_candidate", report.get("overall_deployable_winner", "stitched_extract"))

    promoted_config: Dict[str, Any] = {
        "schema_version": 3,
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
            "use_task_tuned": use_task_tuned_reranker,
            "checkpoint": "checkpoints/reranker/best",
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
            "adapter_path": "checkpoints/generator/hf_adapter",
            "max_new_tokens": 384,
            "device": "cuda:0",
        },
        "candidate_policy": {
            "type": policy_type,
            "best_fixed_candidate": best_fixed,
            "selector_checkpoint": None,
            "guardrail_enabled": True,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(promoted_config, f, sort_keys=False, indent=2)

    print(f"\n================ PRODUCTION PROMOTION SUCCESS ================")
    print(f"Status:                      PROMOTED")
    print(f"Source Report:               {report_path}")
    print(f"Report SHA256:               {report_sha256[:16]}...")
    print(f"Use Task-Tuned Reranker:     {use_task_tuned_reranker}")
    print(f"Use QLoRA Generator:         {use_qlora}")
    print(f"Candidate Selection Policy:  {policy_type} (best_fixed='{best_fixed}')")
    print(f"Saved Authoritative Config:  {output_path}")
    print("==============================================================")

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
