import json
import os
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


def load_config_file(config_path: str) -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("{") or config_path.endswith(".json"):
        return json.loads(content)
    elif yaml is not None:
        return yaml.safe_load(content) or {}
    else:
        try:
            return json.loads(content)
        except Exception:
            return {}


def audit_parameter_budget(
    config_path: str = "configs/models.yaml",
    extra_adapter_params: int = 0,
    adapter_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit all learned model parameters against the official < 4.0B hard budget."""
    config = load_config_file(config_path)
    models = config.get("models", [])
    total = 0
    breakdown = {}

    for m in models:
        if m.get("loaded_at_inference", True):
            p = int(m.get("parameters", 0))
            mid = m.get("model_id", "unknown")
            total += p
            breakdown[mid] = p

    if extra_adapter_params > 0:
        name = adapter_name or "lora_adapter"
        total += extra_adapter_params
        breakdown[name] = extra_adapter_params

    limit = int(config.get("parameter_budget", {}).get("maximum_exclusive", 4000000000))
    return {
        "total_learned_parameters": total,
        "limit": limit,
        "is_compliant": total < limit,
        "margin": limit - total,
        "breakdown": breakdown,
    }


def verify_config_consistency(
    pipeline_path: str = "configs/pipeline.yaml",
    models_path: str = "configs/models.yaml",
) -> Dict[str, Any]:
    """Verify that pipeline.yaml specifies the exact approved models in models.yaml."""
    pipe_cfg = load_config_file(pipeline_path)
    mod_cfg = load_config_file(models_path)

    pipe_dense = pipe_cfg.get("retrieval", {}).get("dense_model", "")
    pipe_reranker = pipe_cfg.get("reranking", {}).get("model", "")
    pipe_gen = pipe_cfg.get("generation", {}).get("model", "")

    approved_models = set(mod_cfg.get("recommended_stack", {}).get("model_ids", []))
    models_list = {m.get("model_id"): m for m in mod_cfg.get("models", [])}

    consistent = True
    issues = []

    if pipe_dense not in approved_models:
        consistent = False
        issues.append(f"Dense model in pipeline ({pipe_dense}) is not in approved models ({approved_models})")

    if pipe_reranker not in approved_models:
        consistent = False
        issues.append(f"Reranker in pipeline ({pipe_reranker}) is not in approved models ({approved_models})")

    if pipe_gen not in approved_models:
        consistent = False
        issues.append(f"Generator in pipeline ({pipe_gen}) is not in approved models ({approved_models})")

    return {
        "is_consistent": consistent,
        "pipeline_models": {
            "dense": pipe_dense,
            "reranker": pipe_reranker,
            "generator": pipe_gen,
        },
        "approved_models": list(approved_models),
        "issues": issues,
    }


def main():
    config_path = "configs/models.yaml"
    result = audit_parameter_budget(config_path)
    consistency = verify_config_consistency()

    print("=== LegalQA Model Parameter & Config Audit ===")
    print(f"Total learned parameters: {result['total_learned_parameters']:,}")
    print(f"Parameter budget limit:   {result['limit']:,}")
    print(f"Remaining safe margin:    {result['margin']:,} parameters")
    print(f"Compliance status:        {'COMPLIANT' if result['is_compliant'] else 'NON-COMPLIANT'}")
    print("Breakdown by model:")
    for k, v in result["breakdown"].items():
        print(f" - {k}: {v:,}")

    print("\n=== Config Consistency Check ===")
    if consistency["is_consistent"]:
        print("PASS: pipeline.yaml matches models.yaml approved stack.")
    else:
        print("FAIL: Inconsistencies detected:")
        for iss in consistency["issues"]:
            print(f" - {iss}")


if __name__ == "__main__":
    main()
