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
    stack: Optional[str] = "stack_a",
    extra_adapter_params: int = 0,
    adapter_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit all learned model parameters against the official < 4.0B hard budget."""
    config = load_config_file(config_path)
    models = config.get("models", [])
    stacks = config.get("stacks", {})
    total = 0
    breakdown = {}

    if stack and stack in stacks:
        stack_info = stacks[stack]
        target_model_ids = set(stack_info.get("model_ids", []))
        for m in models:
            mid = m.get("model_id", "unknown")
            if mid in target_model_ids:
                p = int(m.get("parameters", 0))
                total += p
                breakdown[mid] = p
    else:
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
        "stack": stack,
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
    """Verify that pipeline.yaml specifies valid models present in models.yaml."""
    pipe_cfg = load_config_file(pipeline_path)
    mod_cfg = load_config_file(models_path)

    pipe_dense_a = pipe_cfg.get("retrieval", {}).get("dense", {}).get("stack_a_model") or pipe_cfg.get("retrieval", {}).get("dense_model", "")
    pipe_dense_b = pipe_cfg.get("retrieval", {}).get("dense", {}).get("stack_b_model", "")
    pipe_reranker = pipe_cfg.get("reranker", {}).get("model") or pipe_cfg.get("reranking", {}).get("model", "")
    pipe_gen_a = pipe_cfg.get("generation", {}).get("stack_a_model") or pipe_cfg.get("generation", {}).get("model", "")
    pipe_gen_b = pipe_cfg.get("generation", {}).get("stack_b_model", "")

    all_models = {m.get("model_id") for m in mod_cfg.get("models", [])}

    consistent = True
    issues = []

    for label, m_id in [
        ("Dense Stack A", pipe_dense_a),
        ("Dense Stack B", pipe_dense_b),
        ("Reranker", pipe_reranker),
        ("Generator Stack A", pipe_gen_a),
        ("Generator Stack B", pipe_gen_b),
    ]:
        if m_id and m_id not in all_models:
            consistent = False
            issues.append(f"{label} in pipeline ({m_id}) is not in models.yaml ({all_models})")

    return {
        "is_consistent": consistent,
        "pipeline_models": {
            "dense_a": pipe_dense_a,
            "dense_b": pipe_dense_b,
            "reranker": pipe_reranker,
            "generator_a": pipe_gen_a,
            "generator_b": pipe_gen_b,
        },
        "approved_models": list(all_models),
        "issues": issues,
    }


def main():
    config_path = "configs/models.yaml"
    for st in ["stack_a", "stack_b"]:
        result = audit_parameter_budget(config_path, stack=st)
        print(f"=== Stack '{st}' Parameter Audit ===")
        print(f"Total learned parameters: {result['total_learned_parameters']:,}")
        print(f"Parameter budget limit:   {result['limit']:,}")
        print(f"Remaining safe margin:    {result['margin']:,} parameters")
        print(f"Compliance status:        {'COMPLIANT' if result['is_compliant'] else 'NON-COMPLIANT'}")
        for k, v in result["breakdown"].items():
            print(f" - {k}: {v:,}")

    consistency = verify_config_consistency()
    print("\n=== Config Consistency Check ===")
    if consistency["is_consistent"]:
        print("PASS: pipeline.yaml matches models.yaml approved models.")
    else:
        print("FAIL: Inconsistencies detected:")
        for iss in consistency["issues"]:
            print(f" - {iss}")


if __name__ == "__main__":
    main()
