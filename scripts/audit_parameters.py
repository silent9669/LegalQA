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


DEFAULT_STACK_A_MODELS = {
    "Qwen/Qwen2.5-3B-Instruct": 3086303232,
    "BAAI/bge-reranker-v2-m3": 567419904,
    # Canonical fine-tuned encoder (same PhoBERT 768-dim architecture as dek21).
    "runs/20260920-215402/encoder_ft_v2": 135168000,
    "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2": 135168000,
}


def audit_parameter_budget(
    config_path: str = "configs/task2/algorithm.yaml",
    stack: Optional[str] = "stack_a",
    adapter_manifest_path: Optional[str] = None,
    extra_adapter_params: int = 0,
    adapter_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit all learned model parameters against the official < 4.0B hard budget."""
    config = load_config_file(config_path)
    models = config.get("models", {})
    stacks = config.get("stacks", {})
    total = 0
    breakdown = {}

    if isinstance(models, dict) and "generator" in models and "reranker" in models:
        # Authoritative configs/task2/algorithm.yaml format
        gen_id = models.get("generator", {}).get("id", "Qwen/Qwen2.5-3B-Instruct")
        rerank_id = models.get("reranker", {}).get("id", "BAAI/bge-reranker-v2-m3")
        dense_id = models.get("dense", {}).get("id", "runs/20260920-215402/encoder_ft_v2")

        for mid in [gen_id, rerank_id, dense_id]:
            p = DEFAULT_STACK_A_MODELS.get(mid, 0)
            total += p
            breakdown[mid] = p
    elif stack and stack in stacks:
        stack_info = stacks[stack]
        target_model_ids = set(stack_info.get("model_ids", []))
        for m in (models if isinstance(models, list) else []):
            mid = m.get("model_id", "unknown")
            if mid in target_model_ids:
                p = int(m.get("parameters", 0))
                total += p
                breakdown[mid] = p
    elif isinstance(models, list) and models:
        for m in models:
            if m.get("loaded_at_inference", True):
                p = int(m.get("parameters", 0))
                mid = m.get("model_id", "unknown")
                total += p
                breakdown[mid] = p
    else:
        # Canonical Stack A baseline defaults (< 4.0B competition limit)
        for mid, p in DEFAULT_STACK_A_MODELS.items():
            total += p
            breakdown[mid] = p

    # Read trained adapter parameter count dynamically if manifest path is supplied
    if adapter_manifest_path and os.path.exists(adapter_manifest_path):
        try:
            with open(adapter_manifest_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            ad_p = int(man.get("adapter_trainable_params", 0))
            if ad_p > 0:
                name = adapter_name or "trained_qlora_adapter"
                total += ad_p
                breakdown[name] = ad_p
        except Exception:
            pass
    elif extra_adapter_params > 0:
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
    algorithm_path: str = "configs/task2/algorithm.yaml",
) -> Dict[str, Any]:
    """Verify that algorithm.yaml specifies approved models under the competition limits."""
    algo_cfg = load_config_file(algorithm_path)
    models = algo_cfg.get("models", {})
    gen_id = models.get("generator", {}).get("id")
    reranker_id = models.get("reranker", {}).get("id")
    dense_id = models.get("dense", {}).get("id")

    approved_models = set(DEFAULT_STACK_A_MODELS.keys())
    consistent = True
    issues = []

    for label, mid in [("Generator", gen_id), ("Reranker", reranker_id), ("Dense", dense_id)]:
        if not mid:
            consistent = False
            issues.append(f"{label} model ID is missing in {algorithm_path}")
        elif mid not in approved_models:
            consistent = False
            issues.append(f"{label} model ({mid}) is not in approved models list ({approved_models})")

    return {
        "is_consistent": consistent,
        "algorithm_models": {
            "generator": gen_id,
            "reranker": reranker_id,
            "dense": dense_id,
        },
        "approved_models": list(approved_models),
        "issues": issues,
    }


def main():
    config_path = "configs/task2/algorithm.yaml"
    for st in ["stack_a"]:
        result = audit_parameter_budget(config_path, stack=st)
        print(f"=== Stack '{st}' Parameter Audit ===")
        print(f"Total learned parameters: {result['total_learned_parameters']:,}")
        print(f"Parameter budget limit:   {result['limit']:,}")
        print(f"Remaining safe margin:    {result['margin']:,} parameters")
        print(f"Compliance status:        {'COMPLIANT' if result['is_compliant'] else 'NON-COMPLIANT'}")
        for k, v in result["breakdown"].items():
            print(f" - {k}: {v:,}")

    consistency = verify_config_consistency(config_path)
    print("\n=== Config Consistency Check ===")
    if consistency["is_consistent"]:
        print("PASS: algorithm.yaml matches approved models.")
    else:
        print("FAIL: Inconsistencies detected:")
        for iss in consistency["issues"]:
            print(f" - {iss}")


if __name__ == "__main__":
    main()
