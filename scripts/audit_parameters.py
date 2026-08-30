import json
import os

try:
    import yaml
except ImportError:
    yaml = None

def audit_parameter_budget(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("{") or config_path.endswith(".json"):
        config = json.loads(content)
    elif yaml is not None:
        config = yaml.safe_load(content)
    else:
        # Fallback simple json-like or error
        try:
            config = json.loads(content)
        except Exception:
            config = {}

    models = config.get("models", [])
    total = 0
    breakdown = {}

    for m in models:
        if m.get("loaded_at_inference", True):
            p = m.get("parameters", 0)
            mid = m.get("model_id", "unknown")
            total += p
            breakdown[mid] = p

    limit = config.get("parameter_budget", {}).get("maximum_exclusive", 4000000000)
    return {
        "total_learned_parameters": total,
        "limit": limit,
        "is_compliant": total < limit,
        "breakdown": breakdown
    }

def main():
    config_path = "configs/models.yaml"
    result = audit_parameter_budget(config_path)
    print("=== LegalQA Model Parameter Audit ===")
    print(f"Total learned parameters: {result['total_learned_parameters']:,}")
    print(f"Parameter budget limit:   {result['limit']:,}")
    print(f"Compliance status:        {'COMPLIANT' if result['is_compliant'] else 'NON-COMPLIANT'}")
    print("Breakdown by model:")
    for k, v in result["breakdown"].items():
        print(f" - {k}: {v:,}")

if __name__ == "__main__":
    main()
