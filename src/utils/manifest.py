MAX_LEARNED_PARAMETERS = 4_000_000_000

def audit_parameter_manifest(components: dict) -> tuple[int, bool]:
    """
    Audits learned parameter counts of all system components against the < 4B limit.
    """
    total_params = sum(comp.get("parameters", 0) for comp in components.values())
    is_valid = total_params < MAX_LEARNED_PARAMETERS
    return total_params, is_valid
