from scripts.run_oof_validation import calculate_official_meteor
from scripts.audit_parameters import audit_parameter_budget, verify_config_consistency


def test_calculate_official_meteor():
    ref = ["Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"]
    pred = ["Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"]
    score = calculate_official_meteor(ref, pred)
    assert score >= 0.99


def test_audit_parameter_budget():
    audit = audit_parameter_budget("configs/models.yaml")
    assert audit["total_learned_parameters"] < 4000000000
    assert audit["is_compliant"] is True
    assert audit["margin"] > 200000000  # at least 200M margin for LoRA/adapter


def test_audit_with_lora_adapter():
    lora_params = 18800000  # ~18.8M for r=16
    audit = audit_parameter_budget("configs/models.yaml", extra_adapter_params=lora_params, adapter_name="qlora_adapter")
    assert audit["total_learned_parameters"] < 4000000000
    assert audit["is_compliant"] is True
    assert "qlora_adapter" in audit["breakdown"]


def test_config_consistency():
    consistency = verify_config_consistency("configs/pipeline.yaml", "configs/models.yaml")
    assert consistency["is_consistent"] is True, f"Config mismatch: {consistency['issues']}"
