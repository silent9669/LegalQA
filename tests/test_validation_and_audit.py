from scripts.run_oof_validation import calculate_official_meteor
from scripts.audit_parameters import audit_parameter_budget

def test_calculate_official_meteor():
    ref = ["Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"]
    pred = ["Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"]
    score = calculate_official_meteor(ref, pred)
    assert score >= 0.99

def test_audit_parameter_budget():
    audit = audit_parameter_budget("configs/models.yaml")
    assert audit["total_learned_parameters"] < 4000000000
    assert audit["is_compliant"] is True
