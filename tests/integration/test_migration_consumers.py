"""Task 10: migration + consumer proof regressions."""

from __future__ import annotations

from scripts.audit_migration import build_rollback_bundle, compare_gate_reports, inventory_consumers


def test_migration_detects_lost_stage():
    r = compare_gate_reports({"stages": ["kaggle", "t4", "a100"]}, {"stages": ["kaggle", "a100"]})
    assert r["status"] == "FAIL"


def test_compatible_reports_pass_and_identity_mismatch_fails():
    old = {"stages": ["kaggle_t4x2", "colab_t4", "a100_micro_probe"], "status": "PASS",
           "candidate_sha": "c" * 16, "algorithm_sha256": "a" * 64}
    new = dict(old)
    assert compare_gate_reports(old, new)["status"] == "PASS"
    tampered = dict(new, algorithm_sha256="f" * 64)
    failed = compare_gate_reports(old, tampered)
    assert failed["status"] == "FAIL" and "algorithm_sha256" in failed["identity_mismatches"]


def test_consumer_inventory_finds_legacy_config_users():
    inventory = inventory_consumers(["scripts", "tests", "configs", "src", "notebooks"])
    assert inventory, "inventory must find tracked consumers"
    legacy_users = [f for f, hits in inventory.items() if "legacy_flat_config" in hits]
    assert legacy_users, "colab_train_a100.yaml consumers must be enumerated before any removal"
    assert any("run_gpu_gate" in f or "modal_app" in f for f in inventory), \
        "gate adapter consumers must be enumerated"
    assert not any("colab_remote_entry" in f or "launch_colab_training" in f for f in inventory), \
        "Colab launchers were removed; no consumers may remain"


def test_rollback_bundle_preserves_digests(tmp_path):
    target = tmp_path / "orig.yaml"
    target.write_text("profile_name: legacy\n", encoding="utf-8")
    dest = tmp_path / "rollback"
    record = build_rollback_bundle([str(target)], str(dest))
    assert (dest / "orig.yaml").is_file()
    assert (dest / "rollback_manifest.json").is_file()
    assert len(record["manifest_sha256"]) == 64
    assert "restore_command" in record
