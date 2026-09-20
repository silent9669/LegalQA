"""Modal adapter regressions: request chain, test resolution, image pins."""

from __future__ import annotations

import json

import pytest

from scripts import modal_app as modal_app_module
from scripts.modal_app import (
    build_modal_request,
    resolve_test_file,
    validate_modal_request,
)

CAND = {
    "candidate_id": "c" * 16,
    "git_commit_sha": "a" * 40,
    "models": {"dense": {"id": "d", "revision": "b" * 40}},
}


def _parent(stage="colab_t4"):
    return {"status": "PASS", "candidate_sha": "c" * 16, "stage": stage, "report_sha256": "e" * 64}


def test_modal_request_requires_parent_chain():
    with pytest.raises(ValueError, match="unknown Modal stage"):
        build_modal_request("invalid_stage", CAND, "private-official.json", None)
    with pytest.raises(ValueError, match="no bypass"):
        build_modal_request("micro_probe", CAND, "private-official.json", None)
    with pytest.raises(ValueError, match="no bypass"):
        build_modal_request("full", CAND, "private-official.json", None)

    # Missing or short dense revision
    cand_no_dense = {"candidate_id": "c" * 16, "git_commit_sha": "a" * 40}
    with pytest.raises(ValueError, match="immutable 40-hex dense revision"):
        build_modal_request("micro_probe", cand_no_dense, "private-official.json", _parent("colab_t4"))
    cand_short_dense = {"candidate_id": "c" * 16, "git_commit_sha": "a" * 40, "models": {"dense": {"revision": "short"}}}
    with pytest.raises(ValueError, match="immutable 40-hex dense revision"):
        build_modal_request("micro_probe", cand_short_dense, "private-official.json", _parent("colab_t4"))

    # Non-PASS parent report
    non_pass = dict(_parent("colab_t4"), status="FAIL")
    with pytest.raises(ValueError, match="is not PASS"):
        build_modal_request("micro_probe", CAND, "private-official.json", non_pass)

    bad_stage = _parent("unapproved_stage")
    with pytest.raises(ValueError, match="stage mismatch"):
        build_modal_request("micro_probe", CAND, "private-official.json", bad_stage)
    bad_cand = _parent("colab_t4")
    bad_cand["candidate_sha"] = "f" * 16
    with pytest.raises(ValueError, match="candidate mismatch"):
        build_modal_request("micro_probe", CAND, "private-official.json", bad_cand)
    req = build_modal_request("micro_probe", CAND, "private-official.json", _parent("colab_t4"))
    assert req["dense_revision"] == "b" * 40
    validate_modal_request(req)
    req_kaggle = build_modal_request("micro_probe", CAND, "private-official.json", _parent("kaggle_t4x2"))
    validate_modal_request(req_kaggle)
    full = build_modal_request("full", CAND, "private-official.json", _parent("a100_micro_probe"))
    validate_modal_request(full)
    with pytest.raises(ValueError, match="unknown test file"):
        validate_modal_request(dict(full, test_filename="evil.json"))


def test_modal_app_imports_without_modal_package():
    import builtins
    import types
    from pathlib import Path

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "modal":
            raise ImportError("Mocked: modal not installed")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        mod_code = Path("scripts/modal_app.py").read_text(encoding="utf-8")
        fake_mod = types.ModuleType("scripts.modal_app_nomodal")
        fake_mod.__dict__["__file__"] = str(Path("scripts/modal_app.py").resolve())
        exec(compile(mod_code, "scripts/modal_app.py", "exec"), fake_mod.__dict__)
        assert fake_mod.modal is None
        req = fake_mod.build_modal_request("micro_probe", CAND, "private-official.json", _parent("colab_t4"))
        assert req["stage"] == "micro_probe"
    finally:
        builtins.__import__ = real_import


def test_modal_a100_profile_resolution_contracts():
    from src.task2.pipeline.profiles import resolve_execution_profile
    prof = resolve_execution_profile("modal_a100")
    assert prof.name == "modal_a100"
    assert prof.run_generator_training is True
    assert prof.run_public_inference is True
    assert prof.val_fold is None
    assert prof.requires_generator is True


def test_resolve_test_file_falls_back_and_refuses_unknown(tmp_path):
    (tmp_path / "public-official.json").write_text('{"q1": {"question": "Q", "answer": "A"}}', encoding="utf-8")
    assert resolve_test_file(tmp_path, "public-official.json").name == "public-official.json"
    assert resolve_test_file(tmp_path, "private-official.json").name == "public-official.json"
    with pytest.raises(ValueError, match="unknown test file"):
        resolve_test_file(tmp_path, "evil.json")
    with pytest.raises(FileNotFoundError):
        resolve_test_file(tmp_path / "empty", "private-official.json")


def test_modal_fingerprint_counts_and_hashes(tmp_path):
    payload = {"q1": {"question": "Q?"}, "q2": {"question": "W?"}}
    p = tmp_path / "private-official.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    fp = modal_app_module.test_file_fingerprint(p)
    assert fp["num_queries"] == 2 and len(fp["sha256"]) == 64
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        modal_app_module.test_file_fingerprint(empty)


def test_repo_root_survives_unreadable_system_paths():
    """CI runners cannot stat /root: module import must fall back, not raise."""
    import types
    from pathlib import Path
    from unittest import mock

    real_is_dir = Path.is_dir

    def guarded_is_dir(self):
        if str(self) == "/root/LegalQA" or str(self).startswith("/root/"):
            raise PermissionError(13, "Permission denied")
        return real_is_dir(self)

    with mock.patch.object(Path, "is_dir", autospec=True, side_effect=lambda self: guarded_is_dir(self)):
        mod_code = Path("scripts/modal_app.py").read_text(encoding="utf-8")
        fake_mod = types.ModuleType("scripts.modal_app_noroot")
        fake_mod.__dict__["__file__"] = str(Path("scripts/modal_app.py").resolve())
        exec(compile(mod_code, "scripts/modal_app.py", "exec"), fake_mod.__dict__)
        assert fake_mod.REPO_ROOT == Path("scripts/modal_app.py").resolve().parent.parent
        assert fake_mod.read_pin_file("constraints-gpu.txt") != []


def test_remote_paths_carry_deadline_and_model():
    from scripts.modal_app import MODAL_DEADLINE_BUDGET_SECONDS, build_remote_paths

    paths = build_remote_paths("/data", "/data/idx", "/data/t.json", qwen_model_path="Qwen/Qwen2.5-3B-Instruct")
    assert paths["dek21_dir"] == "/data/idx"
    assert paths["qwen_model_path"] == "Qwen/Qwen2.5-3B-Instruct"
    assert int(paths["deadline_budget_seconds"]) == MODAL_DEADLINE_BUDGET_SECONDS == 17100


def test_remote_production_cfg_restores_generation_ceiling():
    from scripts.modal_app import MODAL_MAX_NEW_TOKENS, build_remote_production_cfg
    from src.task2.production_config import get_default_production_selection

    cfg = build_remote_production_cfg()
    assert cfg.max_new_tokens == MODAL_MAX_NEW_TOKENS == 1400
    assert cfg.max_new_tokens != get_default_production_selection().max_new_tokens


def test_modal_request_colab_t4_stage_contracts():
    from scripts.modal_app import build_modal_request, validate_modal_request

    # colab_t4 requires kaggle_t4x2 parent report
    req = build_modal_request("colab_t4", CAND, "private-official.json", _parent("kaggle_t4x2"))
    assert req["stage"] == "colab_t4"
    validate_modal_request(req)

    # Rejects invalid parents (e.g. colab_t4 or a100_micro_probe for colab_t4 stage)
    with pytest.raises(ValueError, match="stage mismatch"):
        build_modal_request("colab_t4", CAND, "private-official.json", _parent("colab_t4"))

    with pytest.raises(ValueError, match="no bypass"):
        build_modal_request("colab_t4", CAND, "private-official.json", None)



def test_modal_colab_t4_stage_takes_kaggle_parent():
    req = build_modal_request("colab_t4", CAND, "private-official.json", _parent("kaggle_t4x2"))
    assert req["stage"] == "colab_t4"
    validate_modal_request(req)
    with pytest.raises(ValueError, match="stage mismatch"):
        build_modal_request("colab_t4", CAND, "private-official.json", _parent("colab_t4"))
