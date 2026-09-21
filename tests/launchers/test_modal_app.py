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
    assert "predicted_inference_seconds" in paths


def test_remote_production_cfg_restores_generation_ceiling():
    from scripts.modal_app import MODAL_MAX_NEW_TOKENS, build_remote_production_cfg
    from src.task2.production_config import get_default_production_selection

    cfg = build_remote_production_cfg()
    assert cfg.max_new_tokens == MODAL_MAX_NEW_TOKENS == 1536
    assert cfg.best_fixed_candidate == "dual_assembled"
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


def test_modal_a100_micro_probe_profile_allows_batch4(tmp_path):
    """Verify that a100_micro_probe accepts modal_a100's batch_size=4 and does not force batch_size=1."""
    from unittest.mock import MagicMock, patch
    from src.task2.config.loader import load_resolved_config
    from src.task2.generation.trainer import train_generator_qlora

    cfg = load_resolved_config("configs/task2/algorithm.yaml", "configs/task2/runtime/modal_a100.yaml")
    assert cfg.runtime.generator_runtime.per_device_train_batch_size == 4

    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_model = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.state.global_step = 2
    mock_reloaded = MagicMock()
    mock_reloaded.generate.return_value = "Verified"

    with patch("src.task2.generation.trainer.AutoTokenizer.from_pretrained", return_value=mock_tok), \
         patch("src.task2.generation.trainer.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
         patch("src.task2.generation.trainer.SFTTrainer", return_value=mock_trainer), \
         patch("src.task2.generation.trainer.build_v16_sft_config", return_value=MagicMock()), \
         patch("src.task2.generation.trainer.build_grounded_training_examples", return_value=[
             {"prompt": "p", "completion": "c", "text": "p c", "qa_id": "1", "total_tokens": 10, "completion_tokens": 5}
         ]), \
         patch("src.task2.generation.trainer.QwenGenerator.load", return_value=mock_reloaded):

        # When probe_mode='worst_case' and resolved_config has profile_name='modal_a100',
        # it must succeed with batch_size=4 and not crash with batch_size=1 required.
        res = train_generator_qlora(
            model_name_or_path=cfg.algorithm.models.generator.id,
            qa_path="dummy_qa.parquet",
            labels_path="dummy_labels.parquet",
            chunks_path="dummy_chunks.parquet",
            output_dir=str(tmp_path / "probe_out"),
            resolved_config=cfg,
            max_steps=2,
            probe_mode="worst_case",
            execution_profile="modal_a100",
            device="cpu",
        )
        assert res["execution_profile"] == "modal_a100"
        assert res["strict_reload"] == "pass"



def test_modal_request_kaggle_t4x2_stage_contracts():
    req = build_modal_request("kaggle_t4x2", CAND, "private-official.json", None)
    assert req["stage"] == "kaggle_t4x2"
    validate_modal_request(req)

    with pytest.raises(ValueError, match="takes no parent report"):
        build_modal_request("kaggle_t4x2", CAND, "private-official.json", _parent("kaggle_t4x2"))
