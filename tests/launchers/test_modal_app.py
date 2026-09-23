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


def _adapter_spec(**overrides):
    spec = {
        "repo": "dangphuc2109/legalqa-qwen2.5-3b-adapter",
        "revision": "b" * 40,
        "subfolder": "runs/run_5433e8b4787137c9_20260920_193355/final_adapter",
        "base_revision": "a" * 40,
        "file_digests": {"adapter_model.safetensors": "ab" * 32},
    }
    spec.update(overrides)
    return spec


def _write_adapter_fixture(path, base_model="Qwen/Qwen2.5-3B-Instruct", **manifest_overrides):
    from pathlib import Path as _P

    target = _P(path)
    target.mkdir(parents=True, exist_ok=True)
    (target / "adapter_model.safetensors").write_bytes(b"fixture-weights-5433")
    (target / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    manifest = {
        "is_final_checkpoint": True,
        "smoke_only": False,
        "training_scope": "all_allowed_task2_data",
        "val_fold": None,
        "base_model": base_model,
        "optimizer_steps": 936,
        "dataset_size": 7483,
        "num_train_epochs": 2,
    }
    manifest.update(manifest_overrides)
    (target / "generator_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    import hashlib as _h

    return {
        "repo": "dangphuc2109/legalqa-qwen2.5-3b-adapter",
        "revision": "b" * 40,
        "subfolder": "runs/run_5433e8b4787137c9_20260920_193355/final_adapter",
        "base_revision": "a" * 40,
        "file_digests": {
            "adapter_model.safetensors": _h.sha256(b"fixture-weights-5433").hexdigest(),
        },
    }


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
    full = build_modal_request(
        "full", CAND, "private-official.json", _parent("a100_micro_probe"),
        generator_mode="reuse", adapter_spec=_adapter_spec(),
    )
    validate_modal_request(full)
    assert full["upload_policy"] == "disabled"  # experiments never upload by default
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


# ----------------------------------------------------------------------
# P0-A: explicit generator mode + adapter provenance (reuse-first)
# ----------------------------------------------------------------------

def test_r0_5433_baseline_pins_validate():
    """Lock the R0 baseline: 5433 adapter pins + measured digests (P0-A)."""
    from src.task2.provenance.reuse_contract import (
        R0_ADAPTER_FILE_DIGESTS,
        R0_ADAPTER_REPO,
        R0_ADAPTER_REVISION,
        R0_ADAPTER_SUBFOLDER,
        r0_adapter_spec,
        validate_adapter_spec,
    )

    assert R0_ADAPTER_REPO == "dangphuc2109/legalqa-qwen2.5-3b-adapter"
    assert R0_ADAPTER_REVISION == "b6e86e35e20c403bb82b40b25f85690c987e1d02"
    assert R0_ADAPTER_SUBFOLDER == "runs/run_5433e8b4787137c9_20260920_193355/final_adapter"
    assert R0_ADAPTER_FILE_DIGESTS["adapter_model.safetensors"] == (
        "dd5af2848f23234e7bd7cb7987b0b199c6832a4f5fde5dbe59a3e21ee379484e"
    )
    spec = validate_adapter_spec(r0_adapter_spec())
    assert spec["subfolder"] == R0_ADAPTER_SUBFOLDER
    req = build_modal_request(
        "full", CAND, "private-official.json", _parent("a100_micro_probe"),
        generator_mode="reuse", adapter_spec=r0_adapter_spec(),
    )
    assert req["adapter_spec"]["revision"] == R0_ADAPTER_REVISION
    validate_modal_request(req)


def test_full_requires_explicit_generator_mode():
    with pytest.raises(ValueError, match="generator_mode"):
        build_modal_request("full", CAND, "private-official.json", _parent("a100_micro_probe"))
    with pytest.raises(ValueError, match="generator_mode"):
        build_modal_request(
            "full", CAND, "private-official.json", _parent("a100_micro_probe"),
            generator_mode="auto",
        )
    fresh = build_modal_request(
        "full", CAND, "private-official.json", _parent("a100_micro_probe"),
        generator_mode="fresh",
    )
    assert fresh["generator_mode"] == "fresh" and fresh["adapter_spec"] is None
    validate_modal_request(fresh)


def test_reuse_full_requires_pinned_adapter_spec():
    with pytest.raises(ValueError, match="adapter_spec"):
        build_modal_request(
            "full", CAND, "private-official.json", _parent("a100_micro_probe"),
            generator_mode="reuse",
        )
    with pytest.raises(ValueError, match="40-hex revision"):
        build_modal_request(
            "full", CAND, "private-official.json", _parent("a100_micro_probe"),
            generator_mode="reuse", adapter_spec=_adapter_spec(revision="main"),
        )
    with pytest.raises(ValueError, match="file_digests"):
        build_modal_request(
            "full", CAND, "private-official.json", _parent("a100_micro_probe"),
            generator_mode="reuse", adapter_spec=_adapter_spec(file_digests={}),
        )
    req = build_modal_request(
        "full", CAND, "private-official.json", _parent("a100_micro_probe"),
        generator_mode="reuse", adapter_spec=_adapter_spec(),
    )
    assert req["adapter_spec"]["revision"] == "b" * 40
    validate_modal_request(req)


def test_resolve_adapter_plan_blocks_auto_copy_on_fresh():
    from scripts.modal_app import resolve_adapter_plan

    fresh = resolve_adapter_plan("fresh")
    assert fresh["allowed_sources"] == [] and fresh["trainer_runs"] is True
    reuse = resolve_adapter_plan("reuse")
    assert reuse["allowed_sources"] == ["pinned_hf_snapshot"] and reuse["trainer_runs"] is False
    with pytest.raises(ValueError, match="generator_mode"):
        resolve_adapter_plan("auto")


def test_stage_verified_adapter_copies_and_verifies(tmp_path):
    from scripts.modal_app import stage_verified_adapter

    src = tmp_path / "snapshot" / "final_adapter"
    spec = _write_adapter_fixture(src)
    dst = tmp_path / "run" / "checkpoints" / "generator" / "hf_adapter"
    report = stage_verified_adapter(src, dst, spec, "Qwen/Qwen2.5-3B-Instruct")
    assert report["verified"] is True
    assert (dst / "adapter_model.safetensors").is_file()

    # 1-byte tamper in the snapshot fails closed (no fallback source).
    (src / "adapter_model.safetensors").write_bytes(b"fixture-weights-d262")
    with pytest.raises(ValueError, match="digest mismatch"):
        stage_verified_adapter(src, tmp_path / "run2" / "hf_adapter", spec, "Qwen/Qwen2.5-3B-Instruct")

    # Wrong base / scope / fold / smoke fixtures all refuse reuse.
    spec2 = _write_adapter_fixture(tmp_path / "snap_base" / "final_adapter")
    with pytest.raises(ValueError, match="base model mismatch"):
        stage_verified_adapter(
            tmp_path / "snap_base" / "final_adapter", tmp_path / "run3" / "hf_adapter",
            spec2, "Qwen/Other-Base",
        )
    _write_adapter_fixture(tmp_path / "snap_scope" / "f", training_scope="smoke_subset")
    with pytest.raises(ValueError, match="training_scope"):
        stage_verified_adapter(
            tmp_path / "snap_scope" / "f", tmp_path / "run4" / "hf_adapter",
            spec, "Qwen/Qwen2.5-3B-Instruct",
        )
    _write_adapter_fixture(tmp_path / "snap_fold" / "f", val_fold=0)
    with pytest.raises(ValueError, match="val_fold"):
        stage_verified_adapter(
            tmp_path / "snap_fold" / "f", tmp_path / "run5" / "hf_adapter",
            spec, "Qwen/Qwen2.5-3B-Instruct",
        )
    _write_adapter_fixture(tmp_path / "snap_smoke" / "f", smoke_only=True, is_final_checkpoint=True)
    with pytest.raises(ValueError, match="smoke"):
        stage_verified_adapter(
            tmp_path / "snap_smoke" / "f", tmp_path / "run6" / "hf_adapter",
            spec, "Qwen/Qwen2.5-3B-Instruct",
        )


def test_reuse_runner_skips_trainer_only_after_validation(tmp_path):
    from src.task2.pipeline.runner import resolve_generator_training

    staged = tmp_path / "qlora_out"
    spec = _write_adapter_fixture(staged)
    calls = []

    def fake_train(**kwargs):
        calls.append(kwargs)
        return {"status": "completed", "optimizer_steps": 10, "dataset_size": 20}

    res = resolve_generator_training(
        qlora_out=str(staged), generator_mode="reuse", expected_adapter=spec,
        expected_base_model="Qwen/Qwen2.5-3B-Instruct", is_smoke=False,
        train_kwargs={}, train_fn=fake_train,
    )
    assert calls == []  # trainer never runs on verified reuse
    assert res["training_performed"] is False
    assert res["optimizer_steps"] == 0  # this run trained nothing
    assert res["source_adapter"]["optimizer_steps"] == 936  # source figures stay namespaced
    assert res["source_adapter"]["revision"] == "b" * 40

    # Missing staged adapter: reuse fails, trainer still not called.
    with pytest.raises(ValueError, match="no verified adapter"):
        resolve_generator_training(
            qlora_out=str(tmp_path / "absent"), generator_mode="reuse", expected_adapter=spec,
            expected_base_model="Qwen/Qwen2.5-3B-Instruct", is_smoke=False,
            train_kwargs={}, train_fn=fake_train,
        )
    assert calls == []

    # Tampered weights: reuse fails, trainer still not called.
    (staged / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        resolve_generator_training(
            qlora_out=str(staged), generator_mode="reuse", expected_adapter=spec,
            expected_base_model="Qwen/Qwen2.5-3B-Instruct", is_smoke=False,
            train_kwargs={}, train_fn=fake_train,
        )
    assert calls == []


def test_fresh_runner_always_calls_trainer(tmp_path):
    from src.task2.pipeline.runner import resolve_generator_training

    staged = tmp_path / "qlora_out"
    _write_adapter_fixture(staged)  # pre-existing weights must NOT short-circuit fresh
    calls = []

    def fake_train(**kwargs):
        calls.append(kwargs)
        return {"status": "completed", "optimizer_steps": 5, "dataset_size": 8}

    res = resolve_generator_training(
        qlora_out=str(staged), generator_mode="fresh", expected_adapter=None,
        expected_base_model="Qwen/Qwen2.5-3B-Instruct", is_smoke=False,
        train_kwargs={}, train_fn=fake_train,
    )
    assert len(calls) == 1  # trainer ran despite staged weights
    assert res["training_performed"] is True


# ----------------------------------------------------------------------
# P0-C: terminal status + submission parity (no silent PASS)
# ----------------------------------------------------------------------

def _write_submission_fixture(path, ids=("1", "2")):
    from pathlib import Path as _P
    import zipfile as _z

    payload = {i: {"answer": f"Answer text number {i} with enough words to be valid"} for i in ids}
    loose = _P(path)
    loose.write_text(json.dumps(payload), encoding="utf-8")
    zpath = loose.parent / (loose.name + ".zip")
    with _z.ZipFile(zpath, "w", _z.ZIP_DEFLATED) as z:
        z.write(loose, arcname="submission.json")
    return loose, zpath, payload


def test_decide_full_compute_status_gates(tmp_path):
    from scripts.modal_app import decide_full_compute_status

    loose, zpath, _ = _write_submission_fixture(tmp_path / "submission.json")
    ok_outputs = {"stages": {"submission": {"submission_json": str(loose)}}}
    ok = decide_full_compute_status(ok_outputs, loose, zpath, ["1", "2"])
    assert ok["compute_status"] == "PASS"

    inc = decide_full_compute_status({"status": "INCOMPLETE", "stages": {}}, loose, zpath, ["1", "2"])
    assert inc["compute_status"] == "INCOMPLETE"

    missing = decide_full_compute_status({"stages": {}}, loose, zpath, ["1", "2"])
    assert missing["compute_status"] == "FAIL"

    id_mismatch = decide_full_compute_status(ok_outputs, loose, zpath, ["1", "3"])
    assert id_mismatch["compute_status"] == "FAIL"

    # ZIP/member byte mismatch is not a PASS.
    import zipfile as _z

    bad_zip = tmp_path / "bad.zip"
    with _z.ZipFile(bad_zip, "w", _z.ZIP_DEFLATED) as z:
        z.writestr("submission.json", '{"1": {"answer": "different"}}')
    tampered = decide_full_compute_status(ok_outputs, loose, bad_zip, ["1", "2"])
    assert tampered["compute_status"] == "FAIL"

    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    empty_zip = tmp_path / "empty.json.zip"
    with _z.ZipFile(empty_zip, "w", _z.ZIP_DEFLATED) as z:
        z.write(empty, arcname="submission.json")
    assert decide_full_compute_status(ok_outputs, empty, empty_zip, [])["compute_status"] == "FAIL"


def test_resolve_test_file_strict_refuses_silent_fallback(tmp_path):
    (tmp_path / "public-official.json").write_text('{"q1": {"question": "Q"}}', encoding="utf-8")
    assert resolve_test_file(tmp_path, "public-official.json").name == "public-official.json"
    with pytest.raises(FileNotFoundError, match="no fallback"):
        resolve_test_file(tmp_path, "private-official.json", allow_fallback=False)


# ----------------------------------------------------------------------
# P0-D: source identity + gate policy (no synthetic PASS, no auto-pick)
# ----------------------------------------------------------------------

def test_skip_parent_check_never_synthesizes_pass():
    req = build_modal_request("micro_probe", CAND, "private-official.json", None, skip_parent_check=True)
    assert req["parent_report"]["status"] == "BYPASSED_EXPLICIT"
    assert req["parent_report"]["status"] != "PASS"
    assert req["parent_policy"] == "bypass_explicit"

    full = build_modal_request(
        "full", CAND, "private-official.json", None, skip_parent_check=True,
        generator_mode="reuse", adapter_spec=_adapter_spec(),
    )
    assert full["parent_report"]["status"] == "BYPASSED_EXPLICIT"
    assert full["parent_policy"] == "bypass_explicit"


def test_launch_selection_rejects_auto_pick_on_reuse():
    from src.task2.provenance.reuse_contract import decide_launch_selection

    with pytest.raises(ValueError, match="explicit --candidate"):
        decide_launch_selection(
            stage="full", generator_mode="reuse", candidate_arg="",
            parent_arg="p.json", skip_parent_check=False,
            available_candidates=["auto/manifest.json"],
        )
    with pytest.raises(ValueError, match="explicit --parent-report"):
        decide_launch_selection(
            stage="full", generator_mode="reuse", candidate_arg="c.json",
            parent_arg="", skip_parent_check=False, available_candidates=[],
        )
    ok = decide_launch_selection(
        stage="full", generator_mode="reuse", candidate_arg="c.json",
        parent_arg="", skip_parent_check=True, available_candidates=[],
    )
    assert ok["parent_policy"] == "bypass_explicit"
    legacy = decide_launch_selection(
        stage="micro_probe", generator_mode="", candidate_arg="",
        parent_arg="", skip_parent_check=False,
        available_candidates=["auto/manifest.json"],
    )
    assert legacy["candidate_path"] == "auto/manifest.json"


def test_source_identity_keeps_candidate_and_executed_separate():
    from src.task2.provenance.reuse_contract import build_source_identity

    rec = build_source_identity(
        {"candidate_id": "c" * 16, "git_commit_sha": "a" * 40, "algorithm_sha256": "d" * 64},
        "e" * 40, True, {"generator_mode": "reuse"},
    )
    assert rec["candidate_git_sha"] == "a" * 40
    assert rec["executed_git_sha"] == "e" * 40
    assert rec["git_match"] is False  # drift is recorded, never equated
    same = build_source_identity(
        {"candidate_id": "c" * 16, "git_commit_sha": "a" * 40}, "a" * 40, False, None,
    )
    assert same["git_match"] is True


# ----------------------------------------------------------------------
# Review blockers: encoder cache SHA, baked image identity, preload
# ----------------------------------------------------------------------

def _write_encoder_fixture(path, weights=b"encoder-bytes-v1"):
    from pathlib import Path as _P

    target = _P(path)
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.safetensors").write_bytes(weights)
    (target / "config.json").write_text('{"model_type": "roberta"}', encoding="utf-8")
    import hashlib as _h

    return _h.sha256(weights).hexdigest()


def test_verify_encoder_weights_against_independent_sha(tmp_path):
    from scripts.modal_app import R0_ENCODER_WEIGHTS_SHA256, verify_encoder_weights

    assert len(R0_ENCODER_WEIGHTS_SHA256) == 64  # pinned release value, not cache-derived
    good = tmp_path / "enc_good"
    observed = _write_encoder_fixture(good)
    report = verify_encoder_weights(good, observed)
    assert report == {"verified": True, "weights_sha256": observed}

    # Present-but-wrong cache (stale same-family weights) is refused, and
    # the cache's own digest never becomes the expectation.
    bad = tmp_path / "enc_bad"
    _write_encoder_fixture(bad, weights=b"stale-encoder-bytes")
    with pytest.raises(ValueError, match="weights SHA .* != expected"):
        verify_encoder_weights(bad, observed)

    with pytest.raises(FileNotFoundError, match="weights bytes missing"):
        verify_encoder_weights(tmp_path / "absent", observed)
    with pytest.raises(ValueError, match="64-hex"):
        verify_encoder_weights(good, "main")


def test_fetch_pinned_encoder_pins_revision_and_verifies(tmp_path):
    from scripts.modal_app import fetch_pinned_encoder

    snapshot = tmp_path / "snapshot"
    _write_encoder_fixture(snapshot / "runs/20260920-215402/encoder_ft_v2")
    calls = []

    def fake_download(repo_id=None, revision=None, allow_patterns=None):
        calls.append({"repo_id": repo_id, "revision": revision, "allow_patterns": allow_patterns})
        return str(snapshot)

    import hashlib as _h

    expected = _h.sha256(b"encoder-bytes-v1").hexdigest()
    report = fetch_pinned_encoder(
        fake_download, repo="org/enc", revision="d" * 40,
        subfolder="runs/20260920-215402/encoder_ft_v2",
        target_dir=tmp_path / "staged", expected_sha256=expected,
    )
    assert calls[0]["revision"] == "d" * 40  # revision pin reaches the downloader
    assert report["verified"] is True and report["revision"] == "d" * 40
    assert (tmp_path / "staged" / "model.safetensors").is_file()

    # Floating revision never reaches the network.
    with pytest.raises(ValueError, match="floating"):
        fetch_pinned_encoder(
            fake_download, repo="org/enc", revision="main",
            subfolder="runs/20260920-215402/encoder_ft_v2",
            target_dir=tmp_path / "staged2", expected_sha256=expected,
        )

    # Mismatched bytes inside a pinned snapshot still fail.
    (snapshot / "runs/20260920-215402/encoder_ft_v2" / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="weights SHA"):
        fetch_pinned_encoder(
            fake_download, repo="org/enc", revision="d" * 40,
            subfolder="runs/20260920-215402/encoder_ft_v2",
            target_dir=tmp_path / "staged3", expected_sha256=expected,
        )


def test_client_source_identity_shape():
    from scripts.modal_app import _CLIENT_SOURCE_IDENTITY, client_source_identity

    assert _CLIENT_SOURCE_IDENTITY.count(" ") == 1
    live = client_source_identity()
    sha, state = live.split(" ")
    assert (len(sha) == 40 or sha == "unknown") and state in ("clean", "dirty", "unknown")


def test_executed_identity_prefers_baked_file_without_git(tmp_path):
    from src.task2.provenance.reuse_contract import build_source_identity, get_executed_git_identity

    baked = tmp_path / ".image_source_sha"
    baked.write_text("f" * 40 + " clean", encoding="utf-8")
    ident = get_executed_git_identity(repo_root="/nonexistent-root-xyz", image_source_file=baked)
    assert ident == {"executed_git_sha": "f" * 40, "dirty": False, "source": "image_baked"}

    missing = get_executed_git_identity(
        repo_root="/nonexistent-root-xyz", image_source_file=tmp_path / "absent",
    )
    assert missing["executed_git_sha"] == "unknown" and missing["source"] == "unknown"
    # Unknown is never backfilled with the candidate SHA.
    rec = build_source_identity({"candidate_id": "c" * 16, "git_commit_sha": "a" * 40}, "unknown", "unknown", None)
    assert rec["executed_git_sha"] == "unknown" and rec["git_match"] is False


def test_find_verified_preload_source_rejects_arbitrary_adapters(tmp_path):
    from src.task2.provenance.reuse_contract import find_verified_preload_source, r0_adapter_spec

    good = tmp_path / "run_good" / "hf_adapter"
    spec = _write_adapter_fixture(good)
    wrong = tmp_path / "run_d261" / "hf_adapter"  # stale prior run, digest mismatch
    _write_adapter_fixture(wrong, base_model="Qwen/Qwen2.5-3B-Instruct")
    (wrong / "adapter_model.safetensors").write_bytes(b"other-adapter-bytes")
    tampered = tmp_path / "run_tampered" / "hf_adapter"
    _write_adapter_fixture(tampered)
    (tampered / "adapter_model.safetensors").write_bytes(b"tampered")

    picked = find_verified_preload_source(
        [str(wrong), str(tampered), str(good)], spec, "Qwen/Qwen2.5-3B-Instruct",
    )
    assert picked["source_dir"] == str(good)  # skips unverified, takes verified
    assert picked["report"]["verified"] is True

    none = find_verified_preload_source([str(wrong), str(tampered)], spec, "Qwen/Qwen2.5-3B-Instruct")
    assert none["source_dir"] is None and len(none["rejections"]) == 2

    # The measured R0 reference spec itself validates (pins + digests).
    assert r0_adapter_spec()["subfolder"].endswith("final_adapter")
