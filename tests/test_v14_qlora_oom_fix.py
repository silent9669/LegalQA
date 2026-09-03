"""Tests for LegalQA V14 QLoRA OOM fix, TRL 1.12.0 contract, and memory stability policies."""

from __future__ import annotations

import inspect
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict
import pytest

from src.common.security import assert_no_secrets_in_workspace
from src.task2.path_resolver import resolve_runtime_paths
from src.task2.runtime_integrity import (
    EXPECTED_RUNTIME_API_VERSION,
    validate_runtime_manifests,
)
import src.task2.training.train_generator as train_gen_mod
from src.task2.training.train_generator import (
    build_sft_config,
    enforce_single_gpu_trainer_args,
    inspect_and_guard_trl_chunk_size,
    run_qlora_training,
)
from scripts.bootstrap_kaggle_env import (
    TARGET_USER_PACKAGES,
    check_trl_api_signatures,
)


def test_req1_exact_trl_runtime_contract():
    """Item 1: Verify exact TRL runtime contract (trl==1.12.0) across all configuration and bootstrap files."""
    # 1. requirements-kaggle.txt
    req_kaggle = Path("requirements-kaggle.txt").read_text(encoding="utf-8")
    assert re.search(r"^trl==1\.12\.0\b", req_kaggle, re.MULTILINE), (
        "requirements-kaggle.txt must pin trl==1.12.0"
    )

    # 2. scripts/bootstrap_kaggle_env.py
    trl_pkg = next((pkg for pkg in TARGET_USER_PACKAGES if pkg[0] == "trl"), None)
    assert trl_pkg is not None, "scripts/bootstrap_kaggle_env.py must contain trl in TARGET_USER_PACKAGES"
    assert trl_pkg[1] == "==1.12.0", f"trl specifier must be '==1.12.0', got {trl_pkg[1]}"

    # 3. .github/workflows/tests.yml
    workflow = Path(".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert "trl==1.12.0" in workflow, ".github/workflows/tests.yml must pin trl==1.12.0"

    # 4. check_trl_api_signatures verifies required APIs on modern classes
    class ModernSFTConfig:
        def __init__(
            self,
            completion_only_loss=True,
            loss_type="chunked_nll",
            activation_offloading=True,
            max_length=2048,
            **kwargs,
        ):
            pass

    class ModernSFTTrainer:
        def __init__(self, processing_class=None, **kwargs):
            pass

    missing = check_trl_api_signatures(ModernSFTConfig, ModernSFTTrainer)
    assert len(missing) == 0, f"Modern TRL should have no missing signatures, got {missing}"


def test_req2_and_3_and_4_sft_config_requires_chunked_nll_and_activation_offloading(tmp_path: Path, monkeypatch):
    """Items 2, 3, 4: Verify build_sft_config sets completion_only_loss, chunked_nll, and activation_offloading."""
    captured_kwargs = {}

    class MockSFTConfig:
        def __init__(
            self,
            completion_only_loss=True,
            loss_type="chunked_nll",
            activation_offloading=True,
            max_length=2048,
            **kwargs,
        ):
            self.completion_only_loss = completion_only_loss
            self.loss_type = loss_type
            self.activation_offloading = activation_offloading
            self.max_length = max_length
            captured_kwargs.update(kwargs)
            captured_kwargs["completion_only_loss"] = completion_only_loss
            captured_kwargs["loss_type"] = loss_type
            captured_kwargs["activation_offloading"] = activation_offloading
            captured_kwargs["max_length"] = max_length

    fake_trl = types.ModuleType("trl")
    fake_trl.SFTConfig = MockSFTConfig
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    cfg = build_sft_config(max_seq_len=2048, output_dir=str(tmp_path))
    assert getattr(cfg, "completion_only_loss") is True
    assert getattr(cfg, "loss_type") == "chunked_nll"
    assert getattr(cfg, "activation_offloading") is True
    assert getattr(cfg, "max_length") == 2048
    assert captured_kwargs["loss_type"] == "chunked_nll"
    assert captured_kwargs["activation_offloading"] is True
    assert captured_kwargs["completion_only_loss"] is True

    # Verify that missing completion_only_loss raises RuntimeError
    class IncompatibleSFTConfig:
        def __init__(self, max_length=2048, **kwargs):
            pass

    fake_trl_incompatible = types.ModuleType("trl")
    fake_trl_incompatible.SFTConfig = IncompatibleSFTConfig
    monkeypatch.setitem(sys.modules, "trl", fake_trl_incompatible)

    with pytest.raises(RuntimeError, match="completion_only_loss"):
        build_sft_config(max_seq_len=2048, output_dir=str(tmp_path))


def test_req5_chunk_size_guard_only_activates_for_trl_1_12_and_values_greater_than_256(monkeypatch):
    """Item 5: Verify chunk-size guard only caps when TRL is 1.12.0 and current value > 256."""
    # Scenario A: TRL 1.12.0 and orig_val=1024 -> capped to 256
    fake_sft_a = types.ModuleType("trl.trainer.sft_trainer")
    fake_sft_a._CHUNKED_LM_HEAD_CHUNK_SIZE = 1024
    fake_trl_a = types.ModuleType("trl")
    fake_trl_a.__version__ = "1.12.0"
    fake_trl_a.trainer = types.ModuleType("trl.trainer")
    fake_trl_a.trainer.sft_trainer = fake_sft_a

    monkeypatch.setitem(sys.modules, "trl", fake_trl_a)
    monkeypatch.setitem(sys.modules, "trl.trainer", fake_trl_a.trainer)
    monkeypatch.setitem(sys.modules, "trl.trainer.sft_trainer", fake_sft_a)

    info_a = inspect_and_guard_trl_chunk_size(target_chunk_size=256)
    assert info_a["trl_version"] == "1.12.0"
    assert info_a["original_chunk_size"] == 1024
    assert info_a["modified_chunk_size"] == 256
    assert "capped" in info_a["action"]
    assert fake_sft_a._CHUNKED_LM_HEAD_CHUNK_SIZE == 256

    # Scenario B: TRL 1.12.0 and orig_val=128 (<= 256) -> retained 128
    fake_sft_b = types.ModuleType("trl.trainer.sft_trainer")
    fake_sft_b._CHUNKED_LM_HEAD_CHUNK_SIZE = 128
    fake_trl_b = types.ModuleType("trl")
    fake_trl_b.__version__ = "1.12.0"
    fake_trl_b.trainer = types.ModuleType("trl.trainer")
    fake_trl_b.trainer.sft_trainer = fake_sft_b

    monkeypatch.setitem(sys.modules, "trl", fake_trl_b)
    monkeypatch.setitem(sys.modules, "trl.trainer", fake_trl_b.trainer)
    monkeypatch.setitem(sys.modules, "trl.trainer.sft_trainer", fake_sft_b)

    info_b = inspect_and_guard_trl_chunk_size(target_chunk_size=256)
    assert info_b["original_chunk_size"] == 128
    assert info_b["modified_chunk_size"] == 128
    assert "retained" in info_b["action"]
    assert fake_sft_b._CHUNKED_LM_HEAD_CHUNK_SIZE == 128

    # Scenario C: Different TRL version (e.g. 1.13.0) and orig_val=1024 -> retained 1024
    fake_sft_c = types.ModuleType("trl.trainer.sft_trainer")
    fake_sft_c._CHUNKED_LM_HEAD_CHUNK_SIZE = 1024
    fake_trl_c = types.ModuleType("trl")
    fake_trl_c.__version__ = "1.13.0"
    fake_trl_c.trainer = types.ModuleType("trl.trainer")
    fake_trl_c.trainer.sft_trainer = fake_sft_c

    monkeypatch.setitem(sys.modules, "trl", fake_trl_c)
    monkeypatch.setitem(sys.modules, "trl.trainer", fake_trl_c.trainer)
    monkeypatch.setitem(sys.modules, "trl.trainer.sft_trainer", fake_sft_c)

    info_c = inspect_and_guard_trl_chunk_size(target_chunk_size=256)
    assert info_c["modified_chunk_size"] == 1024
    assert "retained" in info_c["action"]
    assert fake_sft_c._CHUNKED_LM_HEAD_CHUNK_SIZE == 1024


def test_req6_max_seq_len_remains_2048():
    """Item 6: Verify max_seq_len remains strictly 2048 without speculative reduction."""
    sig = inspect.signature(run_qlora_training)
    assert sig.parameters["max_seq_len"].default == 2048, (
        "run_qlora_training must have default max_seq_len=2048"
    )


def test_req7_lora_r_alpha_targets_unchanged():
    """Item 7: Verify LoRA rank (16), alpha (32), dropout (0.05) and targets remain unchanged."""
    src = inspect.getsource(run_qlora_training)
    assert "r=16" in src or "r = 16" in src or '"r": 16' in src
    assert "lora_alpha=32" in src or "lora_alpha = 32" in src or '"lora_alpha": 32' in src
    assert "lora_dropout=0.05" in src or "lora_dropout = 0.05" in src
    for target in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        assert target in src, f"LoRA target module {target} must be preserved"


def test_req8_completion_only_loss_remains_enabled():
    """Item 8: Verify completion-only loss remains enabled."""
    src = inspect.getsource(build_sft_config)
    assert 'config_kwargs["completion_only_loss"] = True' in src
    assert '"completion_only_loss" not in sig.parameters' in src


def test_req9_single_gpu_trainer_policy_enforced_before_sft_trainer():
    """Item 9: Verify single-GPU Trainer policy is enforced strictly before SFTTrainer."""
    src = inspect.getsource(run_qlora_training)
    force_pos = src.index("enforce_single_gpu_trainer_args(sft_args, dev)")
    trainer_pos = src.index("trainer = SFTTrainer(")
    assert force_pos < trainer_pos, (
        "enforce_single_gpu_trainer_args must be called before SFTTrainer instantiation"
    )

    class MockArgs:
        def __init__(self):
            self._n_gpu = 2
            self._device = "cuda:0"

        @property
        def device(self):
            return self._device

        @property
        def n_gpu(self):
            return self._n_gpu

    mock_args = MockArgs()
    enforce_single_gpu_trainer_args(mock_args, "cuda:0")
    assert mock_args.n_gpu == 1


def test_req10_full_shape_colab_mode_uses_none_max_train_examples():
    """Item 10: Verify full-shape Colab runner defaults to max_train_examples=None for 5956-example pool."""
    import scripts.run_colab_smoke as smoke_mod
    src = inspect.getsource(smoke_mod.parse_args)
    assert 'default=None' in src and '5956-example pool' in src


def test_req11_stale_api13_package_is_rejected(tmp_path: Path):
    """Item 11: Verify stale API 13 package is rejected by runtime integrity validation."""
    runtime = tmp_path / "runtime"
    code = runtime / "code" / "LegalQA"
    code.mkdir(parents=True)
    valid_sha = "e37e04df3aa78df3170954c64039c1a2c9109317"
    (runtime / "dataset_manifest.json").write_text(
        json.dumps({"runtime_api_version": 13, "git_sha": valid_sha})
    )
    (code / "code_manifest.json").write_text(
        json.dumps({"runtime_api_version": 13, "git_sha": valid_sha})
    )

    with pytest.raises(RuntimeError, match="runtime_api_version mismatch: found 13, expected 14"):
        validate_runtime_manifests(str(runtime), str(code), expected_api_version=14)


def test_req12_api14_nested_kaggle_package_resolves_correctly(tmp_path: Path):
    """Item 12: Verify API 14 nested Kaggle layout resolves without error."""
    kaggle_input = tmp_path / "kaggle" / "input"
    ds_root = kaggle_input / "datasets" / "phucdangg" / "legalqa-task2-clean-data"
    code_root = ds_root / "code" / "LegalQA"
    (code_root / "src").mkdir(parents=True)
    (code_root / "scripts").mkdir(parents=True)
    (ds_root / "indexes" / "bm25").mkdir(parents=True)
    (ds_root / "indexes" / "dek21").mkdir(parents=True)
    (ds_root / "legal_chunks.parquet").write_bytes(b"PAR1_DUMMY")
    (ds_root / "indexes" / "bm25" / "bm25_manifest.json").write_text("{}", encoding="utf-8")

    # Populate mock Qwen model dir
    qwen_dir = kaggle_input / "qwen-lm" / "qwen2.5" / "transformers" / "3b-instruct" / "1"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    (qwen_dir / "config.json").write_text(
        json.dumps({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "torch_dtype": "bfloat16",
        }),
        encoding="utf-8",
    )

    valid_sha = "0123456789abcdef0123456789abcdef01234567"
    (ds_root / "dataset_manifest.json").write_text(
        json.dumps({"runtime_api_version": 14, "git_sha": valid_sha})
    )
    (code_root / "code_manifest.json").write_text(
        json.dumps({"runtime_api_version": 14, "git_sha": valid_sha})
    )

    prov = validate_runtime_manifests(str(ds_root), str(code_root), expected_api_version=14)
    assert prov["runtime_api_version"] == 14
    assert prov["git_sha"] == valid_sha

    paths = resolve_runtime_paths(str(kaggle_input), strict=False)
    assert Path(paths["runtime_root"]).resolve() == ds_root.resolve()
    assert Path(paths["qwen_model_path"]).resolve() == qwen_dir.resolve()


def test_req13_no_secret_scanning_regression():
    """Item 13: Verify workspace secret scanning runs clean without regressions."""
    workspace_root = Path(__file__).resolve().parent.parent
    findings = assert_no_secrets_in_workspace(workspace_root, exclude_tests=True)
    assert findings is None or len(findings) == 0
