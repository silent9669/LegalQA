import inspect
import types
import pytest
from src.task2.training.train_generator import (
    run_qlora_training,
    inspect_and_guard_trl_chunk_size,
    validate_ce_chunk_size,
    build_sft_config,
)


def test_t4_generator_default_ce_chunk_is_32():
    sig = inspect.signature(run_qlora_training)
    assert "ce_chunk_size" in sig.parameters
    assert sig.parameters["ce_chunk_size"].default == 32


def test_v15_reduces_trl_256_chunk_to_32(monkeypatch):
    import sys
    # Create a fake trl module with version 1.12.0
    fake_trl = types.ModuleType("trl")
    fake_trl.__version__ = "1.12.0"
    fake_trainer = types.ModuleType("trl.trainer")
    fake_sft_trainer = types.ModuleType("trl.trainer.sft_trainer")
    fake_sft_trainer._CHUNKED_LM_HEAD_CHUNK_SIZE = 256

    fake_trainer.sft_trainer = fake_sft_trainer
    fake_trl.trainer = fake_trainer

    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setitem(sys.modules, "trl.trainer", fake_trainer)
    monkeypatch.setitem(sys.modules, "trl.trainer.sft_trainer", fake_sft_trainer)

    info = inspect_and_guard_trl_chunk_size(target_chunk_size=32)

    assert info["original_chunk_size"] == 256
    assert info["modified_chunk_size"] == 32
    assert info["action"] == "capped"
    assert fake_sft_trainer._CHUNKED_LM_HEAD_CHUNK_SIZE == 32


def test_chunk_size_must_be_positive_power_of_two():
    with pytest.raises(ValueError):
        validate_ce_chunk_size(0)
    with pytest.raises(ValueError):
        validate_ce_chunk_size(-1)
    with pytest.raises(ValueError):
        validate_ce_chunk_size(48)
    with pytest.raises(ValueError):
        validate_ce_chunk_size(512)
    assert validate_ce_chunk_size(1) == 1
    assert validate_ce_chunk_size(16) == 16
    assert validate_ce_chunk_size(32) == 32
    assert validate_ce_chunk_size(64) == 64
    assert validate_ce_chunk_size(128) == 128
    assert validate_ce_chunk_size(256) == 256


def test_quality_critical_training_semantics_preserved():
    sig = inspect.signature(run_qlora_training)
    assert sig.parameters["model_name_or_path"].default == "Qwen/Qwen2.5-3B-Instruct"
    assert sig.parameters["base_model_id"].default == "Qwen/Qwen2.5-3B-Instruct"
    assert sig.parameters["max_seq_len"].default == 2048
    assert sig.parameters["batch_size"].default == 1
    assert sig.parameters["grad_accum"].default == 8
    assert sig.parameters["lr"].default == 1e-4
    assert sig.parameters["ce_chunk_size"].default == 32


def test_sft_config_preserves_activation_offloading_and_chunked_loss(monkeypatch):
    import sys
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

    fake_trl = types.ModuleType("trl")
    fake_trl.SFTConfig = MockSFTConfig
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    cfg = build_sft_config(max_seq_len=2048)
    assert getattr(cfg, "completion_only_loss", None) is True
    assert getattr(cfg, "loss_type", None) == "chunked_nll"
    assert getattr(cfg, "activation_offloading", None) is True


def test_effective_chunk_size_mismatch_fails_loud(monkeypatch):
    import sys
    for mod_name in ["torch", "datasets", "peft", "transformers", "trl", "bitsandbytes"]:
        m = types.ModuleType(mod_name)
        monkeypatch.setitem(sys.modules, mod_name, m)

    fake_torch = sys.modules["torch"]
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda dev: None)

    fake_peft = sys.modules["peft"]
    fake_peft.LoraConfig = lambda **kwargs: None
    fake_peft.PeftModel = types.SimpleNamespace()

    fake_tf = sys.modules["transformers"]
    fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **kw: types.SimpleNamespace(pad_token=None, eos_token="<eos>", padding_side="right"))
    fake_tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a, **kw: None)
    fake_tf.BitsAndBytesConfig = lambda **kwargs: None

    fake_datasets = sys.modules["datasets"]
    fake_datasets.Dataset = types.SimpleNamespace(from_list=lambda l: [1])

    fake_trl = sys.modules["trl"]
    fake_trl.__version__ = "1.12.0"
    fake_trl.SFTConfig = lambda **kw: None
    fake_trl.SFTTrainer = lambda **kw: None

    def mock_guard(target_chunk_size):
        return {
            "trl_version": "1.12.0",
            "chunk_size_attr_present": True,
            "original_chunk_size": 256,
            "modified_chunk_size": 256,  # mismatch!
            "action": "failed_to_cap",
        }

    monkeypatch.setattr(
        "src.task2.training.train_generator.inspect_and_guard_trl_chunk_size",
        mock_guard,
    )

    with pytest.raises(RuntimeError, match="FINAL_PIPELINE_ERROR.*TRL CE chunk size guard failed"):
        run_qlora_training(ce_chunk_size=32, fail_on_error=True)

