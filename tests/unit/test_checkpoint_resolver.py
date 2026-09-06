import json
import pytest
from pathlib import Path
from src.task2.checkpoint_resolver import resolve_component_checkpoint


def create_mock_generator_checkpoint(dir_path: Path, base_model: str = "Qwen/Qwen2.5-3B-Instruct", api_v: int = 16):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base_model}))
    (dir_path / "adapter_model.safetensors").write_text("weights")
    manifest = {
        "runtime_api_version": api_v,
        "base_model": base_model,
        "component": "generator",
        "is_final": True,
    }
    (dir_path / "generator_manifest.json").write_text(json.dumps(manifest))


def create_mock_reranker_checkpoint(dir_path: Path, base_model: str = "BAAI/bge-reranker-v2-m3", api_v: int = 16):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config.json").write_text(json.dumps({"_name_or_path": base_model}))
    (dir_path / "model.safetensors").write_text("weights")
    manifest = {
        "runtime_api_version": api_v,
        "base_model": base_model,
        "component": "reranker",
        "is_final": True,
    }
    (dir_path / "reranker_manifest.json").write_text(json.dumps(manifest))


def test_resolve_exact_preferred_path(tmp_path: Path):
    ckpt = tmp_path / "checkpoints" / "generator" / "hf_adapter"
    create_mock_generator_checkpoint(ckpt)
    resolved = resolve_component_checkpoint(
        component="generator",
        expected_base_model="Qwen/Qwen2.5-3B-Instruct",
        preferred_path=str(ckpt),
        search_roots=[tmp_path],
    )
    assert resolved == str(ckpt.resolve())


def test_resolve_reranker_success(tmp_path: Path):
    ckpt = tmp_path / "checkpoints" / "reranker" / "best"
    create_mock_reranker_checkpoint(ckpt)
    resolved = resolve_component_checkpoint(
        component="reranker",
        expected_base_model="BAAI/bge-reranker-v2-m3",
        preferred_path=str(ckpt),
        search_roots=[tmp_path],
    )
    assert resolved == str(ckpt.resolve())


def test_resolve_missing_checkpoint_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No valid generator checkpoint found"):
        resolve_component_checkpoint(
            component="generator",
            expected_base_model="Qwen/Qwen2.5-3B-Instruct",
            preferred_path=str(tmp_path / "nonexistent"),
            search_roots=[tmp_path],
        )


def test_resolve_ambiguous_checkpoints_raises_runtime_error(tmp_path: Path):
    ckpt1 = tmp_path / "ds1" / "checkpoints" / "generator" / "hf_adapter"
    ckpt2 = tmp_path / "ds2" / "checkpoints" / "generator" / "hf_adapter"
    create_mock_generator_checkpoint(ckpt1)
    create_mock_generator_checkpoint(ckpt2)

    with pytest.raises(RuntimeError, match="Ambiguous generator checkpoint candidates found"):
        resolve_component_checkpoint(
            component="generator",
            expected_base_model="Qwen/Qwen2.5-3B-Instruct",
            preferred_path=None,
            search_roots=[tmp_path / "ds1", tmp_path / "ds2"],
        )


def test_resolve_rejects_stale_api_manifest(tmp_path: Path):
    ckpt = tmp_path / "checkpoints" / "generator" / "hf_adapter"
    create_mock_generator_checkpoint(ckpt, api_v=15)  # Stale API 15

    with pytest.raises(RuntimeError, match="runtime_api_version"):
        resolve_component_checkpoint(
            component="generator",
            expected_base_model="Qwen/Qwen2.5-3B-Instruct",
            preferred_path=str(ckpt),
            search_roots=[tmp_path],
            expected_runtime_api=16,
        )


def test_resolve_rejects_base_model_mismatch(tmp_path: Path):
    ckpt = tmp_path / "checkpoints" / "generator" / "hf_adapter"
    create_mock_generator_checkpoint(ckpt, base_model="wrong-base-model")

    with pytest.raises(RuntimeError, match="expected base model"):
        resolve_component_checkpoint(
            component="generator",
            expected_base_model="Qwen/Qwen2.5-3B-Instruct",
            preferred_path=str(ckpt),
            search_roots=[tmp_path],
        )
