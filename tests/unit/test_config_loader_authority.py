"""Task 1: authoritative config + frozen candidate regressions.

Covers: protected algorithm fields, effective-batch invariant, exact frozen
model revisions, contradictory legacy config rejection, old/new candidate
schema round-trip, and per-runtime bundle binding.
"""

from __future__ import annotations

import pytest
import yaml

from src.task2.config.loader import load_resolved_config
from src.task2.provenance.candidate import (
    CandidateManifest,
    DatasetRef,
    ModelRevisionRef,
    ModelsRef,
    RuntimeProfilesRef,
    create_candidate_manifest,
    validate_model_revisions,
)

GEN = "Qwen/Qwen2.5-3B-Instruct"
RER = "BAAI/bge-reranker-v2-m3"
DEN = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
REV_G = "d8a1c8901eb4284d720235adcf8849767f40d7e4"
REV_R = "278e2e28328135817d69932cb4ad12d7c58e5d32"
REV_D = "c356b6aa96e00cb1b6a12b48a1c0d4530058b884"


def test_final_rejects_floating_model_revision():
    models = {k: {"id": k, "revision": "a" * 40} for k in ("generator", "reranker", "dense")}
    models["generator"]["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        validate_model_revisions(models)


def test_rejects_missing_and_short_revisions():
    base = {
        "generator": {"id": GEN, "revision": REV_G},
        "reranker": {"id": RER, "revision": REV_R},
        "dense": {"id": DEN, "revision": REV_D},
    }
    validate_model_revisions(base)
    bad = {k: dict(v) for k, v in base.items()}
    del bad["dense"]["revision"]
    with pytest.raises(ValueError, match="immutable"):
        validate_model_revisions(bad)
    bad2 = {k: dict(v) for k, v in base.items()}
    bad2["reranker"]["revision"] = "278e2e2"
    with pytest.raises(ValueError, match="immutable"):
        validate_model_revisions(bad2)
    bad3 = {k: dict(v) for k, v in base.items()}
    bad3["generator"]["revision"] = "v1.0"
    with pytest.raises(ValueError, match="immutable"):
        validate_model_revisions(bad3)


def test_create_candidate_rejects_floating_revision():
    with pytest.raises(ValueError, match="immutable"):
        create_candidate_manifest(
            git_commit_sha="a" * 40,
            dataset_slug="s",
            dataset_version=1,
            dataset_manifest_sha256="b" * 64,
            algorithm_sha256="c" * 64,
            kaggle_t4x2_sha256="d" * 64,
            colab_t4_sha256="e" * 64,
            colab_a100_sha256="f" * 64,
            config_bundle_sha256="0" * 64,
            generator_id=GEN,
            generator_revision="main",
            reranker_id=RER,
            reranker_revision=REV_R,
            dense_id=DEN,
            dense_revision=REV_D,
            dependency_lock_sha256="1" * 64,
        )


def test_protected_field_mutation_rejected_one_at_a_time(tmp_path):
    algo = {
        "schema_version": 1,
        "seed": 42,
        "models": {
            "generator": {"id": GEN, "revision_policy": "exact_commit"},
            "reranker": {"id": RER, "revision_policy": "exact_commit"},
            "dense": {"id": DEN, "revision_policy": "exact_commit"},
        },
        "generator": {
            "max_seq_len": 2048, "quantization": "4bit_nf4", "double_quant": True,
            "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0,
            "target_modules": ["q_proj"], "learning_rate": 1.0e-4,
            "lr_scheduler_type": "cosine", "warmup_ratio": 0.05,
            "effective_batch_size": 8, "num_train_epochs": 3,
            "completion_only_loss": True, "use_liger_fused_ce": True,
            "gradient_checkpointing": True,
        },
        "final_training": {"training_scope": "all_allowed_train", "val_fold": None},
        "evaluation": {"primary_metric": "whitespace_meteor", "secondary_metric": "rouge_l"},
    }
    algo_p = tmp_path / "algo.yaml"
    algo_p.write_text(yaml.dump(algo), encoding="utf-8")
    for protected in ("lora_r", "max_seq_len", "seed", "effective_batch_size"):
        rt = {
            "profile_name": "x", "required_gpu_count": 1, "required_gpu_name_contains": "T4",
            "devices": {"generator": "cuda:0", "retrieval": "cuda:0"},
            "generator_runtime": {
                "compute_dtype": "float16", "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8, "activation_offloading": True,
            },
            protected: 1,
        }
        rt_p = tmp_path / f"rt_{protected}.yaml"
        rt_p.write_text(yaml.dump(rt), encoding="utf-8")
        with pytest.raises(ValueError, match="Protected field or unknown runtime key"):
            load_resolved_config(algo_p, rt_p)


def test_batch_factor_mutation_rejected(tmp_path):
    algo = {
        "schema_version": 1, "seed": 42,
        "models": {
            "generator": {"id": GEN, "revision_policy": "exact_commit"},
            "reranker": {"id": RER, "revision_policy": "exact_commit"},
            "dense": {"id": DEN, "revision_policy": "exact_commit"},
        },
        "generator": {
            "max_seq_len": 2048, "quantization": "4bit_nf4", "double_quant": True,
            "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0,
            "target_modules": ["q_proj"], "learning_rate": 1.0e-4,
            "lr_scheduler_type": "cosine", "warmup_ratio": 0.05,
            "effective_batch_size": 8, "num_train_epochs": 3,
            "completion_only_loss": True, "use_liger_fused_ce": True,
            "gradient_checkpointing": True,
        },
        "final_training": {"training_scope": "all_allowed_train", "val_fold": None},
        "evaluation": {"primary_metric": "whitespace_meteor", "secondary_metric": "rouge_l"},
    }
    algo_p = tmp_path / "algo.yaml"
    algo_p.write_text(yaml.dump(algo), encoding="utf-8")
    rt = {
        "profile_name": "x", "required_gpu_count": 1, "required_gpu_name_contains": "T4",
        "devices": {"generator": "cuda:0", "retrieval": "cuda:0"},
        "generator_runtime": {
            "compute_dtype": "float16", "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2, "activation_offloading": True,
        },
    }
    rt_p = tmp_path / "rt.yaml"
    rt_p.write_text(yaml.dump(rt), encoding="utf-8")
    with pytest.raises(ValueError, match="Effective batch size mismatch"):
        load_resolved_config(algo_p, rt_p)


def _make_manifest(schema_version=2, **over):
    kw = dict(
        schema_version=schema_version, candidate_id="", task="task2",
        git_repository="https://github.com/silent9669/LegalQA.git",
        git_commit_sha="a" * 40,
        dataset=DatasetRef(slug="s", version=1, manifest_sha256="b" * 64),
        algorithm_sha256="c" * 64,
        runtime_profile_sha256=RuntimeProfilesRef(
            kaggle_t4x2="d" * 64, colab_t4="e" * 64, colab_a100="f" * 64,
            modal_a100="9" * 64,
        ),
        config_bundle_sha256="0" * 64,
        models=ModelsRef(
            generator=ModelRevisionRef(id=GEN, revision=REV_G),
            reranker=ModelRevisionRef(id=RER, revision=REV_R),
            dense=ModelRevisionRef(id=DEN, revision=REV_D),
        ),
        dependency_lock_sha256="1" * 64, seed=42,
        created_at_utc="2026-09-18T00:00:00Z",
        bundle_sha256_by_profile={"kaggle_t4x2": "0" * 64},
    )
    kw.update(over)
    return CandidateManifest(**kw)


def test_old_schema_roundtrip_preserves_id(tmp_path):
    m = _make_manifest(
        schema_version=1,
        runtime_profile_sha256=RuntimeProfilesRef(
            kaggle_t4x2="d" * 64, colab_t4="e" * 64, colab_a100="f" * 64,
        ),
        bundle_sha256_by_profile={},
    ).with_computed_id()
    p = tmp_path / "cand.json"
    m.save_json(p)
    loaded = CandidateManifest.load_json(p)
    assert loaded.candidate_id == m.candidate_id
    assert loaded.runtime_profile_sha256.modal_a100 == ""
    assert loaded.bundle_sha256_by_profile == {}


def test_new_modal_schema_roundtrip(tmp_path):
    m = _make_manifest(schema_version=2).with_computed_id()
    p = tmp_path / "cand.json"
    m.save_json(p)
    loaded = CandidateManifest.load_json(p)
    assert loaded.candidate_id == m.candidate_id
    assert loaded.runtime_profile_sha256.modal_a100 == "9" * 64
    # v1 payload excludes modal fields, so v1/v2 ids must differ for same base
    m1 = _make_manifest(
        schema_version=1,
        runtime_profile_sha256=RuntimeProfilesRef(
            kaggle_t4x2="d" * 64, colab_t4="e" * 64, colab_a100="f" * 64,
        ),
        bundle_sha256_by_profile={},
    ).with_computed_id()
    assert m1.candidate_id != m.candidate_id


def test_validate_against_config_binds_selected_runtime():
    cfg = load_resolved_config("configs/task2/algorithm.yaml", "configs/task2/runtime/kaggle_t4x2.yaml")
    m = _make_manifest(
        schema_version=2,
        algorithm_sha256=cfg.algorithm_sha256,
        runtime_profile_sha256=RuntimeProfilesRef(
            kaggle_t4x2=cfg.runtime_sha256, colab_t4="e" * 64,
            colab_a100="f" * 64, modal_a100="9" * 64,
        ),
        config_bundle_sha256=cfg.bundle_sha256,
        bundle_sha256_by_profile={"kaggle_t4x2": cfg.bundle_sha256},
        seed=cfg.algorithm.seed,
    ).with_computed_id()
    m.validate_against_config(cfg)
    cfg_colab = load_resolved_config("configs/task2/algorithm.yaml", "configs/task2/runtime/colab_a100.yaml")
    # Same candidate must reject a different platform runtime (undeclared hash differs)
    with pytest.raises(ValueError):
        m.validate_against_config(cfg_colab)


def test_retrieval_recipe_parses_with_weight_invariants(tmp_path):
    cfg = load_resolved_config("configs/task2/algorithm.yaml", "configs/task2/runtime/kaggle_t4x2.yaml")
    retrieval = cfg.algorithm.retrieval
    assert retrieval.rrf_k == 60 and retrieval.candidate_pool == 50
    assert abs(retrieval.w_bm25_plain + retrieval.w_dense_plain - 1.0) < 1e-9
    assert abs(retrieval.w_bm25_lex + retrieval.w_dense_lex + retrieval.w_lexref_lex - 1.0) < 1e-9


def test_retrieval_weights_must_sum_to_one(tmp_path):
    import copy

    base = yaml.safe_load(open("configs/task2/algorithm.yaml"))
    bad = copy.deepcopy(base)
    bad["retrieval"]["w_bm25_plain"] = 0.9
    algo_p = tmp_path / "algo.yaml"
    algo_p.write_text(yaml.dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to 1"):
        load_resolved_config(algo_p, "configs/task2/runtime/kaggle_t4x2.yaml")


def test_runtime_cannot_override_retrieval_section(tmp_path):
    import copy

    base = yaml.safe_load(open("configs/task2/runtime/kaggle_t4x2.yaml"))
    bad = copy.deepcopy(base)
    bad["retrieval"] = {"w_bm25_plain": 1.0}
    rt_p = tmp_path / "rt.yaml"
    rt_p.write_text(yaml.dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="Protected field or unknown runtime key"):
        load_resolved_config("configs/task2/algorithm.yaml", rt_p)


def test_inference_batch_defaults_and_modal_scale(tmp_path):
    kaggle = load_resolved_config("configs/task2/algorithm.yaml", "configs/task2/runtime/kaggle_t4x2.yaml")
    assert kaggle.runtime.inference.generation_batch_size == 4
    assert kaggle.runtime.inference.reranker_batch_size == 32
    modal = load_resolved_config("configs/task2/algorithm.yaml", "configs/task2/runtime/modal_a100.yaml")
    assert modal.runtime.inference.generation_batch_size == 16
    assert modal.runtime.inference.reranker_batch_size == 128
    bad_rt = {
        "profile_name": "x", "required_gpu_count": 1, "required_gpu_name_contains": "T4",
        "devices": {"generator": "cuda:0", "retrieval": "cuda:0"},
        "generator_runtime": {"compute_dtype": "float16", "per_device_train_batch_size": 1,
                              "gradient_accumulation_steps": 8, "activation_offloading": True},
        "inference": {"generation_batch_size": 0, "reranker_batch_size": 32, "retrieval_batch_size": 32},
    }
    rt_p = tmp_path / "rt.yaml"
    rt_p.write_text(yaml.dump(bad_rt), encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        load_resolved_config("configs/task2/algorithm.yaml", rt_p)
