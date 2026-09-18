"""Task 4: context scope + generator recipe wiring regressions."""

from __future__ import annotations

import pytest

from src.task2.config.loader import load_resolved_config
from src.task2.generation.config import (
    APPROVED_SEQUENCE_POLICIES,
    GeneratorTrainConfig,
    validate_generator_config_for_profile,
)
from src.task2.training.context_builder import (
    build_context_record,
    build_gold_assisted_records,
    build_training_records,
    recipe_epochs,
    recipe_to_train_config,
)


def test_scope_keeps_unlabelled_train_but_excludes_heldout():
    qas = [
        {"qa_id": "q1", "qa_group_id": "train", "question_raw": "Q", "answer_raw": "A"},
        {"qa_id": "q2", "qa_group_id": "dev", "question_raw": "Other", "answer_raw": "B"},
    ]
    # There are no gold retrieval labels; q1 is still eligible for SFT.
    out, _ = build_training_records(qas, {"q1": ["c1"]}, [{"chunk_id": "c1", "text_raw": "E"}], {"train"})
    assert [r["qa_id"] for r in out] == ["q1"]


def test_unlabelled_qa_with_empty_retrieval_stays_eligible():
    qas = [{"qa_id": "q9", "qa_group_id": "train", "question_raw": "Q?", "answer_raw": "A."}]
    out, report = build_training_records(qas, {}, [], {"train"})
    assert [r["qa_id"] for r in out] == ["q9"]
    assert report["missing_evidence_records"] == 1
    assert out[0]["evidence_count"] == 0


def test_repeated_chunk_fragments_all_join_context():
    qas = [{"qa_id": "q1", "qa_group_id": "g1", "question_raw": "Q?", "answer_raw": "A."}]
    chunks = [
        {"chunk_id": "c", "text_raw": "first"},
        {"chunk_id": "c", "text_raw": "second"},
    ]
    out, _ = build_training_records(qas, {"q1": ["c"]}, chunks, {"g1"})
    assert out[0]["evidence_texts"] == ["first", "second"]


def test_full_answer_and_ids_survive_record():
    qa = {"qa_id": "q1", "qa_group_id": "g1", "question_raw": "Q with 800.000 dong?", "answer_raw": "A 800.000 dong, khong mien tru."}
    rec = build_context_record(qa, [{"chunk_id": "c1", "text_raw": "E1"}], policy="retrieved", max_seq_len=2048)
    assert rec["qa_id"] == "q1" and rec["qa_group_id"] == "g1"
    assert "800.000" in rec["answer_raw"] and "khong" in rec["answer_raw"]
    with pytest.raises(ValueError, match="sequence budget"):
        build_context_record(qa, [], policy="retrieved", max_seq_len=999)


def test_gold_assisted_forbids_heldout_groups():
    qas = [{"qa_id": "q2", "qa_group_id": "dev", "question_raw": "Q?", "answer_raw": "A."}]
    with pytest.raises(ValueError, match="held-out"):
        build_gold_assisted_records(qas, {"q2": ["c1"]}, [{"chunk_id": "c1", "text_raw": "E"}], {"train"})
    ok_qas = [{"qa_id": "q1", "qa_group_id": "train", "question_raw": "Q?", "answer_raw": "A."}]
    recs, rep = build_gold_assisted_records(ok_qas, {"q1": ["c1"]}, [{"chunk_id": "c1", "text_raw": "E"}], {"train"})
    assert recs[0]["policy"] == "gold_assisted" and rep["kept_records"] == 1


def test_recipe_propagates_candidate_choices():
    cfg = load_resolved_config("configs/task2/algorithm.yaml", "configs/task2/runtime/kaggle_t4x2.yaml")
    train_cfg = recipe_to_train_config(cfg, device="cuda:0")
    assert isinstance(train_cfg, GeneratorTrainConfig)
    assert train_cfg.lora_r == cfg.algorithm.generator.lora_r == 16
    assert train_cfg.lora_alpha == 32
    assert train_cfg.lora_dropout == cfg.algorithm.generator.lora_dropout == 0.0
    assert train_cfg.batch_size * train_cfg.grad_accum == cfg.algorithm.generator.effective_batch_size == 8
    assert train_cfg.model_id == cfg.algorithm.models.generator.id
    assert recipe_epochs(cfg) == cfg.algorithm.generator.num_train_epochs
    # Legacy and candidate-approved sequence policies both validate.
    assert 2048 in APPROVED_SEQUENCE_POLICIES and 3072 in APPROVED_SEQUENCE_POLICIES
    validate_generator_config_for_profile(train_cfg, profile="generator_probe_worstcase")
    cfg3072 = GeneratorTrainConfig(model_id=train_cfg.model_id, max_seq_len=3072)
    validate_generator_config_for_profile(cfg3072, profile="final_train_and_submit")
    with pytest.raises(ValueError, match="max_seq_len"):
        validate_generator_config_for_profile(
            GeneratorTrainConfig(model_id=train_cfg.model_id, max_seq_len=1024),
            profile="final_train_and_submit",
        )
