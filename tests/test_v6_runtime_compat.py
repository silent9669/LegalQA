"""Regression tests for LegalQA V6 runtime compatibility, strict environment, and promotion logic."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import yaml

from src.common.hashing import sha256_file
from src.task2.checkpoint_manifest import assert_final_checkpoint
from src.task2.path_resolver import find_qwen_model_dir, find_runtime_roots, resolve_runtime_paths
from src.task2.predict import LegalQAPipeline
from src.task2.production_config import (
    ProductionSelection,
    load_production_selection,
    policy_requires_generator,
    validate_production_selection_for_profile,
)
from src.task2.training.train_generator import (
    build_grounded_training_examples,
    build_sft_example_token_aware,
    run_seq_len_diagnostic,
)
from scripts.preflight_kaggle import run_preflight_checks
from scripts.promote_production_selection import promote_production_selection


def test_streaming_sha256_parity(tmp_path):
    """P0-5: Test that sha256_file matches standard in-memory hashlib hashing."""
    test_file = tmp_path / "large_mock_file.bin"
    # Create 1MB test file with pattern
    data = b"LegalQA-V6-Streaming-SHA-Parity-Test" * 30000
    test_file.write_bytes(data)

    import hashlib
    expected_sha = hashlib.sha256(data).hexdigest()
    computed_sha = sha256_file(test_file, chunk_size=4096)

    assert computed_sha == expected_sha


def test_no_obsolete_trl_collator_import():
    """P0-1: Ensure DataCollatorForCompletionOnlyLM is not imported in train_generator."""
    import src.task2.training.train_generator as tg
    assert not hasattr(tg, "DataCollatorForCompletionOnlyLM")


def test_sft_prompt_completion_structure():
    """P0-1: Verify that build_sft_example_token_aware and grounded examples return prompt and completion keys."""
    class MockTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            user_content = messages[1]["content"]
            return f"<|im_start|>system\nPrompt<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids)

    tok = MockTokenizer()
    q = "Quy định xử phạt?"
    ev = "Điều 1. Phạt tiền từ 1 đến 2 triệu đồng."
    ans = "Căn cứ Điều 1, phạt tiền từ 1 đến 2 triệu đồng."

    full_text, diag = build_sft_example_token_aware(
        question=q,
        evidence_text=ev,
        answer=ans,
        tokenizer=tok,
        max_seq_len=100,
    )

    assert full_text is not None
    assert "prompt" in diag
    assert "completion" in diag
    assert "<|im_start|>assistant\n" in diag["prompt"]
    assert ans in diag["completion"]
    assert "<|im_end|>" in diag["completion"]

    # Test grounded dataset builder produces prompt/completion
    df_qa = pd.DataFrame([
        {"qa_id": "q1", "fold_id": 1, "question_raw": q, "answer_raw": ans}
    ])
    examples = build_grounded_training_examples(df_qa=df_qa, tokenizer=tok)
    assert len(examples) == 1
    assert "prompt" in examples[0]
    assert "completion" in examples[0]
    assert "text" in examples[0]


def test_smoke_evidence_loading_bounded(tmp_path):
    """P1-1: Verify that grounded dataset builder samples QA first and only loads needed chunks."""
    qa_path = tmp_path / "qa.parquet"
    labels_path = tmp_path / "labels.parquet"
    chunks_path = tmp_path / "chunks.parquet"

    # Create 10 QA rows
    df_qa = pd.DataFrame([
        {"qa_id": f"q{i}", "fold_id": 1, "question_raw": f"Q{i}", "answer_raw": f"Ans{i}"}
        for i in range(10)
    ])
    df_qa.to_parquet(qa_path)

    # Labels map each QA to chunk_id = c{i}
    df_labels = pd.DataFrame([
        {"qa_id": f"q{i}", "positive_chunk_id": f"c{i}"}
        for i in range(10)
    ])
    df_labels.to_parquet(labels_path)

    # Chunks has 10 chunks
    df_chunks = pd.DataFrame([
        {"chunk_id": f"c{i}", "text_raw": f"Evidence text {i}"}
        for i in range(10)
    ])
    df_chunks.to_parquet(chunks_path)

    # Sample only 2 examples
    examples = build_grounded_training_examples(
        qa_path=str(qa_path),
        labels_path=str(labels_path),
        chunks_path=str(chunks_path),
        max_train_examples=2,
        seed=42,
    )
    assert len(examples) == 2


def test_preflight_cuda_hard_fail():
    """P0-3: Verify preflight hard-fails when require_cuda=True and CUDA is not available."""
    res = run_preflight_checks(
        pipeline_config_path="configs/pipeline.yaml",
        models_config_path="configs/models.yaml",
        production_config_path="configs/production_selection.yaml",
        require_cuda=True,
        check_dataset_files=False,
        check_indexes=False,
    )
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
    except ImportError:
        cuda_avail = False

    if not cuda_avail:
        assert not res["passed"]
        assert any("CUDA is required" in err for err in res["errors"])


def test_preflight_missing_indexes_fail(tmp_path):
    """P0-4: Verify preflight fails when BM25 or DEk21 directories/manifests are missing."""
    res_missing_dir = run_preflight_checks(
        pipeline_config_path="configs/pipeline.yaml",
        models_config_path="configs/models.yaml",
        production_config_path="configs/production_selection.yaml",
        require_cuda=False,
        check_dataset_files=False,
        check_indexes=True,
        bm25_dir=str(tmp_path / "nonexistent_bm25"),
        dek21_dir=str(tmp_path / "nonexistent_dek21"),
    )
    assert not res_missing_dir["passed"]
    assert any("Missing BM25 index directory" in err for err in res_missing_dir["errors"])
    assert any("Missing DEk21 dense directory" in err for err in res_missing_dir["errors"])

    # Directory exists but manifest missing
    empty_bm25 = tmp_path / "empty_bm25"
    empty_bm25.mkdir()
    res_missing_manifest = run_preflight_checks(
        pipeline_config_path="configs/pipeline.yaml",
        models_config_path="configs/models.yaml",
        production_config_path="configs/production_selection.yaml",
        require_cuda=False,
        check_dataset_files=False,
        check_indexes=True,
        bm25_dir=str(empty_bm25),
        dek21_dir=None,
    )
    assert not res_missing_manifest["passed"]
    assert any("Missing BM25 manifest" in err for err in res_missing_manifest["errors"])


def test_path_resolver_ambiguity_guard(tmp_path):
    """P0-6: Verify that ambiguous runtime roots or Qwen candidates raise RuntimeError."""
    # Create two competing dataset roots
    root1 = tmp_path / "dataset1"
    root1.mkdir()
    (root1 / "dataset_manifest.json").write_text("{}")
    (root1 / "legal_chunks.parquet").write_text("data")

    root2 = tmp_path / "dataset2"
    root2.mkdir()
    (root2 / "dataset_manifest.json").write_text("{}")
    (root2 / "legal_chunks.parquet").write_text("data")

    roots = find_runtime_roots(str(tmp_path))
    assert len(roots) == 2

    # resolve_runtime_paths must raise on multiple roots
    with pytest.raises(RuntimeError, match="Ambiguous runtime dataset roots"):
        resolve_runtime_paths(str(tmp_path))


def test_qwen_path_resolver_ambiguity_guard(tmp_path):
    """P0-6: Verify find_qwen_model_dir raises on multiple ambiguous Qwen 3B candidates."""
    m1 = tmp_path / "Qwen2.5-3B-v1"
    m1.mkdir()
    (m1 / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}))

    m2 = tmp_path / "Qwen2.5-3B-v2"
    m2.mkdir()
    (m2 / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}))

    with pytest.raises(RuntimeError, match="Ambiguous Qwen candidate model directories"):
        find_qwen_model_dir(str(tmp_path))


def test_pipeline_retrieve_and_rerank_trace_api():
    """P0-7: Verify LegalQAPipeline.retrieve_and_rerank and predict_single(return_trace=True)."""
    pipeline = LegalQAPipeline.build_mock()
    trace = pipeline.retrieve_and_rerank("Hành vi không tiêm phòng?")

    assert "bm25_results" in trace
    assert "dense_results" in trace
    assert "fused_results" in trace
    assert "reranked_results" in trace
    assert "primary_evidence" in trace
    assert "retrieval_meta" in trace

    # Test predict_single with return_trace=True
    selected, cands, trace_dict = pipeline.predict_single(
        qa_id="test_q1",
        question="Hành vi không tiêm phòng phạt bao nhiêu?",
        return_candidates=True,
        return_trace=True,
    )
    assert isinstance(selected, str)
    assert isinstance(cands, dict)
    assert isinstance(trace_dict, dict)
    assert "reranked_results" in trace_dict


def test_manifest_val_fold_excluded_key(tmp_path):
    """P0-13: Verify assert_final_checkpoint checks val_fold_excluded."""
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    # Manifest with val_fold_excluded set -> must fail final assertion
    bad_manifest = {
        "base_model": "BAAI/bge-reranker-v2-m3",
        "is_final_checkpoint": True,
        "training_scope": "all_allowed_task2_data",
        "val_fold_excluded": 0,
        "smoke_only": False,
    }
    (ckpt_dir / "reranker_manifest.json").write_text(json.dumps(bad_manifest))

    with pytest.raises(ValueError, match="was trained with held-out val_fold"):
        assert_final_checkpoint(str(ckpt_dir), expected_base_model="BAAI/bge-reranker-v2-m3", component_name="reranker")

    # Manifest with val_fold_excluded=None -> must pass
    good_manifest = {
        "base_model": "BAAI/bge-reranker-v2-m3",
        "is_final_checkpoint": True,
        "training_scope": "all_allowed_task2_data",
        "val_fold_excluded": None,
        "smoke_only": False,
    }
    (ckpt_dir / "reranker_manifest.json").write_text(json.dumps(good_manifest))
    verified = assert_final_checkpoint(str(ckpt_dir), expected_base_model="BAAI/bge-reranker-v2-m3", component_name="reranker")
    assert verified["is_final_checkpoint"] is True


def test_policy_requires_generator_v6():
    """P0-11: Verify policy_requires_generator for all V6 policy forms."""
    assert policy_requires_generator("learned") is True
    assert policy_requires_generator("learned_model") is True
    assert policy_requires_generator("meta_selector") is True

    # Fixed baseline with extractive candidate does not require generator
    assert policy_requires_generator("fixed_baseline", "stitched_extract") is False
    assert policy_requires_generator("fixed_baseline", "focused_extract") is False
    assert policy_requires_generator("fixed_baseline", "focused_complete_clause") is False

    # Fixed baseline with generator-dependent candidate requires generator
    assert policy_requires_generator("fixed_baseline", "generated") is True
    assert policy_requires_generator("fixed_baseline", "snapped") is True
    assert policy_requires_generator("fixed_baseline", "strategy_f_300") is True
    assert policy_requires_generator("fixed_baseline", "strategy_f_1000") is True


def test_production_config_rejects_candidate_name_as_policy_type(tmp_path):
    """P0-10: Verify load_production_selection rejects 'type: generated'."""
    bad_yaml = {
        "schema_version": 3,
        "status": "UNVALIDATED",
        "stack": "stack_a",
        "candidate_policy": {
            "type": "generated",  # Invalid! Overloaded candidate name as type
        },
    }
    cfg_file = tmp_path / "bad_config.yaml"
    cfg_file.write_text(yaml.dump(bad_yaml))

    with pytest.raises(ValueError, match="Invalid candidate_policy type 'generated'"):
        load_production_selection(str(cfg_file))


def test_promote_production_selection_script(tmp_path):
    """P1-6: Verify promote_production_selection updates config with report SHA256 and PROMOTED status."""
    report_file = tmp_path / "promotion_report.json"
    sys_template = {
        "sample_ids_sha256": "abcdef123456",
        "sample_size": 250,
        "candidate_family_meteors": {"stitched_extract": 0.310},
        "retrieval_metrics": {"chunk_mrr": 0.45},
        "reranker_checkpoint": "checkpoints/reranker/best",
        "generator_model": "Qwen/Qwen2.5-3B-Instruct",
        "adapter_path": None,
        "dense_model": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        "no_mocks": True,
        "no_fallbacks": True,
    }
    report_data = {
        "screen_protocol_version": 8,
        "held_out_fold": 0,
        "sample_ids_sha256": "abcdef123456",
        "sample_size": 250,
        "evaluated_systems": {
            "R0G0": dict(sys_template, reranker_checkpoint="BAAI/bge-reranker-v2-m3"),
            "R1G0": dict(sys_template, reranker_checkpoint="checkpoints/reranker/best"),
            "R_SELECTED_G1": dict(sys_template, reranker_checkpoint="checkpoints/reranker/best"),
        },
        "selected_reranker": {
            "use_task_tuned": True,
            "checkpoint": "checkpoints/reranker/best",
            "decision_reason": "improved chunk MRR",
        },
        "selected_generator": {
            "use_qlora": False,
            "adapter": None,
            "decision_reason": "did not beat fixed baseline",
        },
        "final_measured_system_key": "R1G0",
        "candidate_policy": {
            "type": "fixed_baseline",
            "best_fixed_candidate": "stitched_extract",
        },
        "overall_deployable_winner": "stitched_extract",
        "overall_deployable_meteor": 0.310,
    }
    report_file.write_text(json.dumps(report_data))

    out_cfg = tmp_path / "promoted_config.yaml"
    promoted = promote_production_selection(
        report_path=str(report_file),
        config_path="configs/production_selection.yaml",
        output_path=str(out_cfg),
    )

    assert promoted["status"] == "PROMOTED"
    assert promoted["screen_protocol_version"] == 8
    assert promoted["source_screen_manifest"] == str(report_file)
    assert promoted["source_screen_sha256"] is not None
    assert promoted["reranker"]["use_task_tuned"] is True
    assert promoted["generator"]["use_qlora"] is False
    assert promoted["candidate_policy"]["best_fixed_candidate"] == "stitched_extract"

    # Reload from disk and validate
    loaded = load_production_selection(str(out_cfg))
    assert loaded.status == "PROMOTED"
    assert loaded.requires_generator is False
