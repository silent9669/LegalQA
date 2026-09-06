import json
import pytest
from pathlib import Path
from src.common.hashing import sha256_file
from src.task2.production_config import (
    ProductionSelection,
    verify_promotion_provenance,
    validate_production_selection_for_profile,
)


def create_mock_report_and_manifest(tmp_path: Path):
    report_file = tmp_path / "promotion_report.json"
    report_data = {
        "screen_protocol_version": 8,
        "sample_ids_sha256": "abcdef1234567890" * 4,
        "sample_size": 250,
        "overall_deployable_winner": "stitched_extract",
        "overall_deployable_meteor": 0.3051,
    }
    report_file.write_text(json.dumps(report_data), encoding="utf-8")
    report_sha = sha256_file(report_file)

    manifest_file = tmp_path / "screen_run_manifest.json"
    manifest_data = {
        "runtime_api_version": 16,
        "screen_protocol_version": 8,
        "promotion_report_sha256": report_sha,
        "status": "SCREEN_PASS",
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")
    manifest_sha = sha256_file(manifest_file)

    return report_file, report_sha, manifest_file, manifest_sha


def test_verify_provenance_success(tmp_path: Path):
    report_file, report_sha, manifest_file, manifest_sha = create_mock_report_and_manifest(tmp_path)
    cfg = ProductionSelection(
        schema_version=3,
        status="PROMOTED",
        source_screen_manifest=str(report_file),
        source_screen_sha256=report_sha,
        stack="stack_a",
        use_task_tuned_reranker=False,
        reranker_base_model="BAAI/bge-reranker-v2-m3",
        reranker_checkpoint="checkpoints/reranker/best",
        use_qlora=False,
        generator_base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_path="checkpoints/generator/hf_adapter",
        max_new_tokens=384,
        candidate_policy="fixed_baseline",
        best_fixed_candidate="stitched_extract",
        selector_checkpoint=None,
        primary_evidence_pack="multi_seed_2500_chars",
        raw_config={"screen_protocol_version": 8},
        provenance={
            "promotion_report_path": str(report_file),
            "promotion_report_sha256": report_sha,
            "screen_run_manifest_path": str(manifest_file),
            "screen_run_manifest_sha256": manifest_sha,
            "sample_ids_sha256": "abcdef1234567890" * 4,
            "runtime_api_version": 16,
            "screen_protocol_version": 8,
        },
    )
    # Must succeed without error
    verify_promotion_provenance(cfg, search_roots=[tmp_path])


def test_verify_provenance_fails_on_tampered_report(tmp_path: Path):
    report_file, report_sha, manifest_file, manifest_sha = create_mock_report_and_manifest(tmp_path)
    # Tamper with the report file
    report_file.write_text(json.dumps({"tampered": True}), encoding="utf-8")

    cfg = ProductionSelection(
        schema_version=3,
        status="PROMOTED",
        source_screen_manifest=str(report_file),
        source_screen_sha256=report_sha,
        stack="stack_a",
        use_task_tuned_reranker=False,
        reranker_base_model="BAAI/bge-reranker-v2-m3",
        reranker_checkpoint="checkpoints/reranker/best",
        use_qlora=False,
        generator_base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_path="checkpoints/generator/hf_adapter",
        max_new_tokens=384,
        candidate_policy="fixed_baseline",
        best_fixed_candidate="stitched_extract",
        selector_checkpoint=None,
        primary_evidence_pack="multi_seed_2500_chars",
        raw_config={"screen_protocol_version": 8},
        provenance={
            "promotion_report_path": str(report_file),
            "promotion_report_sha256": report_sha,
            "runtime_api_version": 16,
            "screen_protocol_version": 8,
        },
    )
    with pytest.raises(RuntimeError, match="Promotion report SHA256 mismatch"):
        verify_promotion_provenance(cfg, search_roots=[tmp_path])


def test_verify_provenance_fails_on_missing_report(tmp_path: Path):
    cfg = ProductionSelection(
        schema_version=3,
        status="PROMOTED",
        source_screen_manifest=str(tmp_path / "nonexistent.json"),
        source_screen_sha256="0" * 64,
        stack="stack_a",
        use_task_tuned_reranker=False,
        reranker_base_model="BAAI/bge-reranker-v2-m3",
        reranker_checkpoint="checkpoints/reranker/best",
        use_qlora=False,
        generator_base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_path="checkpoints/generator/hf_adapter",
        max_new_tokens=384,
        candidate_policy="fixed_baseline",
        best_fixed_candidate="stitched_extract",
        selector_checkpoint=None,
        primary_evidence_pack="multi_seed_2500_chars",
        raw_config={"screen_protocol_version": 8},
        provenance={
            "promotion_report_path": str(tmp_path / "nonexistent.json"),
            "promotion_report_sha256": "0" * 64,
            "runtime_api_version": 16,
            "screen_protocol_version": 8,
        },
    )
    with pytest.raises(FileNotFoundError, match="Promotion report file not found"):
        verify_promotion_provenance(cfg, search_roots=[tmp_path])


def test_validate_profile_triggers_provenance_verification(tmp_path: Path):
    cfg = ProductionSelection(
        schema_version=3,
        status="PROMOTED",
        source_screen_manifest=str(tmp_path / "missing.json"),
        source_screen_sha256="0" * 64,
        stack="stack_a",
        use_task_tuned_reranker=False,
        reranker_base_model="BAAI/bge-reranker-v2-m3",
        reranker_checkpoint="checkpoints/reranker/best",
        use_qlora=False,
        generator_base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_path="checkpoints/generator/hf_adapter",
        max_new_tokens=384,
        candidate_policy="fixed_baseline",
        best_fixed_candidate="stitched_extract",
        selector_checkpoint=None,
        primary_evidence_pack="multi_seed_2500_chars",
        raw_config={"screen_protocol_version": 8},
        provenance={
            "promotion_report_path": str(tmp_path / "missing.json"),
            "promotion_report_sha256": "0" * 64,
            "runtime_api_version": 16,
            "screen_protocol_version": 8,
        },
    )
    with pytest.raises((FileNotFoundError, RuntimeError)):
        validate_production_selection_for_profile(
            cfg,
            profile="final_train_and_submit",
            allow_unvalidated_final=False,
            verify_provenance=True,
        )
