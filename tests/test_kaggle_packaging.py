import json
import os
import sys
import shutil
from pathlib import Path
import pytest
import pandas as pd
from scripts.package_kaggle_dataset import package_kaggle_dataset


def test_kaggle_packaging_self_contained(tmp_path: Path):
    source_dir = tmp_path / "artifacts" / "task2"
    data_dir = source_dir / "data"
    data_dir.mkdir(parents=True)

    # Create dummy data artifacts
    pd.DataFrame([{"chunk_id": "c1", "text_raw": "t1"}]).to_parquet(data_dir / "legal_chunks.parquet")
    pd.DataFrame([{"qa_id": "q1", "question_raw": "q1", "answer_raw": "a1"}]).to_parquet(data_dir / "qa_unique.parquet")
    with open(data_dir / "known_qa.json", "w") as f:
        json.dump({"id_map": {}, "question_map": {}}, f)
    pd.DataFrame([{"qa_id": "q1", "article": "1"}]).to_parquet(data_dir / "qa_citations.parquet")
    pd.DataFrame([{"qa_id": "q1", "positive_chunk_id": "c1"}]).to_parquet(data_dir / "retrieval_labels.parquet")
    pd.DataFrame([{"qa_id": "q1", "fold_id": 0}]).to_parquet(data_dir / "fold_assignments.parquet")
    pd.DataFrame([{"qa_id": "q1", "positive_chunk_id": "c1", "negative_chunk_id": "c2"}]).to_parquet(data_dir / "reranker_training_pairs.parquet")

    staging_dir = tmp_path / "kaggle_dataset" / "staged"

    package_kaggle_dataset(
        source_dir=str(source_dir),
        staging_dir=str(staging_dir),
        include_code=True,
        dry_run=False,
    )

    # Verify data artifacts staged
    assert (staging_dir / "legal_chunks.parquet").exists()
    assert (staging_dir / "qa_unique.parquet").exists()
    assert (staging_dir / "known_qa.json").exists()
    assert (staging_dir / "reranker_training_pairs.parquet").exists()

    # Verify code artifacts staged (src, scripts, configs, requirements)
    staged_code = staging_dir / "code" / "LegalQA"
    assert (staged_code / "src").exists()
    assert (staged_code / "scripts").exists()
    assert (staged_code / "configs").exists()
    assert (staged_code / "requirements-kaggle.txt").exists()
    assert (staged_code / "code_manifest.json").exists()

    # Verify manifest contents
    assert (staging_dir / "code_manifest.json").exists()
    assert (staging_dir / "dataset_manifest.json").exists()
    assert (staging_dir / "dataset-metadata.json").exists()

    # Verify that modules can be imported from the staged runtime root
    staged_str = str(staged_code)
    if staged_str not in sys.path:
        sys.path.insert(0, staged_str)

    from src.task2.predict import LegalQAPipeline
    from scripts.preflight_kaggle import run_preflight_checks
    assert LegalQAPipeline is not None
    assert run_preflight_checks is not None


def test_package_kaggle_dataset_final_training_rejects_unvalidated(tmp_path: Path):
    unval_yaml = tmp_path / "production_selection.yaml"
    unval_yaml.write_text(
        "schema_version: 3\n"
        "status: UNVALIDATED\n"
        "screen_protocol_version: 1\n"
        "candidate_policy:\n"
        "  type: fixed_baseline\n"
        "  best_fixed_candidate: stitched_extract\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Production config status is 'UNVALIDATED'"):
        package_kaggle_dataset(
            source_dir="artifacts/task2",
            staging_dir=str(tmp_path / "stage"),
            profile="final_training",
            production_config_path=str(unval_yaml),
            dry_run=True,
        )


def test_package_kaggle_dataset_final_training_accepts_promoted_protocol_8(tmp_path: Path):
    promoted_yaml = tmp_path / "production_selection.yaml"
    promoted_yaml.write_text(
        "schema_version: 3\n"
        "status: PROMOTED\n"
        "screen_protocol_version: 8\n"
        "source_screen_manifest: artifacts/task2/evaluations/promotion_report.json\n"
        "source_screen_sha256: dummy_sha\n"
        "candidate_policy:\n"
        "  type: fixed_baseline\n"
        "  best_fixed_candidate: stitched_extract\n"
        "reranker:\n"
        "  use_task_tuned: false\n"
        "generator:\n"
        "  use_qlora: false\n",
        encoding="utf-8",
    )
    try:
        package_kaggle_dataset(
            source_dir="artifacts/task2",
            staging_dir=str(tmp_path / "stage"),
            profile="final_training",
            production_config_path=str(promoted_yaml),
            dry_run=True,
        )
    except RuntimeError as e:
        if "UNVALIDATED" in str(e) or "screen_protocol_version" in str(e):
            pytest.fail(f"Promoted Protocol-8 config should pass profile validation: {e}")
    except FileNotFoundError:
        pass  # expected if local raw artifacts are missing in test environment


def test_package_kaggle_dataset_default_profile_allows_unvalidated(tmp_path: Path):
    unval_yaml = tmp_path / "production_selection.yaml"
    unval_yaml.write_text(
        "schema_version: 3\n"
        "status: UNVALIDATED\n"
        "screen_protocol_version: 1\n",
        encoding="utf-8",
    )
    # Default profile is for probes and screening, must allow UNVALIDATED without raising
    try:
        package_kaggle_dataset(
            source_dir="artifacts/task2",
            staging_dir=str(tmp_path / "stage"),
            profile="default",
            production_config_path=str(unval_yaml),
            dry_run=True,
        )
    except RuntimeError as e:
        if "UNVALIDATED" in str(e):
            pytest.fail(f"Default profile must allow UNVALIDATED config: {e}")
    except FileNotFoundError:
        pass
