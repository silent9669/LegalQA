import json
import os
import sys
import shutil
from pathlib import Path
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
