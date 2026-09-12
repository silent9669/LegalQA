import os
import pytest
from src.task2.dataset.validator import validate_dataset

def test_dataset_integrity_on_staged():
    staged_dir = "kaggle_dataset/staged"
    if not os.path.exists(os.path.join(staged_dir, "qa_unique.parquet")):
        pytest.skip("Staged dataset artifacts not found locally.")

    report = validate_dataset(data_dir=staged_dir, schema_path="configs/dataset_schema.yaml")
    assert report["status"] == "PASS", f"Dataset validation failed: {report.get('errors')}"
    assert report["manifest_verified"] is True
    assert "code" not in report.get("detected_directories", [])
    assert report["tables_validated"]["qa_unique"]["rows"] > 0
    assert report["tables_validated"]["legal_chunks"]["rows"] > 0
