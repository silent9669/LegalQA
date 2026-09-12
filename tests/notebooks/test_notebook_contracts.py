import json
import os
import pytest

def test_kaggle_smoke_notebook_contract():
    nb_path = "notebooks/kaggle_smoke_test.ipynb"
    assert os.path.exists(nb_path), f"Missing {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    assert len(cells) >= 5, "Expected at least 5 cells in Kaggle smoke notebook"

    source_all = "\n".join("".join(c.get("source", [])) for c in cells)
    assert "kaggle_smoke_t4.yaml" in source_all
    assert "kaggle_smoke_report.json" in source_all
    assert "HF_DEACTIVATE_ASYNC_LOAD" in source_all
    assert "/kaggle/input/**/code/LegalQA" not in source_all, "Notebook must not look for code inside dataset!"

def test_kaggle_kernel_synced():
    kernel_nb = "kaggle_kernel/legalqa_gpu_pipeline.ipynb"
    assert os.path.exists(kernel_nb), f"Missing {kernel_nb}"
    with open(kernel_nb, "r", encoding="utf-8") as f:
        nb = json.load(f)
    source_all = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
    assert "/kaggle/input/**/code/LegalQA" not in source_all, "Kernel notebook must not expect code in dataset"
    assert "kaggle_smoke_t4.yaml" in source_all

def test_kernel_metadata_slug():
    meta_path = "kaggle_kernel/kernel-metadata.json"
    assert os.path.exists(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["id"] == "phucdangg/legalqa-training"
    assert "phucdangg/legalqa-task2-clean-data" in meta["dataset_sources"]
