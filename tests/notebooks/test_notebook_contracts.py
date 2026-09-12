import json
import os
import ast
import re
import pytest

def test_kaggle_smoke_notebook_contract():
    nb_path = "notebooks/kaggle_smoke.ipynb"
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

def test_colab_train_notebook_contract():
    nb_path = "notebooks/colab_a100_train.ipynb"
    assert os.path.exists(nb_path), f"Missing {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    assert len(cells) >= 5, "Expected at least 5 cells in Colab A100 notebook"

    source_all = "\n".join("".join(c.get("source", [])) for c in cells)
    assert "colab_train_a100.yaml" in source_all
    assert "A100" in source_all
    assert "kaggle_smoke_report.json" in source_all
    assert "verify_smoke_pass" in source_all

def test_kernel_metadata_contract():
    meta_path = "notebooks/kernel-metadata.json"
    assert os.path.exists(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["id"] == "phucdangg/legalqa-training"
    assert meta["code_file"] == "kaggle_smoke.ipynb"
    assert meta.get("machine_shape") == "NvidiaTeslaT4"
    assert "phucdangg/legalqa-task2-clean-data" in meta["dataset_sources"]

def test_notebook_cells_python_ast_compilation():
    """Verify that every python code cell across all notebooks compiles with ast.parse."""
    notebooks = [
        "notebooks/kaggle_smoke.ipynb",
        "notebooks/colab_a100_train.ipynb",
    ]
    for nb_path in notebooks:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "code":
                source_lines = cell.get("source", [])
                # Strip IPython magics like ! or % for ast.parse
                clean_lines = [
                    line for line in source_lines
                    if not line.strip().startswith("!") and not line.strip().startswith("%")
                ]
                code = "".join(clean_lines)
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {nb_path} cell {i}: {e}")
