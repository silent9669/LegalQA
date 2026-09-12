import os
import yaml
import pytest

def test_kaggle_smoke_config_structure():
    path = "configs/kaggle_smoke_t4.yaml"
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg["profile_name"] == "kaggle_smoke_t4"
    assert cfg["devices"]["generator"] == "cuda:0"
    assert cfg["devices"]["retrieval"] == "cuda:1"
    assert cfg["generator"]["load_in_4bit"] is True
    assert cfg["probes"]["run_worstcase_probe"] is True
    assert cfg["probes"]["worstcase_steps"] == 3
    assert cfg["probes"]["endurance_steps"] == 30

def test_colab_train_config_structure():
    path = "configs/colab_train_a100.yaml"
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg["profile_name"] == "colab_train_a100"
    assert cfg["target_hardware"] == "NVIDIA A100"
    assert cfg["generator"]["torch_dtype"] == "bfloat16"
    assert cfg["generator"]["num_train_epochs"] >= 1
    assert "huggingface" in cfg

def test_dataset_schema_config_structure():
    path = "configs/dataset_schema.yaml"
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    assert "tables" in cfg
    assert "legal_chunks" in cfg["tables"]
    assert "qa_unique" in cfg["tables"]
    assert "qa_citations" in cfg["tables"]
    assert "fold_assignments" in cfg["tables"]
    assert "retrieval_labels" in cfg["tables"]
