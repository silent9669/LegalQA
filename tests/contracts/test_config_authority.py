import yaml
import os

def test_config_authority_and_no_drift():
    # 1. Authoritative configs exist
    smoke_path = "configs/kaggle_smoke_t4.yaml"
    colab_path = "configs/colab_train_a100.yaml"
    schema_path = "configs/dataset_schema.yaml"

    assert os.path.exists(smoke_path), f"Missing {smoke_path}"
    assert os.path.exists(colab_path), f"Missing {colab_path}"
    assert os.path.exists(schema_path), f"Missing {schema_path}"

    with open(smoke_path, "r", encoding="utf-8") as f:
        smoke_cfg = yaml.safe_load(f)
    with open(colab_path, "r", encoding="utf-8") as f:
        colab_cfg = yaml.safe_load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_cfg = yaml.safe_load(f)

    # 2. Base Model Alignment (<4B Competition Bound)
    assert smoke_cfg["generator"]["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert colab_cfg["generator"]["model_id"] == "Qwen/Qwen2.5-3B-Instruct"

    # 3. Liger Kernel Enabled Across Both Targets
    assert smoke_cfg["generator"]["use_liger_kernel"] is True
    assert colab_cfg["generator"]["use_liger_kernel"] is True

    # 4. Sequence Length Consistency
    assert smoke_cfg["data"]["max_seq_len"] in (1024, 2048)
    assert colab_cfg["data"]["max_seq_len"] == 2048

    # 5. Dataset Schema Target
    assert schema_cfg["schema_version"] == "2.0.0"
    assert "legal_chunks" in schema_cfg["tables"]
    assert "qa_unique" in schema_cfg["tables"]
