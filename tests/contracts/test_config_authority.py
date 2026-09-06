import yaml
from pathlib import Path


def test_config_authority_and_no_drift():
    # Load authoritative configs
    with open("configs/pipeline.yaml", "r", encoding="utf-8") as f:
        pipeline_cfg = yaml.safe_load(f)
    with open("configs/models.yaml", "r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)
    with open("configs/runtime_api.yaml", "r", encoding="utf-8") as f:
        runtime_cfg = yaml.safe_load(f)
    with open("configs/production_selection.yaml", "r", encoding="utf-8") as f:
        prod_cfg = yaml.safe_load(f)
    with open("configs/task2.yaml", "r", encoding="utf-8") as f:
        task2_cfg = yaml.safe_load(f)

    # 1. Runtime API
    assert runtime_cfg["runtime_api_version"] == 16

    # 2. Models identity alignment with Stack A
    retriever_model = models_cfg["stacks"]["stack_a"]["dense_model"]
    reranker_model = models_cfg["stacks"]["stack_a"]["reranker_model"]
    generator_model = models_cfg["stacks"]["stack_a"]["generator_model"]

    assert pipeline_cfg["retrieval"]["dense"]["stack_a_model"] == retriever_model
    assert pipeline_cfg["reranker"]["model"] == reranker_model
    assert pipeline_cfg["generation"]["stack_a_model"] == generator_model

    assert prod_cfg["retrieval"]["dense"]["model"] == retriever_model
    assert prod_cfg["reranker"]["base_model"] == reranker_model
    assert prod_cfg["generator"]["base_model"] == generator_model

    # 3. Parameter alignment across pipeline and task2.yaml
    assert task2_cfg["models"]["retriever"]["top_k"] == 50
    assert task2_cfg["models"]["generator"]["max_new_tokens"] == 384
