import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.manifest import audit_parameter_manifest

def test_manifest_within_budget():
    components = {
        "generator": {"name": "Qwen/Qwen3-1.7B", "parameters": 1_700_000_000},
        "retriever": {"name": "Qwen/Qwen3-Embedding-0.6B", "parameters": 600_000_000},
        "reranker": {"name": "Qwen/Qwen3-Reranker-0.6B", "parameters": 600_000_000}
    }
    total, valid = audit_parameter_manifest(components)
    assert total == 2_900_000_000
    assert valid is True

def test_manifest_exceeds_budget():
    components = {
        "generator": {"name": "Large-7B", "parameters": 7_000_000_000}
    }
    total, valid = audit_parameter_manifest(components)
    assert valid is False
