"""Final-mode execution contract: reject mocks, fallbacks and substitutions."""

from __future__ import annotations

from typing import Any, Dict, List


def validate_execution_contract(
    profile: Dict[str, Any],
    resources: Dict[str, Any],
    final_mode: bool,
) -> None:
    """Validate the execution contract before a final stage runs.

    Smoke/probe modes (final_mode=False) may use explicitly labelled mocks.
    Final mode rejects: mock dense indexes, fallback/mock generation,
    missing dense or BM25 indexes, GPU-to-CPU substitution, a missing
    required adapter, unvalidated promotion, and a generator-dependent
    policy without a loaded generator. Every violation raises ValueError.
    """
    if not final_mode:
        return
    if resources.get("dense_mock") or resources.get("mock_dense"):
        raise ValueError("final_mode rejects mock dense retriever")
    if resources.get("dense_fallback"):
        raise ValueError("final_mode rejects dense fallback index")
    if resources.get("generator_fallback") or resources.get("fallback_generator"):
        raise ValueError("final_mode rejects fallback generator")
    if resources.get("mock_generator"):
        raise ValueError("final_mode rejects mock generator")
    if resources.get("dense_index_missing") or resources.get("missing_index"):
        raise ValueError("final_mode rejects missing dense index")
    if resources.get("bm25_index_missing"):
        raise ValueError("final_mode rejects missing BM25 index")
    gen_device = str(resources.get("generator_device", "") or "")
    ret_device = str(resources.get("retrieval_device", "") or "")
    if gen_device == "cpu" or ret_device == "cpu" or resources.get("cpu_substitution"):
        raise ValueError("final_mode rejects GPU-to-CPU substitution")
    if resources.get("require_adapter") and not resources.get("adapter_present"):
        raise ValueError("final_mode rejects missing required adapter")
    if resources.get("promotion_validated") is False:
        raise ValueError("final_mode rejects unvalidated promotion provenance")
    if resources.get("policy_needs_generator") and not resources.get("generator_loaded"):
        raise ValueError("final_mode rejects generated policy without a loaded generator")
    if resources.get("extractive_only") and resources.get("policy_needs_generator"):
        raise ValueError("final_mode rejects extractive-only pipeline for a generated policy")


def verify_submission_ids(
    expected: List[str],
    submission: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """Verify the 1,000-ID public submission via the scorer payload validator."""
    from src.task2.scorer_contract import validate_prediction_payload

    return validate_prediction_payload(submission, expected)
