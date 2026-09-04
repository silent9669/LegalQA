"""Pipeline orchestration and execution profiles for LegalQA Task 2 (V16)."""

from src.task2.pipeline.profiles import (
    ExecutionProfile,
    resolve_execution_profile,
    VALID_V16_PROFILES,
)

__all__ = [
    "ExecutionProfile",
    "resolve_execution_profile",
    "VALID_V16_PROFILES",
]
