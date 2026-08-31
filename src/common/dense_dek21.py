"""Dense DEk21 v2 Retriever (backward-compatibility alias for DenseRetriever)."""

from __future__ import annotations

from src.common.dense import DenseRetriever

# Alias for backwards compatibility
DEk21Retriever = DenseRetriever

__all__ = ["DEk21Retriever", "DenseRetriever"]
