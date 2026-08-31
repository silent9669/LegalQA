"""Structured Article Stitcher and Evidence Packer for LegalQA Task 2 (backward compatibility wrapper)."""

from __future__ import annotations

from src.task2.evidence_packer import EvidencePacker

# Alias for backwards compatibility
ArticleStitcher = EvidencePacker

__all__ = ["ArticleStitcher", "EvidencePacker"]
