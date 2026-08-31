"""Surface-form preservation, candidate ensemble, and backward-compatibility selector."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.task2.candidates import (
    apply_strategy_f,
    build_citation_header,
    clean_statutory_text,
    deduplicate_repetitive_lines,
    generate_candidate_ensemble,
    snap_facts_to_evidence,
)


def select_best_answer_candidate(
    candidates: Dict[str, str],
    doc_name: str = "",
    article_num: str = "",
    clause_num: str = "",
    features: Optional[Dict[str, Any]] = None,
) -> str:
    """Calibrated rule selector choosing best candidate based on test-time signals."""
    # 1. Exact Memory is gold
    if candidates.get("exact_memory"):
        return candidates["exact_memory"].strip()

    # 2. High-confidence Similar QA memory
    if candidates.get("fuzzy_memory") and features and features.get("is_direct_reuse"):
        return candidates["fuzzy_memory"].strip()

    evidence = candidates.get("stitched_extract") or candidates.get("focused_extract") or ""
    clean_ev = clean_statutory_text(evidence)

    # 3. Snapped generation with Strategy F
    snapped = candidates.get("snapped", "").strip()
    if snapped and len(snapped) > 30 and not snapped.startswith("Căn cứ quy định pháp luật:\n[DOCUMENT]"):
        word_count = len(snapped.split())
        if word_count < 150:
            return candidates.get("strategy_f_1000", apply_strategy_f(snapped, clean_ev, max_chars=1000))
        elif word_count < 250:
            return candidates.get("strategy_f_600", apply_strategy_f(snapped, clean_ev, max_chars=600))
        return snapped

    gen = candidates.get("generated", "").strip()
    if gen and len(gen) > 30:
        word_count = len(gen.split())
        if word_count < 150:
            return candidates.get("strategy_f_1000", apply_strategy_f(gen, clean_ev, max_chars=1000))
        return gen

    # 4. Structured extract fallback
    if candidates.get("stitched_extract"):
        return candidates["stitched_extract"].strip()

    if candidates.get("focused_extract"):
        return candidates["focused_extract"].strip()

    header = build_citation_header(doc_name, article_num, clause_num)
    return header
