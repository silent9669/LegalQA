"""Canonical evaluation metrics for LegalQA Task 2.

Matches official competition scoring using whitespace-tokenized METEOR (via NLTK WordNet).
"""

from __future__ import annotations

from typing import List

import nltk
import numpy as np
from nltk.translate.meteor_score import meteor_score

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


def ensure_meteor_resources() -> None:
    """Ensure WordNet and omw-1.4 resources are available for NLTK METEOR scoring."""
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
        except Exception:
            pass

    try:
        nltk.data.find("corpora/omw-1.4.zip")
    except LookupError:
        try:
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            pass


def official_meteor(reference: str, prediction: str) -> float:
    """Compute official whitespace-tokenized METEOR score for a single pair."""
    ensure_meteor_resources()
    r_tokens = str(reference).split()
    p_tokens = str(prediction).split()
    if not r_tokens or not p_tokens:
        return 0.0
    return float(meteor_score([r_tokens], p_tokens))


def calculate_official_meteor(references: List[str], predictions: List[str]) -> float:
    """Compute official whitespace-tokenized METEOR score across a list of reference-prediction pairs."""
    ensure_meteor_resources()
    scores = []
    for r, p in zip(references, predictions):
        scores.append(official_meteor(r, p))
    return float(np.mean(scores)) if scores else 0.0


def calculate_rouge_l(references: List[str], predictions: List[str]) -> float:
    """Compute ROUGE-L f-measure without stemming."""
    if rouge_scorer is None or not references:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(str(r), str(p))["rougeL"].fmeasure for r, p in zip(references, predictions)]
    return float(np.mean(scores)) if scores else 0.0
