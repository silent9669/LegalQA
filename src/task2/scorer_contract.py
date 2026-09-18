"""Offline/official scorer contract: strict payload validation + labelled scoring.

Official scoring remains labelled-only. Resource failure raises, never
returns a fake zero. The public submission has no local references: its
official score stays null until a genuine receipt; only schema/ID checks
apply to it.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Union


def _reject_duplicate_keys(pairs: List[tuple]) -> Dict[str, Any]:
    seen: Dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key rejected: {key!r}")
        seen[key] = value
    return seen


def load_predictions_json(path: Union[str, Path]) -> Dict[str, Dict[str, str]]:
    """Load a predictions JSON file, rejecting duplicate keys before parsing."""
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(data, dict):
        raise ValueError("prediction payload must be a JSON object keyed by ID")
    return data


def validate_prediction_payload(
    predictions: Dict[str, Dict[str, str]],
    expected_ids: List[str],
) -> Dict[str, Any]:
    """Validate exact keys and answer-object values.

    - Key set must equal expected_ids exactly (missing or extra IDs fail
      with a ValueError mentioning "ID").
    - Every value must be an object with a nonempty ``answer`` string.
    Returns a report with counts and the payload hash.
    """
    if not isinstance(predictions, dict):
        raise ValueError("prediction payload must be a dict keyed by ID")
    expected = [str(v) for v in expected_ids]
    actual = [str(k) for k in predictions.keys()]
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"prediction ID mismatch: missing={missing[:5]} extra={extra[:5]}")
    if len(actual) != len(expected):
        raise ValueError(f"prediction ID count mismatch: got {len(actual)}, expected {len(expected)}")
    empty_answers = []
    invalid_values = []
    for key, value in predictions.items():
        if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
            invalid_values.append(str(key))
        elif not value["answer"].strip():
            empty_answers.append(str(key))
    if invalid_values:
        raise ValueError(f"prediction values must be objects with an answer string: {invalid_values[:5]}")
    if empty_answers:
        raise ValueError(f"empty answers rejected for IDs: {empty_answers[:5]}")
    payload_hash = hashlib.sha256(
        json.dumps(predictions, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"status": "PASS", "num_predictions": len(actual), "payload_sha256": payload_hash}


def verify_zip_inner_matches_loose(zip_path: Union[str, Path], loose_path: Union[str, Path]) -> Dict[str, Any]:
    """Verify the ZIP contains submission.json whose bytes equal the loose file.

    Returns sizes/digests: the container ZIP has its own digest, distinct
    from the inner payload digest.
    """
    loose_bytes = Path(loose_path).read_bytes()
    with zipfile.ZipFile(str(zip_path), "r") as z:
        names = z.namelist()
        if "submission.json" not in names:
            raise ValueError(f"ZIP missing submission.json entry: {names}")
        inner_bytes = z.read("submission.json")
    if inner_bytes != loose_bytes:
        raise ValueError("ZIP inner submission.json bytes differ from the loose submission.json")
    return {
        "loose_sha256": hashlib.sha256(loose_bytes).hexdigest(),
        "inner_sha256": hashlib.sha256(inner_bytes).hexdigest(),
        "zip_sha256": hashlib.sha256(Path(zip_path).read_bytes()).hexdigest(),
        "loose_bytes": len(loose_bytes),
        "inner_bytes": len(inner_bytes),
    }


def _require_meteor_resources() -> None:
    import nltk

    for resource in ("corpora/wordnet.zip", "corpora/omw-1.4.zip"):
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(resource.split("/")[1].split(".")[0], quiet=True)
            except Exception:
                pass
    try:
        nltk.data.find("corpora/wordnet.zip")
        nltk.data.find("corpora/omw-1.4.zip")
    except LookupError as exc:
        raise RuntimeError(f"scorer WordNet/OMW resources unavailable offline: {exc}") from exc


def score_labelled_rows(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Score labelled reference/prediction lists with the official recipe.

    Whitespace-tokenized METEOR plus ROUGE-L fmeasure without stemming,
    matching Scoring-Program-Task-LegalQA/scoring.py. Length mismatch raises
    (never zip-truncates); missing scorer resources raise (never fake zero).
    """
    if len(references) != len(predictions):
        raise ValueError(f"reference/prediction length mismatch: {len(references)} vs {len(predictions)}")
    if not references:
        raise ValueError("score_labelled_rows requires at least one labelled row")
    _require_meteor_resources()
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError as exc:
        raise RuntimeError(f"NLTK METEOR unavailable: {exc}") from exc
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError(f"rouge_score unavailable: {exc}") from exc
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    meteor_scores = []
    rouge_scores = []
    for ref, pred in zip(references, predictions):
        r_tokens = str(ref).split()
        p_tokens = str(pred).split()
        if not r_tokens or not p_tokens:
            raise ValueError("score_labelled_rows rejects empty reference or prediction strings")
        meteor_scores.append(float(meteor_score([r_tokens], p_tokens)))
        rouge_scores.append(float(scorer.score(str(ref), str(pred))["rougeL"].fmeasure))
    return {
        "meteor": float(sum(meteor_scores) / len(meteor_scores)),
        "rouge": float(sum(rouge_scores) / len(rouge_scores)),
        "num_rows": len(references),
    }


def build_offline_metric_report(payload: Dict[str, Any], scorer_sha: str) -> Dict[str, Any]:
    """Attach scorer identity to an offline metric payload (no public score)."""
    if not scorer_sha or len(scorer_sha) != 64:
        raise ValueError("build_offline_metric_report requires a 64-hex scorer_sha")
    report = dict(payload)
    report["scorer_sha256"] = scorer_sha
    report.setdefault("official_public_score", None)
    return report
