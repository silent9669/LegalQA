# LegalQA CodaBench Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sampled random-row validator with deterministic, leakage-resistant, full-data validation that exactly mirrors the official CodaBench scorer and exposes stage-level failure causes.

**Architecture:** Immutable audit tables feed deterministic question-blocked and document-held-out fold manifests. The pipeline returns a structured trace per query, and evaluators compute official answer metrics plus retrieval/reranking/fidelity diagnostics and grouped bootstrap confidence intervals.

**Tech Stack:** Python 3, pandas/pyarrow, NumPy, NLTK METEOR, vendored official ROUGE-L, pytest.

**Spec:** `docs/superpowers/specs/2026-08-25-legalqa-artifacts-validation-model-design.md`

## Global Constraints

- Evaluate all 7,113 canonical QA rows for promotion runs.
- Keep smoke tests explicit and manifest-backed; never label them full OOF.
- Primary folds must prevent exact and near-duplicate question leakage.
- The official scorer contract requires exact key equality.
- The legal corpus remains available to retrieval in document-held-out tests; only QA-derived supervision is held out.
- No network downloads during evaluation.

---

### Task 1: Make the local scorer contract exact

**Files:**
- Modify: `src/evaluation/codabench_eval.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `validate_prediction_keys(y_pred: dict, y_true: dict) -> None`
- Produces: `evaluate_predictions(..., return_per_row: bool = False) -> dict`

- [ ] **Step 1: Write failing tests for key parity and string conversion**

```python
import pytest
from src.evaluation.codabench_eval import evaluate_predictions


def test_evaluator_rejects_missing_prediction_keys():
    with pytest.raises(ValueError, match="prediction keys"):
        evaluate_predictions({"1": {"answer": "a"}}, {"1": "a", "2": "b"})


def test_evaluator_matches_official_string_conversion():
    score = evaluate_predictions({"1": {"answer": 123}}, {"1": 123})
    assert score["meteor"] >= 0.999
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_evaluation.py -v`
Expected: missing-key test fails because the current evaluator intersects keys.

- [ ] **Step 3: Implement official parity**

Use exactly:

```python
reference_tokens = str(reference).split()
prediction_tokens = str(prediction).split()
score = meteor_score([reference_tokens], prediction_tokens)
```

Reject missing/extra keys and malformed answer objects before scoring. Remove import-time NLTK downloads; instead fail with an actionable offline resource message.

- [ ] **Step 4: Compare fixtures against official scorer**

Run both implementations on punctuation, numeric, empty, Unicode NFC/NFD, and non-string fixtures and assert equal means within `1e-12`.

- [ ] **Step 5: Run tests and commit**

```bash
pytest tests/test_evaluation.py Scoring-Program-Task-LegalQA/tests -v
git add src/evaluation/codabench_eval.py tests/test_evaluation.py
git commit -m "fix(eval): enforce exact CodaBench scorer parity"
```

### Task 2: Build a source-aware QA audit table

**Files:**
- Create: `src/evaluation/data_audit.py`
- Create: `scripts/build_validation_data.py`
- Create: `tests/test_data_audit.py`

**Interfaces:**
- Produces: `build_qa_audit(train_path: Path, warmup_path: Path) -> pd.DataFrame`
- Produces columns: `row_id`, `source_split`, `source_id`, `question`, `answer`, `normalized_question`, `question_hash`, `answer_hash`, `exact_group_id`, `conflict_group`, `question_type`, `answer_length`.

- [ ] **Step 1: Write tests for overlap preservation**

Assert 7,500 raw rows remain visible, 387 overlap groups are identifiable, and 14 conflicting normalized-question groups are flagged.

- [ ] **Step 2: Implement immutable audit construction**

Do not deduplicate before provenance columns are produced. Use stable SHA-256 content hashes and deterministic group IDs derived from sorted hashes.

- [ ] **Step 3: Materialize audit artifact**

Run:

```bash
python scripts/build_validation_data.py \
  --train artifacts/raw/train.json \
  --warmup artifacts/raw/warmup.json \
  --output artifacts/validation/qa_audit.parquet
```

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/test_data_audit.py -v
git add src/evaluation/data_audit.py scripts/build_validation_data.py tests/test_data_audit.py
git commit -m "feat(eval): preserve QA provenance and conflicts"
```

### Task 3: Resolve reliable document/article evidence labels

**Files:**
- Modify: `src/data/label_miner.py`
- Create: `src/data/citation_resolver.py`
- Create: `tests/test_citation_resolver.py`
- Modify: `scripts/prepare_artifacts.py`

**Interfaces:**
- Produces: `resolve_citations(answer: str, chunks: pd.DataFrame) -> list[ResolvedCitation]`
- `ResolvedCitation` fields: document key, article, clause, point, matched document IDs, matched chunk IDs, confidence, unresolved reason.

- [ ] **Step 1: Write failing tests for malformed values and multiple citations**

```python
def test_single_letter_is_not_a_document_number():
    assert resolve_document_number("H") is None


def test_parser_retains_multiple_citations():
    citations = parse_legal_citations("Điều 1 Nghị định 1/2020/NĐ-CP và Điều 2 Luật 2/2021/QH15")
    assert len(citations) == 2
```

- [ ] **Step 2: Implement constrained parsing and corpus resolution**

Resolution priority:

1. normalized official document number;
2. filename/link-derived number;
3. normalized title/name;
4. explicit unresolved bucket.

Never coerce malformed text to a real document.

- [ ] **Step 3: Write separate artifacts**

```text
artifacts/labels/qa_citations.parquet
artifacts/labels/retrieval_labels.parquet
artifacts/labels/citation_resolution_report.json
```

- [ ] **Step 4: Add quality gates**

The report records matched/unmatched counts, confidence tiers, multiple-citation coverage, and examples. The script fails if a one-character label is assigned to a real document.

- [ ] **Step 5: Run tests and commit**

```bash
pytest tests/test_label_miner.py tests/test_citation_resolver.py -v
git add src/data scripts/prepare_artifacts.py tests/test_citation_resolver.py
git commit -m "feat(data): resolve gold citations to corpus evidence"
```

### Task 4: Materialize deterministic fold manifests

**Files:**
- Create: `src/evaluation/folds.py`
- Create: `scripts/build_validation_folds.py`
- Create: `tests/test_folds.py`

**Interfaces:**
- Produces: `build_question_blocked_folds(audit: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame`
- Produces: `build_document_heldout_folds(audit: pd.DataFrame, labels: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame`

- [ ] **Step 1: Write tests that prohibit group leakage**

```python
def test_exact_question_group_never_crosses_folds(folds):
    assert folds.groupby("exact_group_id")["fold"].nunique().max() == 1


def test_reliable_document_group_never_crosses_stress_folds(folds):
    assert folds.dropna(subset=["doc_key"]).groupby("doc_key")["fold"].nunique().max() == 1
```

- [ ] **Step 2: Implement exact and near-duplicate grouping**

Use character 3–5 gram TF-IDF or deterministic MinHash candidates, cosine threshold `0.92`, and a secondary token-Jaccard guard. Save pair similarities and manually-reviewable borderline cases.

- [ ] **Step 3: Implement split balancing**

Balance source, question type, answer-length bin, citation availability, and document frequency without splitting atomic groups.

- [ ] **Step 4: Save manifests**

```text
artifacts/validation/folds_question_blocked.parquet
artifacts/validation/folds_document_heldout.parquet
artifacts/validation/fold_report.json
```

- [ ] **Step 5: Run tests and commit**

```bash
pytest tests/test_folds.py -v
git add src/evaluation/folds.py scripts/build_validation_folds.py tests/test_folds.py
git commit -m "feat(eval): add leakage-resistant validation folds"
```

### Task 5: Add structured pipeline traces

**Files:**
- Modify: `src/pipeline.py`
- Modify: `src/selector/candidate_selector.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `LegalQAPipeline.predict_with_trace(sample_id: str, question: str) -> PredictionTrace`
- Keeps: `predict(...) -> str` as a compatibility wrapper.

- [ ] **Step 1: Write a failing trace test**

Assert trace fields include memory hit type, retrieval candidates/scores, reranked candidates/scores, stitched evidence IDs, candidates, selected name, prediction, and latency.

- [ ] **Step 2: Add immutable trace dataclasses**

Do not repeat full corpus text in the persisted trace; store chunk IDs plus selected evidence text only.

- [ ] **Step 3: Make `predict()` delegate to `predict_with_trace()`**

```python
def predict(self, sample_id: str, question: str) -> str:
    return self.predict_with_trace(sample_id, question).prediction
```

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/test_pipeline.py tests/test_selector.py -v
git add src/pipeline.py src/selector/candidate_selector.py tests/test_pipeline.py
git commit -m "feat(pipeline): expose reproducible prediction traces"
```

### Task 6: Add stage-level retrieval and answer metrics

**Files:**
- Create: `src/evaluation/retrieval_metrics.py`
- Create: `src/evaluation/fidelity_metrics.py`
- Create: `tests/test_retrieval_metrics.py`
- Create: `tests/test_fidelity_metrics.py`

**Interfaces:**
- Produces document/article/chunk Recall@K, MRR, nDCG, and evidence coverage.
- Produces citation, number, date, duration, and amount fidelity metrics.

- [ ] **Step 1: Write metric fixture tests**

Use small ranked lists with hand-calculated Recall, reciprocal rank, and DCG values. Include Vietnamese amounts such as `6.000.000 đồng` and dates such as `01/01/2026`.

- [ ] **Step 2: Implement pure metric functions**

Functions must be deterministic and side-effect-free. Unknown gold evidence is excluded from evidence metrics and reported in the denominator audit.

- [ ] **Step 3: Run tests and commit**

```bash
pytest tests/test_retrieval_metrics.py tests/test_fidelity_metrics.py -v
git add src/evaluation tests/test_retrieval_metrics.py tests/test_fidelity_metrics.py
git commit -m "feat(eval): add retrieval and legal fidelity metrics"
```

### Task 7: Replace the OOF runner

**Files:**
- Rewrite: `scripts/run_oof_validation.py`
- Modify: `validation.py`
- Modify: `tests/test_validation_cli.py`

**Interfaces:**
- Consumes fold manifests and `predict_with_trace`.
- Produces per-query parquet traces and aggregate JSON reports.

- [ ] **Step 1: Write CLI tests**

Require explicit modes:

```text
--mode smoke
--mode question-blocked
--mode document-heldout
```

Promotion modes reject `--samples`; smoke mode requires a saved sample manifest.

- [ ] **Step 2: Implement fold-manifest loading and integrity checks**

Fail before inference if IDs are missing, duplicated, assigned to multiple folds, or hashes disagree with the artifact manifest.

- [ ] **Step 3: Persist outputs**

```text
artifacts/validation/runs/<run_id>/manifest.json
artifacts/validation/runs/<run_id>/predictions.parquet
artifacts/validation/runs/<run_id>/traces.parquet
artifacts/validation/runs/<run_id>/metrics.json
```

- [ ] **Step 4: Add grouped paired bootstrap**

Use deterministic seeds and resample atomic question groups for the main split and document clusters for the stress split.

- [ ] **Step 5: Run smoke validation**

Run: `python validation.py --mode smoke --sample-manifest artifacts/validation/smoke_ids.json`
Expected: complete trace and metric files with exact key parity.

- [ ] **Step 6: Run full baseline validation**

Run: `python validation.py --mode question-blocked --splits 5`
Run: `python validation.py --mode document-heldout --splits 5`
Expected: all 7,113 canonical IDs covered exactly once per split family.

- [ ] **Step 7: Commit**

```bash
git add validation.py scripts/run_oof_validation.py tests/test_validation_cli.py
git commit -m "feat(eval): run full benchmark-equivalent LegalQA validation"
```

### Task 8: Add promotion comparison reports

**Files:**
- Create: `scripts/compare_validation_runs.py`
- Create: `tests/test_compare_validation_runs.py`

**Interfaces:**
- Produces paired deltas, confidence intervals, subgroup regressions, and pass/fail promotion status.

- [ ] **Step 1: Write tests for promotion thresholds**

Verify rejection when METEOR delta is below `0.005`, bootstrap lower bound is nonpositive, Recall@10 regresses, or a major subgroup loses more than `0.02`.

- [ ] **Step 2: Implement comparison command**

```bash
python scripts/compare_validation_runs.py \
  --baseline artifacts/validation/runs/baseline \
  --candidate artifacts/validation/runs/candidate
```

- [ ] **Step 3: Run tests and commit**

```bash
pytest tests/test_compare_validation_runs.py -v
git add scripts/compare_validation_runs.py tests/test_compare_validation_runs.py
git commit -m "feat(eval): enforce LegalQA promotion gates"
```
