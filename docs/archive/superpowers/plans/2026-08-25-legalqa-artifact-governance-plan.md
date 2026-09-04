# LegalQA Artifact Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `artifacts/` the only authoritative data hierarchy, safely quarantine incompatible legacy artifacts, and provide reproducible manifests and collaboration downloads without committing large binaries.

**Architecture:** A small audit/manifest layer records hashes, schemas, counts, and provenance for every required artifact. Runtime scripts resolve only canonical `artifacts/` paths. Large immutable bundles live in private external storage and are verified locally by SHA-256.

**Tech Stack:** Python 3, pathlib, hashlib, json, pandas/pyarrow, pytest, Git, optional Hugging Face Hub or S3-compatible object storage.

**Spec:** `docs/superpowers/specs/2026-08-25-legalqa-artifacts-validation-model-design.md`

## Global Constraints

- Task 2 organizer data only.
- Never commit credentials, model weights, dense indexes, or multi-hundred-MB artifacts as ordinary Git blobs.
- Never substitute `trung_artifacts/legal_chunks.parquet` for the canonical 18-column parquet.
- Every remote download must be verified by SHA-256 before use.
- Cleanup must inspect differing files before deletion.

---

### Task 1: Add artifact manifest interfaces

**Files:**
- Create: `src/data/artifact_manifest.py`
- Create: `tests/test_artifact_manifest.py`
- Create: `artifacts/manifests/.gitkeep`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`
- Produces: `inspect_artifact(path: Path, logical_type: str) -> dict`
- Produces: `write_manifest(entries: list[dict], output_path: Path) -> None`

- [ ] **Step 1: Write failing tests for stable hashes and manifest output**

```python
from pathlib import Path
from src.data.artifact_manifest import sha256_file, write_manifest


def test_sha256_file_is_stable(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"legalqa")
    assert sha256_file(path) == "73eaeeaa8fc53cffeb462bc3e02e42d46f4a3e4a22b5e918d0dc063bcdfe2c22"


def test_write_manifest_sorts_by_path(tmp_path: Path):
    output = tmp_path / "manifest.json"
    write_manifest([{"path": "b"}, {"path": "a"}], output)
    assert output.read_text(encoding="utf-8").index('"a"') < output.read_text(encoding="utf-8").index('"b"')
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_artifact_manifest.py -v`
Expected: FAIL because `src.data.artifact_manifest` does not exist.

- [ ] **Step 3: Implement deterministic manifest helpers**

```python
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(entries: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda item: item["path"])
    output_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
```

Add parquet/JSON/ZIP metadata inspection without loading unnecessary content into memory.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_artifact_manifest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/artifact_manifest.py tests/test_artifact_manifest.py artifacts/manifests/.gitkeep
git commit -m "feat(data): add reproducible artifact manifests"
```

### Task 2: Add a canonical artifact audit command

**Files:**
- Create: `scripts/audit_artifacts.py`
- Modify: `tests/test_artifact_manifest.py`

**Interfaces:**
- Consumes: manifest helpers from Task 1.
- Produces: `artifacts/manifests/artifacts.json`
- Produces: nonzero exit status when required files or schemas are invalid.

- [ ] **Step 1: Write a failing test for the canonical legal chunk schema**

```python
from scripts.audit_artifacts import REQUIRED_CHUNK_COLUMNS


def test_required_chunk_columns_include_runtime_fields():
    assert {"doc_id", "content", "raw_text", "searchable_text"} <= REQUIRED_CHUNK_COLUMNS
```

- [ ] **Step 2: Run the focused test**

Run: `pytest tests/test_artifact_manifest.py::test_required_chunk_columns_include_runtime_fields -v`
Expected: FAIL because the command does not exist.

- [ ] **Step 3: Implement the audit command**

The command must validate:

- Raw QA counts: 7,000 / 500 / 1,000.
- Canonical count: 7,113.
- Chunk count: 365,046.
- Required 18-column schema.
- Exact public submission key parity when a submission exists.
- Context ZIP CRC.
- No literal `None`, `nan`, or `null` in generated answers.
- No artifact selected from `trung_artifacts/`.

- [ ] **Step 4: Run the command**

Run: `python scripts/audit_artifacts.py --output artifacts/manifests/artifacts.json`
Expected: report known warnings for 38 non-empty unchunked contexts, ambiguous QA groups, duplicate stitch parts, malformed citation labels, and current invalid answer text; exit zero only for structural integrity, with warnings recorded separately.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_artifact_manifest.py tests/test_data.py tests/test_chunker.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit_artifacts.py tests/test_artifact_manifest.py artifacts/manifests/artifacts.json
git commit -m "feat(data): audit canonical LegalQA artifacts"
```

### Task 3: Quarantine legacy artifacts and remove safe duplicates

**Files:**
- Move: `trung_artifacts/legal_chunks.parquet` to `artifacts/archive/trung-legacy/legal_chunks.parquet`
- Move: `trung_artifacts/known_qa.json` to `artifacts/archive/trung-legacy/known_qa.json`
- Delete: `trung_artifacts/qa_unique.parquet`
- Delete: `trung_artifacts/retrieval_labels.parquet`
- Create: `artifacts/archive/trung-legacy/provenance.json`
- Modify: `.gitignore`

**Interfaces:**
- Produces: an ignored quarantine directory that runtime code cannot resolve.

- [ ] **Step 1: Record legacy hashes and schemas in `provenance.json`**

Include original path, new path, byte size, SHA-256, parquet columns/rows, discovery date, and `runtime_compatible: false`.

- [ ] **Step 2: Move differing files and delete byte-identical duplicates**

Use filesystem moves only after hashes are recorded. Remove the now-empty `trung_artifacts/` directory.

- [ ] **Step 3: Ignore large quarantine payloads but keep provenance**

Add:

```gitignore
artifacts/archive/**
!artifacts/archive/**/provenance.json
```

- [ ] **Step 4: Run artifact audit and tests**

Run: `python scripts/audit_artifacts.py --output artifacts/manifests/artifacts.json`
Run: `pytest tests/ -v`
Expected: audit uses canonical paths and all tests pass.

- [ ] **Step 5: Commit**

```bash
git add .gitignore artifacts/archive/trung-legacy/provenance.json artifacts/manifests/artifacts.json
git commit -m "chore(data): quarantine incompatible legacy artifacts"
```

### Task 4: Retire legacy path fallbacks

**Files:**
- Modify: `validation.py`
- Modify: `scripts/prepare_artifacts.py`
- Modify: `scripts/run_oof_validation.py`
- Modify: `scripts/predict.py`
- Modify: `scripts/test_codabench_submission.py`
- Modify: `tests/test_validation_cli.py`

**Interfaces:**
- Produces: explicit failures when canonical artifact paths are missing.

- [ ] **Step 1: Write tests that reject old root/data paths**

Assert that defaults point only into `artifacts/` and a missing canonical path raises `FileNotFoundError` containing the exact path.

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/test_validation_cli.py -v`
Expected: FAIL while silent fallbacks remain.

- [ ] **Step 3: Remove fallback resolution**

Delete fallbacks for `data/raw`, `data/intermediate`, root parquet, root JSONL, and root submission paths. Keep user-supplied explicit CLI paths supported.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_validation_cli.py tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add validation.py scripts tests/test_validation_cli.py
git commit -m "refactor(data): enforce canonical artifact paths"
```

### Task 5: Add model and remote bundle manifests

**Files:**
- Create: `configs/models.yaml`
- Create: `artifacts/manifests/models.json`
- Create: `scripts/verify_download.py`
- Create: `tests/test_model_manifest.py`
- Modify: `src/utils/manifest.py`

**Interfaces:**
- Produces: `load_model_manifest(path: Path) -> dict`
- Produces: `total_learned_parameters(manifest: dict) -> int`
- Produces: checksum-verified external bundle downloads.

- [ ] **Step 1: Write tests for conservative parameter accounting**

```python
from src.utils.manifest import total_learned_parameters


def test_recommended_stack_stays_below_four_billion():
    manifest = {"models": [
        {"parameters": 595_776_512},
        {"parameters": 595_776_512},
        {"parameters": 2_031_739_904},
    ]}
    assert total_learned_parameters(manifest) == 3_223_292_928
    assert total_learned_parameters(manifest) < 4_000_000_000
```

- [ ] **Step 2: Implement manifest validation**

Require model ID, pinned revision, SHA-256 or repository revision, license, learned parameters, role, loaded-at-inference flag, and local path/remote URI.

- [ ] **Step 3: Implement checksum verification**

`verify_download.py` must write to a temporary file, verify expected SHA-256, then atomically rename. It must never print credentials.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_model_manifest.py tests/test_manifest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/models.yaml artifacts/manifests/models.json scripts/verify_download.py src/utils/manifest.py tests/test_model_manifest.py
git commit -m "feat(models): add checkpoint and download manifests"
```

### Task 6: Document team collaboration workflow

**Files:**
- Modify: `README.md`
- Create: `docs/artifact-collaboration.md`

**Interfaces:**
- Produces: clear separation between Git content and external large-file storage.

- [ ] **Step 1: Document storage policy**

State:

- GitHub: source, tests, configs, manifests, checksums, download scripts.
- Private Hugging Face model repo: model checkpoints and adapters.
- R2/S3 bucket: data bundles, dense indexes, OOF traces, submission archives.
- Git LFS: optional, not preferred for frequently changing multi-GB files.

- [ ] **Step 2: Document bootstrap flow**

```bash
python scripts/verify_download.py --manifest artifacts/manifests/artifacts.json
python scripts/audit_artifacts.py --output artifacts/manifests/artifacts.json
pytest tests/ -v
```

- [ ] **Step 3: Correct README status claims**

Label dense retrieval, cross-encoder, generator, provision memory, and learned selector as planned. Record exact current parquet size and active pipeline.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/artifact-collaboration.md
git commit -m "docs: define LegalQA artifact collaboration workflow"
```

### Task 7: Synchronize implementation status to Notion

**Files:**
- Create: `docs/notion/task-2-status-update.md`
- Modify externally: Task 2 Notion root and affected component pages

**Interfaces:**
- Consumes: artifact audit, full validation reports, model manifests, and repository status taxonomy.
- Produces: a versioned paste-ready update and, when an authenticated user session is available, matching Notion edits.

- [ ] **Step 1: Build the status update from stored evidence**

Use only claims supported by committed code, tests, manifests, and validation reports. Label every component exactly one of: Implemented, Measured, Configured but unused, Planned, or Unverified.

- [ ] **Step 2: Correct current architecture claims**

Record the active flow as exact memory → BM25 → lexical reranker → Article Stitcher → extractive composer → source snap → rule selector. Label dense retrieval, neural reranking, generation, provision memory, and learned selection according to their actual state at execution time.

- [ ] **Step 3: Record artifact and collaboration policy**

State that `artifacts/` is authoritative, the legacy `trung_artifacts` bundle was quarantined/de-duplicated, GitHub stores only code and small manifests, private Hugging Face stores checkpoints/adapters, and R2/S3 stores artifact bundles/indexes/traces.

- [ ] **Step 4: Edit Notion only through the user's authenticated session**

Do not collect credentials or bypass login. If the session is not authenticated, leave `docs/notion/task-2-status-update.md` complete and report authentication as the sole blocker.

- [ ] **Step 5: Verify external edits against the local update**

Re-read the edited root and affected subpages. Check that measured scores include their run IDs and that no planned component is presented as implemented.

- [ ] **Step 6: Commit the local synchronization record**

```bash
git add docs/notion/task-2-status-update.md
git commit -m "docs: synchronize Task 2 implementation status"
```
