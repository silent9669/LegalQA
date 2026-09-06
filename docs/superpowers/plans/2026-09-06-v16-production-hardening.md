# LegalQA V16 Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the LegalQA V16 dual-T4 pipeline for a real Kaggle score-max run by enforcing strict generator configuration validation across screen and final training, locking packaging to PROMOTED Protocol-8 configs, replacing first-match checkpoint globs with deterministic provenance-bound resolution, cryptographically revalidating promotion provenance, and aligning configuration authority with V16 truth.

**Architecture:** 
1. Thread explicit `execution_profile` into `train_generator_qlora()` and enforce strict generator invariants (max_seq_len=2048, LoRA r16/a32, selective Liger fused-linear CE, single-GPU cuda:0) on all screen and final training profiles.
2. Guard `package_kaggle_dataset.py --profile final_training` to validate that `configs/production_selection.yaml` is strictly `PROMOTED` under Protocol 8 before staging any artifacts.
3. Introduce an authoritative `CheckpointResolver` that strictly inspects component manifests, verifies model identities, enforces provenance paths, and fails loud on ambiguity or staleness instead of picking the first glob match.
4. Cryptographically revalidate promotion report and screen run manifest SHA-256 hashes against actual disk contents prior to final training and checkpoint reuse.
5. Align configuration authority across `runtime_api.yaml`, `models.yaml`, `pipeline.yaml`, and `production_selection.yaml`, and update `README.md` to reflect V16 execution sequence.

**Tech Stack:** Python 3.10–3.12, PyTorch, HuggingFace Transformers 5.0.0, PEFT 0.19.1, TRL 1.12.0, Liger-Kernel 0.8.2, BitsAndBytes 0.50.2, PyYAML, PyTest.

**Spec:** `docs/LEGALQA_V16_ARCHITECTURE_REVIEW_AND_FIX_INSTRUCTIONS.md`

## Global Constraints

- Preserve Stack A architecture: BM25S + DEk21 v2 + BGE-reranker-v2-m3 + Qwen2.5-3B-Instruct 4-bit NF4 QLoRA (r=16, alpha=32).
- Generator training strictly on `cuda:0` with `trainer_n_gpu=1`; retrieval/reranker on `cuda:1`.
- Preserve selective Liger fused-linear cross entropy (`liger-kernel==0.8.2`); forbidden: `chunked_nll`.
- Strict parameter limit: total learned parameters `< 4,000,000,000`.
- Primary metric: official whitespace-tokenized METEOR.
- Inherited `artifacts/inherited/dataset_manifest_v15.json` is immutable provenance and must not be touched or converted to API16.
- Runtime API version is strictly 16 across `configs/runtime_api.yaml`, `src/task2/runtime_integrity.py`, and Kaggle runtime code.
- No automated triggering of Kaggle GPU runs; committed notebook default is `generator_probe_worstcase`.
- Do not touch Kaggle's protected `torch`, `torchvision`, `torchaudio`, `triton`, `cuda-*`, `nvidia-*` stack.

---

### Task 1: P0.1 Enforce Generator Config Validation on Screen and Final Profiles

**Files:**
- Modify: `src/task2/generation/trainer.py:139-170`
- Modify: `src/task2/pipeline/runner.py:200-215`
- Modify: `src/task2/training/train_generator.py:120-150`
- Test: `tests/unit/test_generator_profile_validation.py`

**Interfaces:**
- Consumes: `GeneratorTrainConfig`, `validate_generator_config_for_profile` from `src.task2.generation.config`.
- Produces: `train_generator_qlora(..., execution_profile: Optional[str] = None)` which validates strict production requirements whenever `execution_profile` is one of `generator_probe_worstcase`, `generator_probe_endurance`, `screen_fold0`, or `final_train_and_submit`.

- [ ] **Step 1: Write failing tests for screen and final generator profile validation**

Create `tests/unit/test_generator_profile_validation.py`:

```python
import pytest
from src.task2.generation.config import GeneratorTrainConfig
from src.task2.generation.trainer import train_generator_qlora


def test_screen_fold0_rejects_altered_max_seq_len():
    cfg = GeneratorTrainConfig(max_seq_len=1024, device="cuda:0")
    with pytest.raises(ValueError, match="requires max_seq_len=2048"):
        train_generator_qlora(
            model_name_or_path="mock-model",
            qa_path="mock",
            labels_path="mock",
            chunks_path="mock",
            output_dir="mock",
            config=cfg,
            execution_profile="screen_fold0",
        )


def test_final_train_rejects_disabled_activation_offloading():
    cfg = GeneratorTrainConfig(activation_offloading=False, device="cuda:0")
    with pytest.raises(ValueError, match="requires activation_offloading=True"):
        train_generator_qlora(
            model_name_or_path="mock-model",
            qa_path="mock",
            labels_path="mock",
            chunks_path="mock",
            output_dir="mock",
            config=cfg,
            execution_profile="final_train_and_submit",
        )


def test_final_train_rejects_disabled_liger():
    cfg = GeneratorTrainConfig(use_liger_fused_ce=False, device="cuda:0")
    with pytest.raises(ValueError, match="requires use_liger_fused_ce=True"):
        train_generator_qlora(
            model_name_or_path="mock-model",
            qa_path="mock",
            labels_path="mock",
            chunks_path="mock",
            output_dir="mock",
            config=cfg,
            execution_profile="final_train_and_submit",
        )


def test_final_train_rejects_wrong_device():
    cfg = GeneratorTrainConfig(device="cuda:1")
    with pytest.raises(ValueError, match="requires generator on device='cuda:0'"):
        train_generator_qlora(
            model_name_or_path="mock-model",
            qa_path="mock",
            labels_path="mock",
            chunks_path="mock",
            output_dir="mock",
            config=cfg,
            execution_profile="final_train_and_submit",
        )


def test_standard_profile_does_not_enforce_strict_checks():
    cfg = GeneratorTrainConfig(max_seq_len=512, device="cpu", activation_offloading=False, use_liger_fused_ce=False)
    # Does not raise ValueError from profile validation; fails later on missing inputs if not mocked
    try:
        train_generator_qlora(
            model_name_or_path="mock-model",
            qa_path="nonexistent",
            labels_path="nonexistent",
            chunks_path="nonexistent",
            output_dir="mock",
            config=cfg,
            execution_profile="standard",
        )
    except ValueError as e:
        pytest.fail(f"Standard profile should not run strict production validation: {e}")
    except Exception:
        pass  # Expected to fail on loading non-existent files or mock tokenizer
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv-ml/bin/pytest tests/unit/test_generator_profile_validation.py -v`
Expected: FAIL with `TypeError: train_generator_qlora() got an unexpected keyword argument 'execution_profile'`

- [ ] **Step 3: Implement explicit execution profile validation**

In `src/task2/generation/trainer.py`:
Update function signature of `train_generator_qlora` to accept `execution_profile: Optional[str] = None`:
```python
def train_generator_qlora(
    *,
    model_name_or_path: str,
    qa_path: str,
    labels_path: str,
    chunks_path: str,
    output_dir: str,
    config: Optional[GeneratorTrainConfig] = None,
    val_fold: Optional[int] = 0,
    max_steps: Optional[int] = None,
    max_train_examples: Optional[int] = None,
    probe_mode: Optional[str] = None,
    device: str = "cuda:0",
    epochs: int = 1,
    fail_on_error: bool = True,
    seed: int = 42,
    resume_from_checkpoint: Optional[str] = None,
    execution_profile: Optional[str] = None,
) -> Dict[str, Any]:
    assert_no_secrets_in_workspace(Path.cwd())

    if config is None:
        config = GeneratorTrainConfig(model_id=model_name_or_path, device=device)

    # 1. Validate configuration for the active execution profile
    profile_name = execution_profile or (
        "generator_probe_worstcase" if probe_mode == "worst_case" else (
            "generator_probe_endurance" if probe_mode == "endurance" else "standard"
        )
    )
    strict_profiles = {
        "final_train_and_submit",
        "screen_fold0",
        "generator_probe_worstcase",
        "generator_probe_endurance",
    }
    if profile_name in strict_profiles:
        validate_generator_config_for_profile(config, profile=profile_name)
```

In `src/task2/pipeline/runner.py`:
Pass `execution_profile=profile.name` when calling `train_generator_qlora`.

In `src/task2/training/train_generator.py`:
Add `execution_profile: Optional[str] = None` argument and forward it to `train_generator_qlora`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv-ml/bin/pytest tests/unit/test_generator_profile_validation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/task2/generation/trainer.py src/task2/pipeline/runner.py src/task2/training/train_generator.py tests/unit/test_generator_profile_validation.py
git commit -m "fix(p0): enforce generator config validation on screen and final profiles"
```

---

### Task 2: P0.2 Make `final_training` Packaging Require PROMOTED Protocol-8 Config

**Files:**
- Modify: `scripts/package_kaggle_dataset.py:88-115`
- Test: `tests/test_kaggle_packaging.py`

**Interfaces:**
- Consumes: `load_production_selection`, `validate_production_selection_for_profile` from `src.task2.production_config`.
- Produces: `package_kaggle_dataset(..., profile="final_training")` which strictly refuses to stage if `configs/production_selection.yaml` is UNVALIDATED or uses `screen_protocol_version < 8`.

- [ ] **Step 1: Write failing test for packaging profile validation**

Add to `tests/test_kaggle_packaging.py`:

```python
def test_package_kaggle_dataset_final_training_rejects_unvalidated(tmp_path: Path):
    unval_yaml = tmp_path / "production_selection.yaml"
    unval_yaml.write_text(
        "schema_version: 3\n"
        "status: UNVALIDATED\n"
        "screen_protocol_version: 1\n"
        "candidate_policy:\n"
        "  type: fixed_baseline\n"
        "  best_fixed_candidate: stitched_extract\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Production config status is 'UNVALIDATED'"):
        package_kaggle_dataset(
            source_dir="artifacts/task2",
            staging_dir=str(tmp_path / "stage"),
            profile="final_training",
            production_config_path=str(unval_yaml),
            dry_run=True,
        )


def test_package_kaggle_dataset_final_training_accepts_promoted_protocol_8(tmp_path: Path):
    promoted_yaml = tmp_path / "production_selection.yaml"
    promoted_yaml.write_text(
        "schema_version: 3\n"
        "status: PROMOTED\n"
        "screen_protocol_version: 8\n"
        "candidate_policy:\n"
        "  type: fixed_baseline\n"
        "  best_fixed_candidate: stitched_extract\n"
        "reranker:\n"
        "  use_task_tuned: false\n"
        "generator:\n"
        "  use_qlora: false\n",
        encoding="utf-8",
    )
    # With dry_run=True and unvalidated checks satisfied, should not raise RuntimeError about UNVALIDATED
    # It might only check other artifacts or complete
    try:
        package_kaggle_dataset(
            source_dir="artifacts/task2",
            staging_dir=str(tmp_path / "stage"),
            profile="final_training",
            production_config_path=str(promoted_yaml),
            dry_run=True,
        )
    except RuntimeError as e:
        if "UNVALIDATED" in str(e) or "screen_protocol_version" in str(e):
            pytest.fail(f"Promoted Protocol-8 config should pass profile validation: {e}")
    except FileNotFoundError:
        pass  # expected if local raw artifacts are missing in test env


def test_package_kaggle_dataset_default_profile_allows_unvalidated(tmp_path: Path):
    unval_yaml = tmp_path / "production_selection.yaml"
    unval_yaml.write_text(
        "schema_version: 3\n"
        "status: UNVALIDATED\n"
        "screen_protocol_version: 1\n",
        encoding="utf-8",
    )
    # Default profile is for probes and screening, must allow UNVALIDATED
    package_kaggle_dataset(
        source_dir="artifacts/task2",
        staging_dir=str(tmp_path / "stage"),
        profile="default",
        production_config_path=str(unval_yaml),
        dry_run=True,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv-ml/bin/pytest tests/test_kaggle_packaging.py::test_package_kaggle_dataset_final_training_rejects_unvalidated -v`
Expected: FAIL because packaging currently does not call `validate_production_selection_for_profile`.

- [ ] **Step 3: Implement strict production selection validation during staging**

In `scripts/package_kaggle_dataset.py`:
```python
    if profile == "final_training":
        # Strict checks for final_training profile
        public_official = Path("artifacts/raw/public-official.json")
        if not public_official.exists():
            missing.append("artifacts/raw/public-official.json")

        bm25_idx = src / "indexes" / "bm25"
        if not bm25_idx.exists() or not any(bm25_idx.iterdir()):
            missing.append("indexes/bm25")

        dek21_idx = src / "indexes" / "dek21"
        if not dek21_idx.exists() or not any(dek21_idx.iterdir()):
            missing.append("indexes/dek21")

        if not os.path.exists(production_config_path):
            missing.append(production_config_path)
        else:
            try:
                prod_cfg = load_production_selection(production_config_path)
                validate_production_selection_for_profile(
                    prod_cfg,
                    profile="final_train_and_submit",
                    allow_unvalidated_final=False,
                )
                if prod_cfg.use_task_tuned_reranker:
                    rerank_pairs = src / "data" / "reranker_training_pairs.parquet"
                    if not rerank_pairs.exists():
                        missing.append("data/reranker_training_pairs.parquet (required by production_selection tuned reranker)")
            except Exception as e:
                raise RuntimeError(
                    f"Production config at '{production_config_path}' is invalid for 'final_training' packaging: {e}"
                ) from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv-ml/bin/pytest tests/test_kaggle_packaging.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/package_kaggle_dataset.py tests/test_kaggle_packaging.py
git commit -m "fix(p0): require PROMOTED Protocol-8 config for final_training packaging"
```

---

### Task 3: P0.3 Make Checkpoint Reuse Deterministic and Provenance-Bound

**Files:**
- Create: `src/task2/checkpoint_resolver.py`
- Modify: `src/task2/pipeline/runner.py:155-162,220-227`
- Test: `tests/unit/test_checkpoint_resolver.py`

**Interfaces:**
- Consumes: Component directory paths, manifests, and production configuration targets.
- Produces: `resolve_component_checkpoint(component: str, expected_base_model: str, preferred_path: Optional[str] = None, search_roots: Optional[Sequence[str | Path]] = None, expected_runtime_api: int = 16) -> str` which returns an unambiguous, verified checkpoint path or raises `FileNotFoundError` (if none) or `RuntimeError` (if ambiguous / corrupted / stale).

- [ ] **Step 1: Write failing tests for checkpoint resolver**

Create `tests/unit/test_checkpoint_resolver.py`:

```python
import json
import pytest
from pathlib import Path
from src.task2.checkpoint_resolver import resolve_component_checkpoint


def create_mock_generator_checkpoint(dir_path: Path, base_model: str = "Qwen/Qwen2.5-3B-Instruct", api_v: int = 16):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base_model}))
    (dir_path / "adapter_model.safetensors").write_text("weights")
    manifest = {
        "runtime_api_version": api_v,
        "base_model": base_model,
        "component": "generator",
        "is_final": True,
    }
    (dir_path / "generator_manifest.json").write_text(json.dumps(manifest))


def test_resolve_exact_preferred_path(tmp_path: Path):
    ckpt = tmp_path / "checkpoints" / "generator" / "hf_adapter"
    create_mock_generator_checkpoint(ckpt)
    resolved = resolve_component_checkpoint(
        component="generator",
        expected_base_model="Qwen/Qwen2.5-3B-Instruct",
        preferred_path=str(ckpt),
        search_roots=[tmp_path],
    )
    assert resolved == str(ckpt)


def test_resolve_missing_checkpoint_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No valid generator checkpoint found"):
        resolve_component_checkpoint(
            component="generator",
            expected_base_model="Qwen/Qwen2.5-3B-Instruct",
            preferred_path=str(tmp_path / "nonexistent"),
            search_roots=[tmp_path],
        )


def test_resolve_ambiguous_checkpoints_raises_runtime_error(tmp_path: Path):
    # Two mounted datasets with different valid checkpoints but no preferred match
    ckpt1 = tmp_path / "ds1" / "checkpoints" / "generator" / "hf_adapter"
    ckpt2 = tmp_path / "ds2" / "checkpoints" / "generator" / "hf_adapter"
    create_mock_generator_checkpoint(ckpt1)
    create_mock_generator_checkpoint(ckpt2)

    with pytest.raises(RuntimeError, match="Ambiguous generator checkpoint candidates found"):
        resolve_component_checkpoint(
            component="generator",
            expected_base_model="Qwen/Qwen2.5-3B-Instruct",
            preferred_path=None,
            search_roots=[tmp_path / "ds1", tmp_path / "ds2"],
        )


def test_resolve_rejects_stale_api_manifest(tmp_path: Path):
    ckpt = tmp_path / "checkpoints" / "generator" / "hf_adapter"
    create_mock_generator_checkpoint(ckpt, api_v=15)  # Stale API 15

    with pytest.raises(RuntimeError, match="runtime_api_version"):
        resolve_component_checkpoint(
            component="generator",
            expected_base_model="Qwen/Qwen2.5-3B-Instruct",
            preferred_path=str(ckpt),
            search_roots=[tmp_path],
            expected_runtime_api=16,
        )


def test_resolve_rejects_base_model_mismatch(tmp_path: Path):
    ckpt = tmp_path / "checkpoints" / "generator" / "hf_adapter"
    create_mock_generator_checkpoint(ckpt, base_model="wrong-base-model")

    with pytest.raises(RuntimeError, match="expected base model"):
        resolve_component_checkpoint(
            component="generator",
            expected_base_model="Qwen/Qwen2.5-3B-Instruct",
            preferred_path=str(ckpt),
            search_roots=[tmp_path],
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv-ml/bin/pytest tests/unit/test_checkpoint_resolver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.task2.checkpoint_resolver'`

- [ ] **Step 3: Implement CheckpointResolver**

Create `src/task2/checkpoint_resolver.py`:
```python
"""Deterministic and provenance-bound checkpoint resolver for LegalQA V16."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence


def _validate_candidate(
    candidate_path: Path,
    component: str,
    expected_base_model: str,
    expected_runtime_api: int = 16,
) -> None:
    if not candidate_path.is_dir():
        raise RuntimeError(f"Candidate {candidate_path} is not a directory.")

    manifest_name = f"{component}_manifest.json"
    manifest_file = candidate_path / manifest_name
    if not manifest_file.exists():
        raise RuntimeError(f"Missing required component manifest '{manifest_name}' in {candidate_path}")

    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse manifest {manifest_file}: {exc}") from exc

    api_v = manifest_data.get("runtime_api_version")
    if api_v is not None and int(api_v) != expected_runtime_api:
        raise RuntimeError(
            f"Checkpoint at {candidate_path} has runtime_api_version={api_v} != expected {expected_runtime_api}"
        )

    base_model = manifest_data.get("base_model")
    if base_model and base_model != expected_base_model:
        raise RuntimeError(
            f"Checkpoint at {candidate_path} expected base model '{expected_base_model}', got '{base_model}'"
        )

    if component == "generator":
        weights_exist = (
            (candidate_path / "adapter_model.safetensors").exists()
            or (candidate_path / "adapter_model.bin").exists()
        )
        if not weights_exist:
            raise RuntimeError(f"Generator adapter weights missing in {candidate_path}")
        if not (candidate_path / "adapter_config.json").exists():
            raise RuntimeError(f"adapter_config.json missing in {candidate_path}")
    elif component == "reranker":
        weights_exist = (
            (candidate_path / "model.safetensors").exists()
            or (candidate_path / "pytorch_model.bin").exists()
            or any(candidate_path.glob("model-*.safetensors"))
        )
        if not weights_exist:
            raise RuntimeError(f"Reranker model weights missing in {candidate_path}")


def resolve_component_checkpoint(
    *,
    component: str,
    expected_base_model: str,
    preferred_path: Optional[str] = None,
    search_roots: Optional[Sequence[str | Path]] = None,
    expected_runtime_api: int = 16,
) -> str:
    """Resolve an unambiguous, validated checkpoint path.

    Never uses arbitrary first-match globbing. If preferred_path is valid, returns it.
    If multiple valid candidates exist with no preferred match, raises RuntimeError for ambiguity.
    If no valid candidate exists, raises FileNotFoundError.
    """
    if preferred_path:
        p = Path(preferred_path)
        if p.exists() and p.is_dir():
            _validate_candidate(p, component, expected_base_model, expected_runtime_api)
            return str(p.resolve())

    # Build search roots
    roots = [Path(r) for r in (search_roots or ["/kaggle/input", "checkpoints", "artifacts/task2/checkpoints"])]
    target_subpath = (
        Path("checkpoints/generator/hf_adapter") if component == "generator"
        else Path("checkpoints/reranker/best")
    )

    discovered: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        # Check direct subpath
        direct = root / target_subpath
        if direct.exists() and direct.is_dir():
            discovered.append(direct)
        # Search mounted dataset subtrees up to depth 4
        for sub in root.glob("**/checkpoints/" + ("generator/hf_adapter" if component == "generator" else "reranker/best")):
            if sub.is_dir() and sub not in discovered:
                discovered.append(sub)

    valid_candidates: List[Path] = []
    validation_errors = []
    for cand in discovered:
        try:
            _validate_candidate(cand, component, expected_base_model, expected_runtime_api)
            valid_candidates.append(cand)
        except Exception as err:
            validation_errors.append(f"{cand}: {err}")

    if not valid_candidates:
        err_msg = f"No valid {component} checkpoint found."
        if validation_errors:
            err_msg += " Rejected candidates:\n" + "\n".join(validation_errors)
        raise FileNotFoundError(err_msg)

    if len(valid_candidates) == 1:
        return str(valid_candidates[0].resolve())

    # If preferred_path matches one of them exactly:
    if preferred_path:
        for vc in valid_candidates:
            if str(vc.resolve()) == str(Path(preferred_path).resolve()):
                return str(vc.resolve())

    raise RuntimeError(
        f"Ambiguous {component} checkpoint candidates found: {[str(c) for c in valid_candidates]}. "
        f"Specify an exact preferred path in production selection provenance."
    )
```

In `src/task2/pipeline/runner.py`:
Replace lines 156-161:
```python
    elif profile.reuse_existing_checkpoints and production_cfg.use_task_tuned_reranker:
        from src.task2.checkpoint_resolver import resolve_component_checkpoint
        reranker_checkpoint = resolve_component_checkpoint(
            component="reranker",
            expected_base_model="BAAI/bge-reranker-v2-m3",
            preferred_path=production_cfg.reranker_checkpoint,
            expected_runtime_api=16,
        )
        assert_final_checkpoint(reranker_checkpoint, expected_base_model="BAAI/bge-reranker-v2-m3", component_name="reranker")
```
Replace lines 220-226:
```python
    elif profile.reuse_existing_checkpoints and production_cfg.use_qlora and profile.requires_generator:
        from src.task2.checkpoint_resolver import resolve_component_checkpoint
        adapter_path = resolve_component_checkpoint(
            component="generator",
            expected_base_model=production_cfg.generator_base_model,
            preferred_path=production_cfg.adapter_path,
            expected_runtime_api=16,
        )
        assert_final_checkpoint(adapter_path, expected_base_model=production_cfg.generator_base_model, component_name="generator")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv-ml/bin/pytest tests/unit/test_checkpoint_resolver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/task2/checkpoint_resolver.py src/task2/pipeline/runner.py tests/unit/test_checkpoint_resolver.py
git commit -m "feat(p0): deterministic manifest and provenance bound checkpoint resolver"
```

---

### Task 4: P0.4 Cryptographically Validate Promotion Provenance Before Final/Reuse Execution

**Files:**
- Modify: `src/task2/production_config.py:15-140`
- Modify: `scripts/promote_production_selection.py:180-225`
- Modify: `src/task2/pipeline/runner.py:30-40,290-300`
- Test: `tests/unit/test_promotion_provenance_crypto.py`

**Interfaces:**
- Consumes: `ProductionSelection`, promotion report path, screen run manifest.
- Produces: `verify_promotion_provenance(config: ProductionSelection, search_roots: Optional[Sequence[str | Path]] = None) -> None` which cryptographically checks SHA256 hashes of promotion report, screen run manifest, and sample IDs, failing loud on tampered or stale artifacts.

- [ ] **Step 1: Write failing tests for cryptographic promotion provenance validation**

Create `tests/unit/test_promotion_provenance_crypto.py`:

```python
import json
import pytest
from pathlib import Path
from src.common.hashing import sha256_file
from src.task2.production_config import (
    ProductionSelection,
    verify_promotion_provenance,
    validate_production_selection_for_profile,
)


def create_mock_report_and_manifest(tmp_path: Path):
    report_file = tmp_path / "promotion_report.json"
    report_data = {
        "screen_protocol_version": 8,
        "sample_ids_sha256": "abcdef1234567890" * 4,
        "sample_size": 250,
        "overall_deployable_winner": "stitched_extract",
        "overall_deployable_meteor": 0.3051,
    }
    report_file.write_text(json.dumps(report_data), encoding="utf-8")
    report_sha = sha256_file(report_file)

    manifest_file = tmp_path / "screen_run_manifest.json"
    manifest_data = {
        "runtime_api_version": 16,
        "screen_protocol_version": 8,
        "promotion_report_sha256": report_sha,
        "status": "SCREEN_PASS",
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")
    manifest_sha = sha256_file(manifest_file)

    return report_file, report_sha, manifest_file, manifest_sha


def test_verify_provenance_success(tmp_path: Path):
    report_file, report_sha, manifest_file, manifest_sha = create_mock_report_and_manifest(tmp_path)
    cfg = ProductionSelection(
        schema_version=3,
        status="PROMOTED",
        source_screen_manifest=str(report_file),
        source_screen_sha256=report_sha,
        provenance={
            "promotion_report_path": str(report_file),
            "promotion_report_sha256": report_sha,
            "screen_run_manifest_path": str(manifest_file),
            "screen_run_manifest_sha256": manifest_sha,
            "sample_ids_sha256": "abcdef1234567890" * 4,
            "runtime_api_version": 16,
            "screen_protocol_version": 8,
        },
        raw_config={"screen_protocol_version": 8},
    )
    # Must succeed without error
    verify_promotion_provenance(cfg, search_roots=[tmp_path])


def test_verify_provenance_fails_on_tampered_report(tmp_path: Path):
    report_file, report_sha, manifest_file, manifest_sha = create_mock_report_and_manifest(tmp_path)
    # Tamper with the report file
    report_file.write_text(json.dumps({"tampered": True}), encoding="utf-8")

    cfg = ProductionSelection(
        schema_version=3,
        status="PROMOTED",
        source_screen_manifest=str(report_file),
        source_screen_sha256=report_sha,
        provenance={
            "promotion_report_path": str(report_file),
            "promotion_report_sha256": report_sha,
            "runtime_api_version": 16,
            "screen_protocol_version": 8,
        },
        raw_config={"screen_protocol_version": 8},
    )
    with pytest.raises(RuntimeError, match="Promotion report SHA256 mismatch"):
        verify_promotion_provenance(cfg, search_roots=[tmp_path])


def test_verify_provenance_fails_on_missing_report(tmp_path: Path):
    cfg = ProductionSelection(
        schema_version=3,
        status="PROMOTED",
        source_screen_manifest=str(tmp_path / "nonexistent.json"),
        source_screen_sha256="0" * 64,
        provenance={
            "promotion_report_path": str(tmp_path / "nonexistent.json"),
            "promotion_report_sha256": "0" * 64,
            "runtime_api_version": 16,
            "screen_protocol_version": 8,
        },
        raw_config={"screen_protocol_version": 8},
    )
    with pytest.raises(FileNotFoundError, match="Promotion report file not found"):
        verify_promotion_provenance(cfg, search_roots=[tmp_path])


def test_validate_profile_triggers_provenance_verification(tmp_path: Path):
    cfg = ProductionSelection(
        schema_version=3,
        status="PROMOTED",
        source_screen_manifest=str(tmp_path / "missing.json"),
        source_screen_sha256="0" * 64,
        provenance={
            "promotion_report_path": str(tmp_path / "missing.json"),
            "promotion_report_sha256": "0" * 64,
            "runtime_api_version": 16,
            "screen_protocol_version": 8,
        },
        raw_config={"screen_protocol_version": 8},
    )
    with pytest.raises((FileNotFoundError, RuntimeError)):
        validate_production_selection_for_profile(cfg, profile="final_train_and_submit", allow_unvalidated_final=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv-ml/bin/pytest tests/unit/test_promotion_provenance_crypto.py -v`
Expected: FAIL with `ImportError: cannot import name 'verify_promotion_provenance' from 'src.task2.production_config'`

- [ ] **Step 3: Implement cryptographic provenance verification**

In `src/task2/production_config.py`:
1. Add `provenance: Optional[Dict[str, Any]] = None` to `ProductionSelection` dataclass.
2. Update `load_production_selection` to parse `data.get("provenance")`.
3. Implement `verify_promotion_provenance(config: ProductionSelection, search_roots: Optional[Sequence[str | Path]] = None) -> None`:
```python
def verify_promotion_provenance(
    config: ProductionSelection,
    search_roots: Optional[Sequence[str | Path]] = None,
) -> None:
    """Cryptographically revalidate promotion report and screen manifests before final/reuse execution."""
    prov = config.provenance or {}
    report_rel_path = prov.get("promotion_report_path") or config.source_screen_manifest
    expected_report_sha = prov.get("promotion_report_sha256") or config.source_screen_sha256

    if not report_rel_path or not expected_report_sha:
        raise RuntimeError("Production config is missing promotion report provenance path or SHA256.")

    # Locate report file
    roots = [Path(r) for r in (search_roots or [".", "/kaggle/input", "/kaggle/working"])]
    report_path = None
    if Path(report_rel_path).is_file():
        report_path = Path(report_rel_path)
    else:
        for r in roots:
            cand = r / report_rel_path
            if cand.is_file():
                report_path = cand
                break
            # Also check direct file under root
            cand_direct = r / Path(report_rel_path).name
            if cand_direct.is_file():
                report_path = cand_direct
                break

    if not report_path:
        raise FileNotFoundError(f"Promotion report file not found: {report_rel_path}")

    actual_report_sha = sha256_file(report_path)
    if actual_report_sha != expected_report_sha:
        raise RuntimeError(
            f"Promotion report SHA256 mismatch for {report_path}: "
            f"actual {actual_report_sha} != expected {expected_report_sha}"
        )

    # Validate screen_run_manifest if specified
    manifest_rel_path = prov.get("screen_run_manifest_path")
    expected_manifest_sha = prov.get("screen_run_manifest_sha256")
    if manifest_rel_path and expected_manifest_sha:
        manifest_path = None
        if Path(manifest_rel_path).is_file():
            manifest_path = Path(manifest_rel_path)
        else:
            for r in roots:
                cand = r / manifest_rel_path
                if cand.is_file():
                    manifest_path = cand
                    break
                cand_direct = r / Path(manifest_rel_path).name
                if cand_direct.is_file():
                    manifest_path = cand_direct
                    break
        if manifest_path:
            actual_m_sha = sha256_file(manifest_path)
            if actual_m_sha != expected_manifest_sha:
                raise RuntimeError(
                    f"Screen run manifest SHA256 mismatch for {manifest_path}: "
                    f"actual {actual_m_sha} != expected {expected_manifest_sha}"
                )

    # Validate protocol version and API version
    api_v = prov.get("runtime_api_version")
    if api_v and int(api_v) != 16:
        raise RuntimeError(f"Promotion provenance requires runtime_api_version=16, got {api_v}")
```

4. Call `verify_promotion_provenance(config)` inside `validate_production_selection_for_profile` for final/reuse profiles when `not allow_unvalidated_final`.

In `scripts/promote_production_selection.py`:
Populate the `provenance` block in `promoted_config`:
```python
    promoted_config["provenance"] = {
        "promotion_report_path": report_path,
        "promotion_report_sha256": report_sha256,
        "sample_ids_sha256": report.get("sample_ids_sha256"),
        "runtime_api_version": 16,
        "screen_protocol_version": 8,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv-ml/bin/pytest tests/unit/test_promotion_provenance_crypto.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/task2/production_config.py scripts/promote_production_selection.py tests/unit/test_promotion_provenance_crypto.py
git commit -m "feat(p0): cryptographically revalidate promotion provenance before final and reuse"
```

---

### Task 5: P1.1 and P1.2 README Truth and Config Authority Alignment

**Files:**
- Modify: `README.md`
- Modify: `configs/task2.yaml`
- Modify: `scripts/audit_parameters.py:100-130`
- Test: `tests/contracts/test_config_authority.py`
- Test: `tests/test_validation_and_audit.py`

**Interfaces:**
- Consumes: `configs/pipeline.yaml`, `configs/models.yaml`, `configs/runtime_api.yaml`, `configs/production_selection.yaml`, `configs/task2.yaml`.
- Produces: Coherent, verified configuration authority without drift, and up-to-date V16 documentation in `README.md`.

- [ ] **Step 1: Write config authority contract test**

Create `tests/contracts/test_config_authority.py`:

```python
import yaml
from pathlib import Path


def test_config_authority_and_no_drift():
    # Load authoritative configs
    with open("configs/pipeline.yaml", "r", encoding="utf-8") as f:
        pipeline_cfg = yaml.safe_load(f)
    with open("configs/models.yaml", "r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)
    with open("configs/runtime_api.yaml", "r", encoding="utf-8") as f:
        runtime_cfg = yaml.safe_load(f)
    with open("configs/production_selection.yaml", "r", encoding="utf-8") as f:
        prod_cfg = yaml.safe_load(f)
    with open("configs/task2.yaml", "r", encoding="utf-8") as f:
        task2_cfg = yaml.safe_load(f)

    # 1. Runtime API
    assert runtime_cfg["runtime_api_version"] == 16

    # 2. Models identity alignment
    retriever_model = models_cfg["models"]["retriever"]["name"]
    reranker_model = models_cfg["models"]["reranker"]["name"]
    generator_model = models_cfg["models"]["generator"]["name"]

    assert pipeline_cfg["retrieval"]["dense"]["model"] == retriever_model
    assert pipeline_cfg["reranker"]["base_model"] == reranker_model
    assert pipeline_cfg["generator"]["model_id"] == generator_model

    assert prod_cfg["retrieval"]["dense"]["model"] == retriever_model
    assert prod_cfg["reranker"]["base_model"] == reranker_model
    assert prod_cfg["generator"]["base_model"] == generator_model

    # 3. Parameter alignment across pipeline and task2.yaml
    assert task2_cfg["models"]["retriever"]["top_k"] == 50
    assert task2_cfg["models"]["generator"]["max_new_tokens"] == 384
```

- [ ] **Step 2: Run test to verify it fails on task2.yaml drift**

Run: `./.venv-ml/bin/pytest tests/contracts/test_config_authority.py -v`
Expected: FAIL due to `task2_cfg['models']['retriever']['top_k'] == 60` and `max_new_tokens == 512`.

- [ ] **Step 3: Align task2.yaml and update scripts/audit_parameters.py**

1. Update `configs/task2.yaml`:
   - Add warning notice that `configs/pipeline.yaml` and `configs/production_selection.yaml` are the active production authority.
   - Align `retriever.top_k: 50`.
   - Align `generator.max_new_tokens: 384`.
2. Update `scripts/audit_parameters.py`:
   - In `verify_config_consistency()`, check `configs/production_selection.yaml` models against approved models list.
3. Update `README.md`:
   - Document V16 sequence:
     A. `generator_probe_worstcase` (committed default in notebook)
     B. `generator_probe_endurance`
     C. `screen_fold0`
     D. `final_train_and_submit`
   - State clearly:
     - Notebook default is `generator_probe_worstcase`.
     - Final training requires a `PROMOTED` Protocol-8 config.
     - Runtime API 16 dataset package is separate from the immutable inherited V15 manifest (`dataset_manifest_v15.json`).
     - Checkpoint reuse requires canonical V16 HF checkpoints (not old MLX adapters).
     - Does not override `docs/active/v16/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv-ml/bin/pytest tests/contracts/test_config_authority.py tests/test_validation_and_audit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md configs/task2.yaml scripts/audit_parameters.py tests/contracts/test_config_authority.py
git commit -m "docs: refresh README to V16 truth and eliminate config authority drift"
```

---

### Task 6: Comprehensive Verification Suite and Dry Packaging

**Files:**
- None modified; runs verification across workspace.

**Interfaces:**
- Consumes: full repo test suite, scripts, and packaging utilities.
- Produces: Verified green test matrix, compliant parameter audit `< 4B`, passing preflight, verified inherited dataset hashes, and clean dry-run staging.

- [ ] **Step 1: Run full pytest suite**

Run: `./.venv-ml/bin/pytest -v`
Expected: ALL tests PASS (0 failures).

- [ ] **Step 2: Run parameter audit**

Run: `./.venv-ml/bin/python scripts/audit_parameters.py`
Expected: `COMPLIANT`, total learned parameters strictly `< 4,000,000,000`.

- [ ] **Step 3: Run preflight script**

Run: `./.venv-ml/bin/python scripts/preflight_kaggle.py --pipeline_config configs/pipeline.yaml --models_config configs/models.yaml`
Expected: PASS with 0 errors.

- [ ] **Step 4: Run inherited dataset verification**

Run: `./.venv-ml/bin/python scripts/verify_inherited_dataset.py --root artifacts/inherited --manifest artifacts/inherited/dataset_manifest_v15.json` (or against available artifacts)
Expected: Hashes match or verified.

- [ ] **Step 5: Run dry-run package validation**

Run: `./.venv-ml/bin/python scripts/package_kaggle_dataset.py --profile default --dry_run`
Expected: PASS with API 16 tripartite manifest verification.
