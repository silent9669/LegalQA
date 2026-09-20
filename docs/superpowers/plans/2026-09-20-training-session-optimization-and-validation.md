# LegalQA Modal A100 Production Training Optimization and Provenance Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the production-grade training pipeline configuration, provenance bundle packaging, and execution validation for Modal A100, ensuring minimal training time, maximum retrieval and generation accuracy, zero empty answers, and seamless Hugging Face release.

**Architecture:** Build upon the unified 2-gate DAG (`kaggle_t4x2` -> `a100_micro_probe` -> `full`). Ensure `build_production_run_bundle` supports the `modal_a100` runtime profile, packages the official submission and provenance artifacts, and dynamically structures the model card. Validate all contracts with zero regressions against the 248-test suite.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, Transformers 5.0.0, PEFT 0.19.1, TRL 1.12.0, Liger-Kernel 0.8.2, Modal, Hugging Face Hub.

**Spec:** `docs/ARCHITECTURE.md`, `configs/task2/algorithm.yaml`, `configs/task2/runtime/modal_a100.yaml`.

## Global Constraints

- Never mutate protected algorithm fields (`max_seq_len: 2048`, `quantization: 4bit_nf4`, `effective_batch_size: 8`).
- Effective batch size invariant: `per_device_train_batch_size * gradient_accumulation_steps == 8`.
- Generator training must target `cuda:0` with `trainer_n_gpu: 1`.
- Liger fused-linear cross-entropy must remain active with `loss_type: nll` (never `chunked_nll`).
- Dense index must satisfy the identical-text pair consistency gate (`aligned == True`, `pass_rate >= 0.95`).
- Pre-push check (`./test.sh`) must remain 100% PASS with 0 parameter budget violations (< 4.0B) and 0 secrets leaked.

---

### Task 1: Enhance `build_production_run_bundle` to Support `runtime_profile` Parameter and Submission Artifacts

**Files:**
- Modify: `src/task2/provenance/run_bundle.py:27-108,111-200,290-315`
- Test: `tests/provenance/test_run_bundle_contract.py`

**Interfaces:**
- Consumes: `CandidateManifest`, `GateReport`, `ResolvedTask2Config`
- Produces: `build_production_run_bundle(..., runtime_profile="colab_a100", submission_path=None, submission_provenance_path=None)`

- [ ] **Step 1: Write the failing test for modal_a100 runtime profile and submission inclusion**

```python
def test_build_production_run_bundle_modal_runtime_and_submission(tmp_path):
    from src.task2.provenance.run_bundle import build_production_run_bundle, verify_run_bundle
    candidate = sample_candidate()
    run_id = f"run_modal_sub_{candidate.candidate_id}"
    output_bundle_dir = tmp_path / "runs" / run_id

    adapter_src = tmp_path / "src_adapter"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text(json.dumps({"lora_r": 16}))
    (adapter_src / "adapter_model.safetensors").write_bytes(b"weights")

    k_rep = sample_gate_report("kaggle_t4x2", candidate.candidate_id, candidate.git_commit_sha)
    a_rep = sample_gate_report("a100_micro_probe", candidate.candidate_id, candidate.git_commit_sha, k_rep.compute_sha256())

    k_path = tmp_path / "k.json"
    a_path = tmp_path / "a.json"
    k_rep.save_json(k_path)
    a_rep.save_json(a_path)

    log_file = tmp_path / "train.log"
    log_file.write_text("Step 100 loss: 0.5\n")

    sub_file = tmp_path / "submission.json"
    sub_file.write_text(json.dumps({"1": {"answer": "A1"}}))
    prov_file = tmp_path / "submission_provenance.json"
    prov_file.write_text(json.dumps({"1": {"source": "generated"}}))

    manifest = build_production_run_bundle(
        run_id=run_id,
        candidate=candidate,
        adapter_source_dir=adapter_src,
        kaggle_report_path=k_path,
        colab_t4_report_path=None,
        a100_micro_probe_report_path=a_path,
        train_log_path=log_file,
        output_dir=output_bundle_dir,
        metrics={"meteor": 0.52},
        optimizer_steps=100,
        training_sample_count=800,
        runtime_profile="modal_a100",
        submission_path=sub_file,
        submission_provenance_path=prov_file,
    )

    assert manifest["run_id"] == run_id
    assert "submission" in manifest
    assert (output_bundle_dir / "submission.json").exists()
    assert (output_bundle_dir / "submission_provenance.json").exists()
    assert verify_run_bundle(output_bundle_dir) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv311/bin/pytest tests/provenance/test_run_bundle_contract.py::test_build_production_run_bundle_modal_runtime_and_submission -v`
Expected: FAIL with TypeError or unexpected keyword argument `runtime_profile`

- [ ] **Step 3: Implement `runtime_profile`, `submission_path`, and `has_colab_t4` in `run_bundle.py`**

Support `runtime_profile: str = "colab_a100"`, `submission_path: Optional[Union[Path, str]] = None`, and `submission_provenance_path: Optional[Union[Path, str]] = None` in `build_production_run_bundle`. When `submission_path` is passed, copy to `out_root / "submission.json"` before checksums calculation.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv311/bin/pytest tests/provenance/test_run_bundle_contract.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/task2/provenance/run_bundle.py tests/provenance/test_run_bundle_contract.py
git commit -m "feat(provenance): support runtime_profile and submission artifact binding in run bundle"
```

---

### Task 2: Update `scripts/modal_app.py` Production Release Packaging and Verification

**Files:**
- Modify: `scripts/modal_app.py:530-615`
- Test: `tests/launchers/test_modal_app.py`

**Interfaces:**
- Consumes: `build_production_run_bundle`, `run_output_dir`, `outputs`
- Produces: Complete run bundle with `submission.json`, `submission_provenance.json`, and `runtime_profile="modal_a100"`

- [ ] **Step 1: Write test for remote run bundle packaging helper in `test_modal_app.py`**

Verify that `scripts/modal_app.py` passes `runtime_profile="modal_a100"`, handles submission artifact copying, and validates parent reports defensively.

- [ ] **Step 2: Run test to verify failure**

Run: `./.venv311/bin/pytest tests/launchers/test_modal_app.py -v`

- [ ] **Step 3: Update `scripts/modal_app.py`**

Wire `runtime_profile="modal_a100"`, copy `sub_json` and `prov_json`, and pass `dataset_manifest_path=data_dir / "dataset_manifest.json"`.

- [ ] **Step 4: Run test to verify passes**

Run: `./.venv311/bin/pytest tests/launchers/test_modal_app.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/modal_app.py tests/launchers/test_modal_app.py
git commit -m "feat(modal): wire modal_a100 runtime and submission artifacts into HF bundle"
```

---

### Task 3: Full End-to-End Test Suite Validation (`./test.sh`)

**Files:**
- Test: All 248+ tests in `tests/`
- Tool: `./test.sh`

- [ ] **Step 1: Run full `./test.sh`**

Run: `./test.sh`
Expected: ALL 248+ unit, integration, contract, and dataset tests PASS in < 50s.
Parameter budget compliant (< 4.0B).
Zero secrets detected.
Working tree clean.
