# LegalQA V13 — Real Kaggle Screen-Before-Final Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the next user-run Kaggle Dual-T4 execution a trustworthy Protocol-8 production screen, remove the unsafe UNVALIDATED final override, and generate exact promotion artifacts for the later max-score final run.

**Architecture:** Keep Runtime API 13 and the verified Kaggle dataset bound to `757620ce8ccd41753ff217d4fe6593627a196899`. No more Colab and no Stack A redesign. Restore the notebook promotion gate, run `screen_fold0` on Kaggle T4x2, convert the measured `promotion_report.json` with the existing deterministic promoter, and save a small handoff archive. The user sends that handoff back before the final configuration is frozen.

## Global constraints

- Keep Runtime API **13**; do not create API14 for this control-flow/config fix.
- Keep the current API13 dataset unchanged for the next screening run.
- Do not regenerate BM25, DEk21, QA artifacts, reranker pairs, or training data.
- Do not run another Colab smoke.
- Coding agent must never push/run Kaggle; only the user runs Kaggle T4x2.
- Keep `ALLOW_SINGLE_GPU_SMOKE = False` and `HF_DEACTIVATE_ASYNC_LOAD=1`.
- Keep the V13 single-GPU Trainer policy for QLoRA.
- Never print/persist/commit `HF_TOKEN`.
- Do not alter model identities, LoRA rank, max sequence length, batch/accumulation, METEOR, or parameter budget.
- `ALLOW_UNVALIDATED_FINAL=True` is forbidden for a score-max production run.
- Final component choices must come only from a valid Protocol-8 promotion report.
- The next Kaggle run is `screen_fold0`, not `final_train_and_submit`.

---

## 0. Audited evidence and blocker

Verified generator smoke:

```text
git_sha = 757620ce8ccd41753ff217d4fe6593627a196899
GPU = Tesla T4
mode = full
status = PASS
generator_steps = 30
peak_vram_mb = 4304.69
adapter_reload = pass
```

Drive package:

```text
runtime_api_version = 13
git_sha = 757620ce8ccd41753ff217d4fe6593627a196899
profile = final_training
BM25 complete
DEk21 complete
public-official.json present
reranker_training_pairs.parquet present
4-bit generator inference/reload fix packaged
```

Latest repository HEAD audited:

```text
af52e4d9ff0b805431de6fe5321df0e3421efcd6
```

Latest CI is green.

### Remaining blocker: production selection semantics

The notebook currently commits:

```python
ALLOW_UNVALIDATED_FINAL = True
EXECUTION_PROFILE = "final_train_and_submit"
```

while `configs/production_selection.yaml` is still:

```yaml
status: "UNVALIDATED"
reranker:
  use_task_tuned: true
generator:
  use_qlora: true
candidate_policy:
  type: "fixed_baseline"
  best_fixed_candidate: "stitched_extract"
```

`stitched_extract` is not generator-dependent. Therefore:

```text
REQUIRES_GENERATOR = False
RUN_GENERATOR_TRAINING = use_qlora and REQUIRES_GENERATOR = False
```

Final inference also bypasses Qwen. The current notebook may create a valid submission, but it is not a trustworthy max-score final run because an UNVALIDATED template silently decides the system and discards the verified QLoRA component.

The repository already has the correct mechanisms:

- `run_screen_matrix()` — Protocol-8 R0G0 → R1G0 → R_SELECTED_G1 measurement.
- `promote_production_selection()` — validates that report and creates a `PROMOTED` config.

Use them. Do not invent another selector.

---

## Task 1 — Restore the strict next-run profile

**Modify:** `kaggle_kernel/legalqa_gpu_pipeline.ipynb`, `tests/test_v9_notebook_contract.py`  
**Create:** `tests/test_real_kaggle_screen_gate.py`

Cell 1 must become:

```python
ALLOW_SINGLE_GPU_SMOKE = False
ALLOW_UNVALIDATED_FINAL = False
EXECUTION_PROFILE = "screen_fold0"
```

Keep:

```python
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
REQUIRED_RUNTIME_API_VERSION = 13
```

Regression tests must assert all four values and must assert:

```python
assert "ALLOW_UNVALIDATED_FINAL = True" not in notebook_source()
```

Do not change Runtime API.

---

## Task 2 — Auto-create the measured PROMOTED config after screen

Immediately after the existing successful `run_screen_matrix(...)` in Cell 10:

```python
from scripts.promote_production_selection import promote_production_selection
from src.task2.production_config import (
    load_production_selection,
    validate_production_selection_for_profile,
)

promotion_report_path = "/kaggle/working/promotion_report.json"
promoted_config_path = "/kaggle/working/promoted_production_selection.yaml"

if not os.path.isfile(promotion_report_path):
    raise FileNotFoundError(
        "SCREEN_PROMOTION_ERROR: missing /kaggle/working/promotion_report.json"
    )

promote_production_selection(
    report_path=promotion_report_path,
    config_path=resolved_prod_cfg_path,
    output_path=promoted_config_path,
)

promoted_cfg = load_production_selection(promoted_config_path)
if promoted_cfg.status != "PROMOTED":
    raise RuntimeError(
        f"SCREEN_PROMOTION_ERROR: expected PROMOTED, got {promoted_cfg.status!r}"
    )

validate_production_selection_for_profile(
    promoted_cfg,
    "final_train_and_submit",
    allow_unvalidated_final=False,
)

print("Protocol-8 PROMOTED configuration verified.")
print(f"Use task-tuned reranker: {promoted_cfg.use_task_tuned_reranker}")
print(f"Use QLoRA:              {promoted_cfg.use_qlora}")
print(f"Best fixed candidate:   {promoted_cfg.best_fixed_candidate}")
```

Do **not** automatically start full all-data training after this screen. Stop at a clean selection handoff.

---

## Task 3 — Add a report sanity gate

Before promotion, load the report and require:

```python
with open(promotion_report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

assert report.get("screen_protocol_version") == 8
required = {"R0G0", "R1G0", "R_SELECTED_G1"}
actual = set(report.get("evaluated_systems", {}))
if not required.issubset(actual):
    raise RuntimeError(f"SCREEN_PROMOTION_ERROR: incomplete systems: {sorted(actual)}")
if not report.get("overall_deployable_winner"):
    raise RuntimeError("SCREEN_PROMOTION_ERROR: missing overall_deployable_winner")
if report.get("overall_deployable_meteor") is None:
    raise RuntimeError("SCREEN_PROMOTION_ERROR: missing overall_deployable_meteor")
```

Never hand-author the winner. The existing promoter is authoritative.

---

## Task 4 — Save a small screening handoff

Create:

```text
/kaggle/working/screen_handoff/
├── promotion_report.json
├── promoted_production_selection.yaml
└── screen_run_manifest.json
```

`screen_run_manifest.json` must contain:

```json
{
  "runtime_api_version": 13,
  "dataset_runtime_git_sha": "<RUNTIME_PROVENANCE git_sha>",
  "execution_profile": "screen_fold0",
  "screen_protocol_version": 8,
  "promotion_report_sha256": "...",
  "promoted_config_sha256": "...",
  "status": "SCREEN_PASS"
}
```

Zip it using Python `zipfile` to:

```text
/kaggle/working/screen_handoff.zip
```

Do not include checkpoints, embeddings, BM25 files, model weights, secrets, or the full dataset.

---

## Task 5 — Preserve real Protocol-8 screen semantics

Tests must ensure `screen_fold0` still means:

```text
RUN_RERANKER_TRAINING = True
RUN_GENERATOR_TRAINING = True
RUN_DEV_EVALUATION = True
RUN_PUBLIC_INFERENCE = False
REUSE_EXISTING_CHECKPOINTS = False
TRAIN_VAL_FOLD = 0
MAX_RERANKER_STEPS = None
MAX_GENERATOR_STEPS = None
DEV_EVAL_SIZE = 250
```

`run_screen_matrix()` must receive the real base/tuned reranker, base generator/QLoRA adapter, BM25/DEk21 indexes, `cuda:1` retrieval and `cuda:0` generator.

Do not reduce the 250-query screen: this is the evidence used to choose the final competition system.

---

## Task 6 — Preserve strict final behavior for the later run

Add a test proving an UNVALIDATED config cannot enter `final_train_and_submit` with the strict flag:

```python
cfg = load_production_selection("configs/production_selection.yaml")
if cfg.status == "UNVALIDATED":
    with pytest.raises(RuntimeError):
        validate_production_selection_for_profile(
            cfg,
            "final_train_and_submit",
            allow_unvalidated_final=False,
        )
```

Never make a score-max final run work by changing the override to `True`.

---

## Task 7 — Static verification and CI

Run:

```bash
pytest tests/test_real_kaggle_screen_gate.py -v
pytest tests/test_v9_notebook_contract.py -v
pytest tests/test_v10_runtime_release_binding.py -v
pytest tests/test_v13_single_gpu_qlora.py -v
pytest tests/ -v
```

Commit and push GitHub; wait for all Actions jobs to pass.

For this screening-only notebook change:

```text
Runtime API = 13
Dataset = unchanged
Dataset republish = NO
Colab = NO
Kaggle agent execution = NO
```

---

## Task 8 — User-only real Kaggle screening run

When CI is green, return:

```text
READY FOR USER MANUAL KAGGLE SCREEN
```

User manually configures:

```text
Notebook: latest screen_fold0 notebook
Dataset: existing API13 package SHA 757620ce8ccd41753ff217d4fe6593627a196899
Model: qwen-lm/qwen2.5/transformers/3b-instruct/1
Accelerator: T4 x2
Internet: On
HF_TOKEN: enabled
```

Required early log:

```text
COMMITTED EXECUTION PROFILE: screen_fold0
CUDA GPUs Detected: 2
Runtime API 13 verified
```

Required QLoRA policy:

```text
QLoRA Trainer GPU Policy:
target=cuda:0 | visible_cuda=2 | trainer_n_gpu=1
```

Required end:

```text
Protocol-8 screen complete
promotion_report.json created
promoted_production_selection.yaml verified
screen_handoff.zip created
SCREEN_PASS
```

This is a real Kaggle training/evaluation run, not another smoke test.

---

## Task 9 — Stop and audit before final all-data training

User sends back `screen_handoff.zip`.

Only then:

1. verify report/config hashes and provenance;
2. compare measured R0G0 / R1G0 / R_SELECTED_G1;
3. verify selected reranker, generator, winner candidate and METEOR;
4. commit the exact PROMOTED config;
5. switch notebook to `final_train_and_submit`;
6. keep `ALLOW_UNVALIDATED_FINAL=False`;
7. package only the final runtime/config changes required;
8. run CI;
9. give the user the final manual Kaggle run instructions.

Do not pre-select QLoRA, tuned reranker, or `stitched_extract` before the measured report exists.

---

## Verdict encoded by this plan

```text
V13 runtime/package              READY
Generator T4 30-step smoke       PASS
Adapter strict reload            PASS
Latest GitHub CI                 GREEN
BM25 / DEk21 / public data       PRESENT
Protocol-8 production selection  NOT YET MEASURED
Current config                    UNVALIDATED
Current final override            UNSAFE FOR MAX-SCORE RUN
```

Therefore:

```text
DO NOT run current final_train_and_submit for the score-max attempt.
DO NOT spend time on another Colab smoke.
DO run one real Kaggle Dual-T4 screen_fold0 next.
```

## Required agent completion report

```text
final HEAD SHA
working tree clean
Runtime API = 13
dataset repackaged = NO
notebook profile = screen_fold0
ALLOW_UNVALIDATED_FINAL = False
Protocol-8 auto-promotion hook = PASS
screen_handoff.zip hook = PASS
focused tests result
full pytest result
GitHub Actions run ID/result
Kaggle notebook pushed/run by agent = NO
verdict = READY FOR USER MANUAL KAGGLE SCREEN | BLOCKED
```
