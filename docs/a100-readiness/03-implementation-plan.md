# 03 — Implementation Plan

> **For agentic workers:** implement task-by-task. Each task ends in a command that must
> pass before the next begins. Steps use `- [ ]` for tracking.

**Goal:** make one Modal A100 run produce the **highest-scoring submission this
architecture allows**, with a provenance record that is true.

> **Priority directive (2026-09-21, from the project owner):** *maximise score; training
> time is not the constraint.* This reorders the plan. Speed fixes are kept only where they
> are free or where they buy headroom to spend on quality — a faster decode is what makes a
> 1536-token ceiling and multi-epoch training affordable inside one container. Where speed
> and score conflict, **score wins**.
>
> **Colab is out of the workflow.** The `colab_t4` gate is removed from the operational
> path (Task 12). The chain is `kaggle_t4x2 → a100_micro_probe → full`, which the existing
> DAG already permits.

**Architecture:** unchanged in shape — QLoRA SFT on Qwen2.5-3B, hybrid retrieval, candidate
assembly. What changes is (a) inference executes in bf16 instead of 4-bit, (b) generation
is resume-safe and cached, (c) the assembly strategy is chosen by offline measurement
rather than by a hard-coded constant, (d) the training set no longer contains
evidence-free examples.

**Tech stack:** Python 3.11, PyTorch + CUDA 12.1, transformers/TRL/PEFT/bitsandbytes,
Liger-Kernel, Modal, NLTK + rouge_score (the official scorer).

**Spec:** [`02-scoring-model.md`](02-scoring-model.md) — the plan argues from it.
**Evidence:** [`00-evidence-and-measurements.md`](00-evidence-and-measurements.md) — every number cited below.

---

## Global constraints

Copy these verbatim into any sub-task; they bind everywhere.

1. **Parameter budget:** total learned parameters `< 4,000,000,000`. Current
   3,788,891,136 (margin 211 M). `scripts/audit_parameters.py` enforces it at
   `runner.py:298-303`. **Do not change LoRA rank or add trainable modules.**
2. **Model pins are immutable 40-hex revisions.** Generator
   `Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1`; reranker
   `BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`; dense
   `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2@99a2963b2f51fa7a570a3e7f550d7993b9de90a8`.
3. **Any change to `configs/task2/**` or to tracked code re-mints `candidate_id`**
   (`src/task2/provenance/candidate.py:92-109`) and invalidates every gate report. Batch
   all changes, mint once (Task 9).
4. **Score-affecting settings belong in hashed config**, not in Python constants. The
   inference dtype (B-03) and `max_new_tokens` (B-02) both change output and must move
   into `configs/task2/runtime/modal_a100.yaml`.
5. **`./test.sh` must stay green** — 252/252 at the time of this audit. Never mark a task
   done on a suite you did not run.
6. **Seed is 42** everywhere; determinism is part of the contract.
7. **Never stitch more than one article** ([M-5](00-evidence-and-measurements.md#m-5--multi-article-stitching-under-imperfect-retrieval-n--150)); citation budget is 4000 chars ([M-4](00-evidence-and-measurements.md#m-4--assembly-strategy-oracle-retrieval-n--150)).

---

## File structure

| path | responsibility | task |
|---|---|---|
| `configs/task2/runtime/modal_a100.yaml` | runtime knobs: dtype, batch, token cap | 1, 2 |
| `src/task2/config/schema.py`, `config/loader.py` | carry the two new fields | 1, 2 |
| `src/task2/generator.py` | dtype selection, length-sorted batching | 1, 5 |
| `src/task2/generation/trainer.py` | `group_by_length`, loss-mask assertion | 3, 4 |
| `src/task2/generation/dataset.py` | drop evidence-free examples | 4 |
| `src/task2/predict.py` | batch BM25/lexref, prose cache, header guard | 5, 6 |
| `src/task2/pipeline/runner.py` | deadline prediction, telemetry | 6, 8 |
| `scripts/modal_app.py` | remove Python-constant overrides, dual-T4 fn | 2, 10 |
| `scripts/sweep_assembly.py` **(new)** | offline assembly sweep vs official scorer | 7 |
| `tests/unit/`, `tests/integration/` | one test per behaviour below | all |

---

## Task 1 — Inference dtype becomes a declared runtime setting *(fixes B-03)*

Biggest single win: this is what turns 6.7 h of decode into tens of minutes.

**Files**
- Modify: `src/task2/config/schema.py` (`InferenceRuntime` dataclass)
- Modify: `src/task2/config/loader.py` (parse + default)
- Modify: `configs/task2/runtime/modal_a100.yaml`
- Modify: `src/task2/generator.py:140-190`
- Test: `tests/unit/test_generator_load_mode.py` *(new)*

**Interfaces**
- Produces: `InferenceRuntime.generator_load_mode: str` ∈ `{"nf4", "bfloat16"}`, default `"nf4"`
- Produces: `QwenGenerator.load(..., load_mode: str = "nf4", merge_adapter: bool = False)`
- Consumed by: `runner.py` Stage 7 (Task 6)

- [x] **Step 1 — write the failing test**

```python
# tests/unit/test_generator_load_mode.py
import pytest
from src.task2.config.loader import load_resolved_config


def test_modal_a100_declares_bf16_inference():
    cfg = load_resolved_config(
        "configs/task2/algorithm.yaml",
        "configs/task2/runtime/modal_a100.yaml",
    )
    assert cfg.runtime.inference.generator_load_mode == "bfloat16"
    assert cfg.runtime.inference.merge_adapter is True


def test_default_load_mode_is_nf4_for_small_gpus():
    cfg = load_resolved_config(
        "configs/task2/algorithm.yaml",
        "configs/task2/runtime/colab_t4.yaml",
    )
    assert cfg.runtime.inference.generator_load_mode == "nf4"


def test_generator_load_rejects_unknown_mode():
    from src.task2.generator import QwenGenerator
    with pytest.raises(ValueError, match="unknown generator load mode"):
        QwenGenerator.load(model_path="Qwen/Qwen2.5-3B-Instruct", load_mode="int3")
```

- [x] **Step 2 — run it, confirm it fails**

`.venv311/bin/python -m pytest tests/unit/test_generator_load_mode.py -v`
Expected: `AttributeError: 'InferenceRuntime' object has no attribute 'generator_load_mode'`

- [x] **Step 3 — add the fields**

In `src/task2/config/schema.py`, on the inference dataclass:

```python
generator_load_mode: str = "nf4"   # "nf4" | "bfloat16"
merge_adapter: bool = False
```

In `src/task2/config/loader.py`, beside the existing `generation_batch_size` parse:

```python
generator_load_mode=str(inf_raw.get("generator_load_mode", "nf4")),
merge_adapter=bool(inf_raw.get("merge_adapter", False)),
```

In `configs/task2/runtime/modal_a100.yaml`, under `inference:`:

```yaml
  # A100-40GB: bf16 weights 6.18 GB + KV 5.44 GB @ batch 48 x 3072 = ~12-14 GB of 40 GB.
  # NF4 dequantises every forward pass and is ~3-4x slower to decode. See docs M-8.
  generator_load_mode: bfloat16
  merge_adapter: true
```

- [x] **Step 4 — thread it through the loader**

In `src/task2/generator.py:100-112`, add `load_mode: str = "nf4"` and
`merge_adapter: bool = False`. Replace the unconditional block at `:174-181`:

```python
if load_mode not in ("nf4", "bfloat16"):
    raise ValueError(f"unknown generator load mode: {load_mode!r}")
if load_mode == "nf4" and BitsAndBytesConfig is not None:
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
```

After the adapter is applied (`:191-193`), fold it in when asked:

```python
if merge_adapter and is_peft_model(model) and load_mode == "bfloat16":
    model = model.merge_and_unload()   # removes 7 modules x 36 layers of per-token matmuls
```

Guard the merge on `bfloat16`: merging into an NF4 base is not supported cleanly.

- [x] **Step 5 — run the test, confirm it passes**

`.venv311/bin/python -m pytest tests/unit/test_generator_load_mode.py -v` → 3 passed

- [x] **Step 6 — full suite**

`./test.sh` → all green.

- [x] **Step 7 — commit**

```bash
git add -A && git commit -m "perf(inference): declare generator load mode; bf16 + merged adapter on A100"
```

> **Honest caveat to carry forward.** Training happens on an NF4 base; inference on a bf16
> base is numerically *different*, not merely faster. This is standard QLoRA practice and
> the quality delta is normally small, but it is a real train/serve skew and it has not
> been measured here. Task 7's sweep runs on the bf16 outputs, so the chosen assembly is at
> least matched to the model that will actually serve. Logged as [Q-3](05-open-questions.md).

---

## Task 2 — `max_new_tokens` moves to config and rises to 1536 *(fixes B-02)*

Under the max-score directive the cap is set by the **score ceiling**, not by decode cost.
From [M-3](00-evidence-and-measurements.md#m-3--perfect-model-ceiling-vs-generation-cap-official-scorer):

| cap | perfect-model METEOR | coverage of gold answers |
|---|---|---|
| 512 (current) | 0.8961 | 67.5 % |
| 1024 | 0.9874 | 96.7 % |
| **1536** | **0.9979** | **99.5 %** |

1536 buys the last 0.010 over 1024 and costs only decode time, which is no longer scarce.
EOS stops most sequences far earlier, so the *average* cost is well below the cap.

**Files**
- Modify: `configs/task2/runtime/modal_a100.yaml`
- Modify: `scripts/modal_app.py:196-232`
- Modify: `tests/launchers/test_modal_app.py:154-161`

- [x] **Step 1 — update the existing assertion to the new contract**

```python
def test_remote_production_cfg_uses_configured_generation_ceiling():
    from scripts.modal_app import build_remote_production_cfg
    from src.task2.config.loader import load_resolved_config

    cfg_yaml = load_resolved_config(
        "configs/task2/algorithm.yaml", "configs/task2/runtime/modal_a100.yaml"
    )
    # 1536 reaches a 0.9979 perfect-model ceiling / 99.5% coverage. See docs M-2/M-3.
    assert cfg_yaml.runtime.inference.max_new_tokens == 1536
    cfg = build_remote_production_cfg(cfg_yaml)
    assert cfg.max_new_tokens == 1536
    assert cfg.best_fixed_candidate == "dual_assembled"
```

- [x] **Step 2 — run it, confirm it fails**

- [x] **Step 3 — implement**

Add `max_new_tokens: 1536` under `inference:` in `modal_a100.yaml`.
Delete `MODAL_MAX_NEW_TOKENS` from `scripts/modal_app.py:198-200` **including its false
comment**, and give `build_remote_production_cfg(resolved_cfg)` a parameter so the value
comes from the hashed config:

```python
def build_remote_production_cfg(resolved_cfg) -> Any:
    """Production selection for the Modal full run, sourced from the hashed runtime config."""
    import dataclasses
    from src.task2.production_config import get_default_production_selection
    return dataclasses.replace(
        get_default_production_selection(),
        max_new_tokens=int(resolved_cfg.runtime.inference.max_new_tokens),
        best_fixed_candidate=str(resolved_cfg.runtime.inference.best_fixed_candidate),
    )
```

Update the call site at `modal_app.py:521`. Add `best_fixed_candidate: dual_assembled` to
the YAML so the assembly policy is hashed too — Task 7 may change it, and that change must
be visible in the candidate.

- [x] **Step 4 — tests pass; `./test.sh` green**
- [x] **Step 5 — commit** `feat(config): source generation ceiling and assembly policy from hashed runtime config`

---

## Task 3 — `group_by_length` *(fixes B-06)*

**Files** — Modify `src/task2/generation/trainer.py:338-355`; Test `tests/unit/test_trainer_batching.py` *(new)*

- [x] **Step 1 — failing test**

```python
def test_sft_config_groups_by_length():
    from src.task2.generation.config import GeneratorTrainConfig
    from src.task2.generation.trainer import build_v16_sft_config
    args = build_v16_sft_config(
        GeneratorTrainConfig(model_id="Qwen/Qwen2.5-3B-Instruct", device="cpu"),
        output_dir="/tmp/x", per_device_train_batch_size=4,
    )
    assert args.group_by_length is True, "length grouping avoids padding every example to 2048"
    assert getattr(args, "packing", False) is False, "packing would break completion-only masking"
```

- [x] **Step 2 — run, confirm fail**
- [x] **Step 3 — add one line** to `sft_kwargs`: `"group_by_length": True,`
- [x] **Step 4 — run, confirm pass; `./test.sh` green**
- [x] **Step 5 — commit** `perf(train): group batches by length to stop padding to max_seq_len`

---

## Task 4 — Drop evidence-free SFT examples + assert the loss mask *(fixes B-07, B-13)*

**Files**
- Modify: `src/task2/generation/dataset.py:110-160`
- Modify: `src/task2/generation/trainer.py` (post-trainer assertion)
- Test: `tests/unit/test_dataset_requires_evidence.py` *(new)*

**Interfaces**
- Produces: `build_grounded_training_examples(..., require_evidence: bool = True)`
- Produces: diagnostics keys `dropped_no_evidence: int`, `kept_count: int`

- [x] **Step 1 — failing test**

```python
import pandas as pd
from src.task2.generation.dataset import build_grounded_training_examples


def test_examples_without_evidence_are_dropped(tmp_path):
    qa = pd.DataFrame([
        {"qa_id": "a", "question_raw": "Q1?", "answer_raw": "A1 statutory text", "fold_id": 0},
        {"qa_id": "b", "question_raw": "Q2?", "answer_raw": "A2 statutory text", "fold_id": 0},
    ])
    labels = pd.DataFrame([{"qa_id": "a", "positive_chunk_id": "c1"}])
    chunks = pd.DataFrame([{"chunk_id": "c1", "text_raw": "Điều 1. Nội dung."}])
    qa_p, l_p, c_p = (tmp_path / n for n in ("qa.parquet", "l.parquet", "c.parquet"))
    qa.to_parquet(qa_p); labels.to_parquet(l_p); chunks.to_parquet(c_p)

    ex, diag = build_grounded_training_examples(
        qa_path=str(qa_p), labels_path=str(l_p), chunks_path=str(c_p),
        require_evidence=True, return_diagnostics=True,
    )
    assert [e["qa_id"] for e in ex] == ["a"]       # "b" has no evidence
    assert diag["dropped_no_evidence"] == 1
```

- [x] **Step 2 — run, confirm fail**
- [x] **Step 3 — implement.** In the row loop at `dataset.py:110`, after `raw_evidence` is
  built:

```python
if require_evidence and not raw_evidence.strip():
    dropped_no_evidence += 1
    continue
```

Initialise the counter beside `dropped_count` and add it to `diag_summary`.
**Do not change the default for other call sites** — pass `require_evidence=True`
explicitly from `train_generator_qlora` so probe profiles keep their current behaviour
until someone decides otherwise.

- [x] **Step 4 — add the loss-mask assertion** in `trainer.py`, right after the
  `SFTTrainer` is constructed:

```python
if config.completion_only_loss:
    probe = trainer.data_collator([trainer.train_dataset[0]])
    labels = probe["labels"][0]
    masked = int((labels == -100).sum())
    if masked == 0:
        raise RuntimeError(
            "completion_only_loss=True but the collator masked 0 tokens — "
            "the installed TRL is not applying prompt masking; loss would include "
            "the entire evidence block."
        )
    print(f"[+] completion-only loss verified: {masked} prompt tokens masked to -100")
```

- [x] **Step 5 — run, confirm pass**
- [x] **Step 6 — measure the real effect** (CPU, no GPU):

```bash
.venv311/bin/python -c "
from src.task2.generation.dataset import build_grounded_training_examples
for req in (False, True):
    ex, d = build_grounded_training_examples(
        qa_path='kaggle_dataset/qa_unique.parquet',
        labels_path='kaggle_dataset/retrieval_labels.parquet',
        chunks_path='kaggle_dataset/legal_chunks.parquet',
        require_evidence=req, return_diagnostics=True)
    print(f'require_evidence={req}: kept={d[\"kept_count\"]} dropped_no_ev={d.get(\"dropped_no_evidence\",0)}')
"
```

Record both numbers in the task notes — they set the optimizer-step count used in
[`04-runbook-modal-a100.md`](04-runbook-modal-a100.md).

- [x] **Step 7 — `./test.sh` green; commit** `fix(train): require evidence for SFT examples and verify completion-only masking`

---

## Task 5 — Batch the retrieval arms and length-sort the prompts *(fixes B-09, B-10, B-12)*

**Files** — Modify `src/task2/predict.py:524-556`, `src/task2/generator.py:276-300`;
Test `tests/unit/test_batch_order_preserved.py` *(new)*

⚠️ The permutation inversion is the risky part: an off-by-one silently attaches answers to
the wrong questions and **no downstream check would catch it**. Test it first.

- [ ] **Step 1 — failing test (order preservation is the contract)**

```python
def test_length_sorted_generation_returns_original_order(monkeypatch):
    from src.task2.generator import QwenGenerator
    gen = QwenGenerator(runtime="torch")
    gen.model, gen.tokenizer = object(), object()
    # stub: echo the prompt so order is checkable
    monkeypatch.setattr(gen, "_generate_texts", lambda batch, mnt: [f"ans::{p}" for p in batch])
    monkeypatch.setattr(gen, "format_instance_prompt", lambda q, ev: q)
    items = [("q" * n, "") for n in (5, 200, 1, 90, 40)]
    out = gen.generate_batch(items, max_new_tokens=8, batch_size=2)
    assert out == [f"ans::{q}" for q, _ in items]   # original order, not sorted order
```

- [x] **Step 2 — run, confirm fail** (or pass trivially today — then still keep it as the
  regression that guards the change you are about to make)
- [x] **Step 3 — implement length sorting** in `generator.py:276`:

```python
prompts = [self.format_instance_prompt(q, ev) for q, ev in items]
order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))   # longest first
sorted_prompts = [prompts[i] for i in order]
# ... generate over sorted_prompts into sorted_results ...
results = [None] * len(prompts)
for slot, text in zip(order, sorted_results):
    results[slot] = text
```

Longest-first also surfaces an OOM on the very first batch rather than 80 % of the way in,
where the existing halving retry at `:297-300` is far more expensive.

- [x] **Step 4 — batch the retrieval arms** in `predict.py`, hoisting above the loop at `:543`:

```python
lex_all = (
    search_legal_references(dense_queries, self.legal_index, self.legal_rows, k=pool)
    if options.get("use_legal_reference") and self.legal_index is not None
    else [[] for _ in dense_queries]
)
bm25_all = (
    self.bm25.search_batch(dense_queries, top_k=pool)
    if self.bm25 and hasattr(self.bm25, "search_batch")
    else [self.bm25.search(q, top_k=pool) if self.bm25 else [] for q in dense_queries]
)
```

then index `lex_all[idx]` / `bm25_all[idx]` inside the loop.

- [x] **Step 5 — prove equivalence before trusting it**

```bash
.venv311/bin/python -m pytest tests/unit/test_batch_order_preserved.py tests/ -k "predict or retriev" -v
```

Then a 50-query A/B of batched vs per-query retrieval asserting identical chunk ids.
**Do not skip this** — a silent retrieval regression costs more than the speedup gains.

- [ ] **Step 6 — `./test.sh` green; commit** `perf(inference): batch bm25/lexref arms and length-sort generation batches`

---

## Task 6 — Resume-safe generation cache + honest deadline *(fixes B-04, B-05, B-11, B-17)*

The highest-value task after Task 1: it makes Task 7 free and makes a timeout survivable.

**Files** — Modify `src/task2/predict.py` (`predict_batch`), `src/task2/pipeline/runner.py:414-600`,
`scripts/modal_app.py:205-214`; Test `tests/integration/test_generation_cache_resume.py` *(new)*

**Interfaces**
- Produces: `predict_batch(..., raw_cache_path: Optional[str] = None, deadline: Optional[Callable[[], int]] = None)`
- Produces: JSONL at `raw_cache_path`, one record per line:
  `{"qa_id": str, "prompt_sha256": str, "raw": str}`
- Produces: `results["stages"]["submission"]["answer_length"] = {"mean","median","p90"}`

- [x] **Step 1 — failing test**

```python
def test_generation_resumes_from_prompt_keyed_cache(tmp_path):
    """A second run with an identical prompt must reuse the cache and not regenerate."""
    # build a pipeline with a counting stub generator, run 4 items, kill after 2,
    # re-run, assert the stub was called exactly 2 more times and all 4 answers returned.
```

- [x] **Step 2 — run, confirm fail**
- [x] **Step 3 — implement the cache.** Key on `sha256(prompt + adapter_path + str(max_new_tokens))`,
  never on `qa_id` alone — v10 of the notebook keyed by id and shipped 1,000 stale answers
  generated before the retrieval fixes existed. Append-and-flush after every batch.
- [x] **Step 4 — check the deadline between batches**, returning `INCOMPLETE` with the cache
  intact instead of being killed mid-stage.
- [x] **Step 5 — derive the prediction** in `build_remote_paths`, replacing the 3300 s literal:

```python
"predicted_inference_seconds": int(
    num_queries * max_new_tokens / max(1.0, measured_tokens_per_second) * 1.4
),
```

with `measured_tokens_per_second` supplied by the A100 micro-probe (Task 10).

- [x] **Step 6 — add the header-only guard** at `predict.py:472` and `:669`:

```python
_HEADER_ONLY = {"căn cứ quy định của pháp luật:"}
if (not selected or not str(selected).strip()
        or str(selected).strip().lower() in _HEADER_ONLY):
    ...existing fallback...
```

and count occurrences in the provenance report.

- [x] **Step 7 — add the length telemetry** and a sanity band in `runner.py`; fail the stage
  if mean answer words < 150 or > 2000.
- [x] **Step 8 — `./test.sh` green; commit** `feat(inference): prompt-keyed resume cache, derived deadline, answer telemetry`

---

## Task 7 — Offline assembly sweep against the official scorer *(resolves the open architecture question)*

This is the task that decides whether stitching stays on. It costs **zero GPU time**
because it runs on Task 6's cache.

**Files** — Create `scripts/sweep_assembly.py`; Test `tests/unit/test_sweep_assembly.py` *(new)*

**Interfaces**
- Consumes: the raw-prose JSONL from Task 6, a held-out QA set with gold answers, retrieval contexts
- Produces: `artifacts/labs/assembly_sweep.json` — `{strategy: {"meteor": float, "rouge_l": float, "mean_words": float}}`

- [x] **Step 1 — failing test:** given 3 synthetic (ref, prose, article) triples, the sweep
  returns one entry per strategy and ranks `dual_assembled` above `generated` when the
  prose is a poor 40-word stub.
- [x] **Step 2 — run, confirm fail**
- [x] **Step 3 — implement.** Score with the **official** metric, not a reimplementation:

```python
from nltk.translate.meteor_score import meteor_score   # alpha=0.9 defaults — do not override
from rouge_score import rouge_scorer

meteor = np.mean([meteor_score([r.split()], h.split()) for r, h in zip(refs, hyps)])
```

Sweep at minimum: `generated`, `snapped`, `dual_assembled` at citation budgets
{2000, 3000, 4000, 6000}, `focused_complete_clause`, `primary_article_budgeted_units`,
and `cite-only`.

- [ ] **Step 4 — run it on real held-out data, record the table** into
  `artifacts/labs/assembly_sweep.json` and paste it into [`05-open-questions.md`](05-open-questions.md) under Q-1.
- [ ] **Step 5 — set `best_fixed_candidate` in `modal_a100.yaml` to the measured winner.**
  If `generated` wins, stitching is off and B-02's raised ceiling matters even more
  ([M-6](00-evidence-and-measurements.md#m-6--prose-length-inside-dual-assembly-n--150-oracle-article): end-to-end reaches 0.9953 at 1024 vs 0.9026 at 512).
- [x] **Step 6 — commit** `feat(labs): offline assembly sweep scored with the official metric`

---

## Task 8 — Honest dev metric, honest release *(fixes B-08, B-14)*

- [x] **Step 1 — failing test:** `modal_a100` must not report a `selected_meteor` computed on
  a fold that training consumed.
- [x] **Step 2 — implement.** Either hold out a fold from training, or set
  `run_dev_evaluation=False` for `modal_a100` (`profiles.py:130-147`) and emit
  `"selected_meteor": None, "reason": "no held-out fold: production trains on all data"`.
- [x] **Step 3 — narrow the HF `except`** at `modal_app.py:622`: let the deliberate
  `ValueError("refusing release: …")` propagate; catch only upload transport errors, and
  reflect a failed release in the run's top-level status.
- [x] **Step 4 — `./test.sh` green; commit** `fix(provenance): stop publishing a leaked dev metric; stop swallowing release refusals`

---

## Task 9 — Re-mint the candidate *(fixes B-01)*

**Only after Tasks 1-8 have landed.** Every prior task changes the hash; minting earlier
means minting twice.

- [ ] **Step 1 — add the commit guard** in `src/task2/provenance/candidate.py:157`:

```python
live = os.environ.get("GIT_COMMIT_SHA") or _read_git_head()
if live and self.git_commit_sha != live and not allow_commit_drift:
    raise ValueError(
        f"candidate pins commit {self.git_commit_sha[:8]} but the checkout is {live[:8]}; "
        "re-mint the candidate (scripts/freeze_candidate.py) before running"
    )
```

- [ ] **Step 2 — test it** both ways: matching sha passes, mismatched sha raises.
- [x] **Step 3 — commit everything, then mint**

```bash
git add -A && git commit -m "feat(a100): inference dtype, resume cache, evidence-gated SFT, honest provenance"
.venv311/bin/python scripts/freeze_candidate.py            # writes artifacts/candidates/<new_id>/
```

- [x] **Step 4 — verify** the new manifest's `git_commit_sha` equals `git rev-parse HEAD`
  and that the four `runtime_profile_sha256` entries changed.
- [x] **Step 5 — `./test.sh` green.**

---

## Task 10 — Dual-T4 Modal gate, then the chain *(fixes B-16)*

- [x] **Step 1 — add the function** in `scripts/modal_app.py` beside `run_modal_t4_remote`:

```python
@app.function(gpu="T4:2", timeout=3600,
              volumes={"/data": data_volume, "/runs": runs_volume}, secrets=secrets)
def run_modal_t4x2_remote(request: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the kaggle_t4x2 root gate on two Modal T4s (no Kaggle notebook needed)."""
```

Mirror `run_modal_t4_remote`, calling `run_gpu_gate(stage="kaggle_t4x2", ...)`. Extend
`build_modal_request` to accept `stage="kaggle_t4x2"` with **no** parent
(`GATE_PARENTS["kaggle_t4x2"] == ()`).

- [x] **Step 2 — extend `tests/launchers/test_modal_app.py`:** `kaggle_t4x2` accepts no
  parent and rejects one if supplied.
- [ ] **Step 3 — have the micro-probe emit `measured_tokens_per_second`** so Task 6's
  deadline formula has a real input.
- [x] **Step 4 — `./test.sh` green; commit** `feat(modal): dual-T4 root gate and measured decode throughput`
- [ ] **Step 5 — run the chain.** Commands, expected timings and abort criteria are in
  [`04-runbook-modal-a100.md`](04-runbook-modal-a100.md). **Stop at the micro-probe and read its numbers before
  launching the full stage.**

---

## Task 11 — 🔴 Retrieval accuracy: the +0.052 lever *(the largest single item)*

**This is now the highest-priority task.** [`02-scoring-model.md`](02-scoring-model.md) gives
`METEOR ≈ 0.4522 + 0.2288 · p`, with p ≈ 0.42 today and p ≈ 0.646 needed for 0.60. Every
other task in this plan combined is worth less than this one.

**Blocked on one thing first:** the local dense index is a mock of random vectors
([M-12](00-evidence-and-measurements.md#m-12--the-local-dek21-dense-index-is-a-mock-index-of-random-vectors), bug [B-18](01-bug-register.md#b-18)), so **p has never actually been measured** — 0.42 is inferred
from the observed leaderboard score. Step 1 fixes that.

**Files**
- Delete: `kaggle_dataset/indexes/dek21/` (mock)
- Create: `scripts/measure_retrieval.py` — offline top-k article accuracy
- Modify: `configs/task2/algorithm.yaml` (pool and fusion weights — **score-affecting, re-mints**)
- Modify: `src/task2/predict.py` (article-level aggregation)
- Test: `tests/unit/test_article_aggregation.py` *(new)*

**Interfaces**
- Produces: `measure_retrieval(queries, gold_article_ids, index_dir, k_list) -> {"top1": float, "top5": …, "top20": …}`
- Produces: `artifacts/labs/retrieval_accuracy.json`

- [ ] **Step 1 — build a real index and get a real number.**
  Delete the mock, rebuild on GPU (this is the one GPU dependency in the task), then
  measure top-1 / top-5 / top-20 **article** accuracy against
  `retrieval_labels.parquet::positive_article_id` (4,759 labelled questions).
  Report dense-only, BM25-only, and fused. **Do not tune anything before this number exists.**

- [ ] **Step 2 — article-level aggregation instead of chunk top-1.**
  The citation that gets stitched is an **article**, but ranking happens over **chunks**
  (`src/task2/predict.py:543-580`). An article whose five clauses each rank 6-10 currently
  loses to an article with one chunk at rank 5. Aggregate chunk scores per
  `parent_article_id` (max and sum) over the top-N pool, then rank articles.
  Measure both against Step 1's baseline. Cheap, offline, and plausibly worth several points of p.

- [ ] **Step 3 — widen the pool, rerank harder.**
  `candidate_pool: 50` (`configs/task2/algorithm.yaml:45`) caps what the cross-encoder can
  rescue. With time no longer scarce, sweep 50 → 100 → 200 offline and keep the best.
  Reranking 200 × 1,918 pairs on an A100 is minutes, not hours.

- [ ] **Step 4 — re-tune the fusion weights** (`algorithm.yaml:52-56`) on the offline harness
  now that dense is real. The current weights were set when the dense arm may have been
  contributing noise locally; they have never been validated against measured article accuracy.

- [ ] **Step 5 — record the measured p** in `artifacts/labs/retrieval_accuracy.json` and
  update the projection in [`02-scoring-model.md` §2](02-scoring-model.md) with the real figure.

- [ ] **Step 6 — `./test.sh` green; commit** `feat(retrieval): measured article accuracy, article-level aggregation, retuned fusion`

> Every change here touches `algorithm.yaml` and therefore re-mints the candidate. Do the
> whole task before Task 9, not after.

---

## Task 12 — Remove `colab_t4` from the workflow *(fixes B-19)*

Colab is no longer part of this project. The gate DAG already permits
`kaggle_t4x2 → a100_micro_probe`, so nothing depends on the middle hop.

**Files** — `scripts/modal_app.py` (drop `run_modal_t4_remote`, the `"colab_t4"` stage,
`colab_report` plumbing), `scripts/run_gpu_gate.py:81-93`,
`src/task2/pipeline/profiles.py:87-104`, plus **13 test files** that reference the stage.

- [ ] **Step 1 — operational removal first (zero risk).** Stop running the stage and stop
  passing `colab_report` into `build_modal_request` (`scripts/modal_app.py:698-706`,
  `:129-133`). `build_production_run_bundle` already accepts
  `colab_t4_report_path=None` (`src/task2/provenance/run_bundle.py:128`), so the release
  packages cleanly without it.
- [ ] **Step 2 — confirm the chain still validates:**

```bash
.venv311/bin/python -m pytest tests/integration/test_candidate_gate_chain.py tests/launchers/ -v
```

- [ ] **Step 3 — code removal (separate commit, do not mix with Step 1).** Drop the stage
  from `GATE_PARENTS`, `STAGE_RUNTIME_PROFILE`, `VALID_V16_PROFILES`, the Modal T4 function,
  and `configs/task2/runtime/colab_t4.yaml`. Update each referencing test to assert the
  **new** two-stage chain rather than deleting the assertion.
- [ ] **Step 4 — `./test.sh` green** (expect the count to drop below 252; the fall must be
  fully explained by removed `colab_t4` cases — check each one).
- [x] **Step 5 — commit** `chore(gates): remove colab_t4 stage; chain is kaggle_t4x2 -> a100_micro_probe -> full`

> Removing `colab_t4` changes `runtime_profile_sha256` and therefore the candidate id. Land
> it before Task 9.

---

## Task 13 — Spend the freed time on training quality

Only after Tasks 3-4 have made a step cheap. Each sub-step is an independent A/B on a
held-out fold — **measure, do not assume**.

- [ ] **Step 1 — epochs 1 → 2, then 3.** `configs/task2/algorithm.yaml:34`
  (`num_train_epochs: 1`). With ~4,372 deduplicated evidence-bearing examples (Task 4 + Q-7)
  at effective batch 8, one epoch is only ~547 steps — light for a LoRA. Watch held-out
  METEOR, not train loss; stop at the first epoch that does not improve it.
- [ ] **Step 2 — LoRA rank 16 → 32.** `algorithm.yaml:19`. ⚠️ **Check the parameter budget
  first** — `scripts/audit_parameters.py` enforces `< 4e9` and the current margin is 211 M.
  Compute the new count *before* launching, not after.
- [ ] **Step 3 — re-sweep assembly (Task 7) after every generator change.** [M-6](00-evidence-and-measurements.md#m-6--prose-length-inside-dual-assembly-n--150-oracle-article) shows the
  optimal assembly **flips** as the generator improves: stitching helps a weak model by
  +0.23 and hurts a strong one by −0.18. A better adapter with a stale assembly policy can
  score *worse*. This re-sweep is mandatory, not optional.
- [ ] **Step 4 — commit each A/B separately** with its measured held-out delta in the message.

---

## Self-review

- **Spec coverage:** every lever in [`02-scoring-model.md` §3](02-scoring-model.md) now maps to a task —
  **lever 1 → Task 11** (added under the max-score directive; it is the largest single
  item at +0.052), lever 2 → Task 2, lever 3 → already implemented (verify via Task 7's
  provenance counts), lever 4 → Task 7, lever 5 → Task 4, levers 6-7 → Global Constraint 7.
- **Recommended order under "score first":** 11 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 13 → 8 → 12 → 9 → 10.
  Task 11 leads because it is worth more than everything after it combined; Tasks 1-6 follow
  because they are what make Task 13's extra epochs and Task 2's 1536-token ceiling
  affordable inside one container.
- **Type consistency:** `generator_load_mode` / `merge_adapter` / `max_new_tokens` /
  `best_fixed_candidate` are read from `cfg.runtime.inference` in Tasks 1, 2, 6 under those
  exact names.
- **Placeholders:** Task 6 Step 1 and Task 7 Step 1 give test *intent* rather than full
  bodies, because both need fixtures that depend on Task 5's final signatures. Write them
  first, from the interface blocks, before implementing.
