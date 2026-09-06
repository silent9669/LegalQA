# LegalQA V16 — Architecture Review & Required Hardening

**Repository:** `silent9669/LegalQA`  
**Audited HEAD:** `7b24a1b66ee231482cea47905bce0b07f8e1f152`  
**Target:** DSC 2026 Task 2 LegalQA, Kaggle dual Tesla T4  
**Runtime contract:** API 16  
**Primary metric:** organizer whitespace-tokenized METEOR  
**Constraint:** total learned parameters strictly `< 4,000,000,000`

---

## 1. Executive decision

### Architecture verdict

**KEEP the current V16 architecture. Do not redesign Stack A.**

The current intended production stack is sound:

```text
QA memory
   + BM25S
   + DEk21 v2 dense retrieval
        ↓
      RRF
        ↓
BGE-reranker-v2-m3
        ↓
structured evidence packing
        ↓
extractive/stitch candidates
        +
Qwen2.5-3B-Instruct 4-bit QLoRA
with selective Liger fused-linear CE
        ↓
Protocol-8 component/candidate promotion
        ↓
PROMOTED production policy
```

GPU ownership must remain:

```text
cuda:0 → Qwen2.5-3B generator training/inference
cuda:1 → dense retrieval + BGE reranker
```

Do **not** introduce DataParallel, DDP, FSDP, ZeRO, another learned model, a larger generator, or an external corpus/API.

### Current readiness

- **GO:** CPU/contract/compatibility state is strong enough to prepare the API16 package and run the first real Kaggle GPU probe.
- **GO:** `generator_probe_worstcase`.
- **After it passes:** `generator_probe_endurance`.
- **After both pass:** `screen_fold0`.
- **NO-GO:** `final_train_and_submit` until Protocol-8 produces a valid `PROMOTED` configuration.
- **NO-GO:** `reuse_final_checkpoints_and_submit` using the current Drive checkpoint tree. The Drive tree contains old MLX generator adapters, not the canonical V16 HF adapter/checkpoint set.

---

## 2. Evidence that must remain unchanged

The inherited dataset is intentionally bound to **V15**:

```text
artifacts/inherited/dataset_manifest_v15.json
runtime_api_version = 15
git_sha = 151313fc3126615ec11c08ca68f154d5b0c5406f
```

This is **not** stale runtime code. It is the immutable provenance manifest for inherited artifacts.

The new Kaggle runtime package must independently be API16 and bind its root/code manifests to the exact final V16 HEAD.

Drive layout inspected:

```text
raw/
  train.json
  warmup.json
  public-official.json
  selected-contexts/
  selected-contexts.zip

task2/data/
  legal_chunks.parquet
  qa_unique.parquet
  known_qa.json
  qa_citations.parquet
  retrieval_labels.parquet
  fold_assignments.parquet
  reranker_training_pairs.parquet

task2/indexes/
  bm25/
  dek21/

task2/evaluations/
  oof_predictions.parquet
  oof_summary.json

task2/checkpoints/generator/
  mlx_adapter/
  mlx_test_adapter/
```

The DEk21 `embeddings.npy` size is consistent with approximately `801,863 × 768 × FP16`, and the BM25 directory layout matches the inherited manifest's expected index files.

Before publishing any new Kaggle package, still run the repository's SHA/hash verification; do not replace hash validation with size-only checks.

---

## 3. Why the V16 generator strategy should stay

The V15 real T4 probe localized the first-step OOM to the old TRL LM-head projection path. Reducing token CE chunks did not solve it.

V16 correctly changes the **loss backend**, not model capacity:

```text
Qwen/Qwen2.5-3B-Instruct
4-bit NF4 + double quant
FP16 compute
LoRA r=16, alpha=32, dropout=0.05
targets = q/k/v/o/gate/up/down projections
max_seq_len=2048
batch=1
grad_accum=8
paged_adamw_8bit
gradient checkpointing
activation offloading
completion-only loss
TRL loss_type = nll
Liger fused_linear_cross_entropy = true
chunked_nll = disabled
trainer_n_gpu = 1
```

Keep `liger-kernel==0.8.2`.

---

# 4. REQUIRED FIXES

Implement these before calling the repository production-hardened.

## P0.1 — Enforce generator configuration on screen/final paths

### Problem

`train_generator_qlora()` infers a strict profile only for:

```text
probe_mode=worst_case
probe_mode=endurance
```

When `screen_fold0` or `final_train_and_submit` calls it with `probe_mode=None`, the trainer treats the profile as `"standard"` and therefore does not run the strict `validate_generator_config_for_profile()` checks for screen/final.

The runner currently builds matching defaults, so this is not an immediate behavior bug, but it is a contract hole: future config drift could reach production without being rejected.

### Required implementation

Preferred low-risk design:

1. Add an explicit `execution_profile: str` argument to `train_generator_qlora()`.
2. Pass `profile.name` from `src/task2/pipeline/runner.py`.
3. Always call:

```python
validate_generator_config_for_profile(config, execution_profile)
```

when the execution profile is one of:

```text
generator_probe_worstcase
generator_probe_endurance
screen_fold0
final_train_and_submit
```

4. Preserve compatibility for unit/CPU utility calls that intentionally use `"standard"`.
5. Do not duplicate hard-coded generator settings across more modules.

### Tests

Add/extend tests proving a mutated production config is rejected in both:

```text
screen_fold0
final_train_and_submit
```

Test at least:
- max_seq_len != 2048
- LoRA rank/alpha drift
- activation_offloading=False
- use_liger_fused_ce=False
- device != cuda:0
- trainer_n_gpu != 1

---

## P0.2 — Make `final_training` packaging require a PROMOTED config

### Problem

`scripts/package_kaggle_dataset.py --profile final_training` currently checks that the production config exists and uses it to decide whether reranker pairs are required, but it does **not** enforce:

```text
status = PROMOTED
screen_protocol_version >= 8
```

The runtime later blocks an unvalidated final run, but a package named `final_training` should never be generated from an UNVALIDATED policy.

### Required implementation

For `profile == "final_training"`:

```python
prod_cfg = load_production_selection(...)
validate_production_selection_for_profile(
    prod_cfg,
    "final_train_and_submit",
    allow_unvalidated_final=False,
)
```

The package must fail before staging if the config is not eligible.

Use the existing `default` package profile for probes and `screen_fold0`, or add a clearly named `screen_runtime` profile only if needed. Do not weaken final strictness.

### Tests

- UNVALIDATED + `final_training` → deterministic failure.
- PROMOTED Protocol-8 + `final_training` → allowed.
- default/probe package remains possible before promotion.

---

## P0.3 — Make checkpoint reuse deterministic and provenance-bound

### Problem

The runner currently searches broadly and takes the first result from patterns similar to:

```python
glob("/kaggle/input/**/checkpoints/reranker/best", recursive=True)
glob("/kaggle/input/**/checkpoints/generator/hf_adapter", recursive=True)
```

If multiple Kaggle dataset versions/checkpoint inputs are mounted, ordering can select the wrong checkpoint.

### Required implementation

Create one authoritative checkpoint resolver.

For each candidate:

1. Require the expected directory structure.
2. Require the component checkpoint manifest.
3. Verify:
   - runtime API 16 where applicable,
   - expected base model,
   - component type,
   - checkpoint completeness,
   - release/provenance fields available in the manifest.
4. Prefer an exact path from the promoted production configuration/provenance.
5. If zero valid candidates exist → fail.
6. If more than one valid candidate remains and no exact provenance selects one → fail as ambiguous.
7. Never silently choose `candidates[0]`.

Add tests with two mounted fake checkpoint versions and prove stale/ambiguous selection is rejected.

---

## P0.4 — Validate promotion provenance again before final/reuse

### Problem

The production schema fields are named:

```text
source_screen_manifest
source_screen_sha256
```

but the current promoter stores the **promotion report** path/hash in them. In addition, final profile validation primarily checks `status` and `screen_protocol_version`; it does not cryptographically revalidate the source artifact before a final/reuse execution.

### Required implementation

Choose the smallest safe migration.

Recommended:

- Introduce a schema-compatible provenance block containing explicit fields such as:

```yaml
provenance:
  promotion_report_path: ...
  promotion_report_sha256: ...
  screen_run_manifest_path: ...
  screen_run_manifest_sha256: ...
  sample_ids_sha256: ...
  runtime_api_version: 16
  screen_protocol_version: 8
```

- Keep legacy fields readable if needed.
- Before `final_train_and_submit` and `reuse_final_checkpoints_and_submit`:
  - source file exists,
  - SHA256 matches,
  - Protocol = 8,
  - runtime API = 16,
  - sample/provenance linkage is internally consistent.
- Fail loud on stale/copied/tampered promotion artifacts.

Do not silently auto-promote.

---

# 5. P1 CLEANUP / RELEASE QUALITY

## P1.1 — Refresh README to V16 truth

The root README currently describes old “V3 Production” behavior and an outdated `train_and_submit` default.

Update it so the canonical sequence is:

```text
A. generator_probe_worstcase
B. generator_probe_endurance
C. screen_fold0
D. final_train_and_submit
```

State clearly:
- committed notebook default is `generator_probe_worstcase`;
- final requires a PROMOTED Protocol-8 config;
- API16 runtime package is separate from the immutable inherited V15 data manifest;
- reuse requires canonical V16 HF checkpoints.

Do not make README override `docs/active/v16/`.

---

## P1.2 — Remove configuration authority drift

There are overlapping config files with different values. Examples include differences between `configs/task2.yaml` and the active pipeline/production config for retrieval top-K and generator `max_new_tokens`.

Define authority:

```text
configs/runtime_api.yaml            → runtime API
configs/models.yaml                 → model identities / parameter accounting
configs/pipeline.yaml               → infrastructure/runtime defaults
configs/production_selection.yaml   → promoted production decisions
```

Then either:
- remove/deprecate `configs/task2.yaml` from production code paths, or
- derive/validate it so conflicting values cannot exist.

Add a config-consistency contract test.

---

## P1.3 — Freeze the final user-space HF environment after the real probe passes

Current CI proves the exact HF stack:

```text
transformers 5.0.0
accelerate 1.13.0
datasets 5.0.0
peft 0.19.1
trl 1.12.0
bitsandbytes 0.50.2
liger-kernel 0.8.2
```

The Kaggle requirements intentionally use several version floors.

Do **not** touch Kaggle's protected:

```text
torch
torchvision
torchaudio
triton
cuda-*
nvidia-*
```

Once Run A succeeds on the real T4 image, record/freeze the tested user-space versions for the score-max final run, while preserving the existing runtime symbol checks.

Do not preemptively change the protected stack.

---

# 6. P2 SCORE-MAXIMIZATION HARDENING

These are secondary to getting a real GPU PASS.

## P2.1 — Make Protocol-8 promotion uncertainty-aware

Current canonical screen size is 250 held-out queries.

Keep the first 250-query screen for iteration speed, but before the final score-max run add one of:

1. deterministic larger confirmation sample (e.g. 500–1000 if runtime permits), or
2. full held-out fold evaluation, or
3. bootstrap confidence intervals over the same fixed IDs.

For reranker/QLoRA promotion, do not promote based on a tiny difference inside noise.

At minimum report:
- METEOR mean,
- bootstrap CI,
- delta vs baseline,
- retrieval coverage,
- same sample hash for all systems.

Keep the existing METEOR tolerance as a floor, not the only statistical criterion.

---

## P2.2 — Preserve the fixed-candidate guardrail

Historical Drive fast-OOF summary (`100` examples) was directionally very strong:

```text
stitched_extract ≈ 0.3051 METEOR
selected         ≈ 0.2009
generated        ≈ 0.0749
oracle_best      ≈ 0.3112
```

This does **not** prove final leaderboard behavior, but it shows that:
- the stitched extractive answer was much stronger than the old selector/generator path;
- the oracle gap over stitched was small on that quick sample.

Therefore:

- keep `stitched_extract`/best fixed candidate as the deployment guardrail;
- do not force QLoRA into final inference merely because it trained successfully;
- only promote QLoRA when Protocol-8 shows a real held-out gain;
- only promote a learned meta-selector if leakage-safe cross-fit clearly beats the best fixed family.

---

# 7. Required tests / verification

The coding agent must finish with all relevant tests green.

Minimum local checks:

```bash
pytest -q
python scripts/audit_parameters.py
python scripts/preflight_kaggle.py \
  --pipeline_config configs/pipeline.yaml \
  --models_config configs/models.yaml
```

Run inherited-data verification using the repository's existing verifier against the actual source root/package:

```bash
python scripts/verify_inherited_dataset.py \
  --root <source-root> \
  --manifest artifacts/inherited/dataset_manifest_v15.json
```

Package a **probe/screen** API16 runtime before promotion:

```bash
python scripts/package_kaggle_dataset.py --profile default
```

Do not create a `final_training` package until a PROMOTED Protocol-8 config exists.

The coding agent must not trigger the user's Kaggle GPU run automatically.

---

# 8. Manual Kaggle acceptance sequence

After the code hardening is merged and a fresh API16 dataset version is uploaded:

## Run A

```python
EXECUTION_PROFILE = "generator_probe_worstcase"
ALLOW_SINGLE_GPU_SMOKE = False
```

PASS only if:
- two Tesla T4s found;
- API16 verified;
- selective Liger active;
- `fused_linear_cross_entropy=True`;
- `loss_type=nll`;
- `trainer_n_gpu=1`;
- three optimizer steps complete;
- finite loss;
- no OOM;
- adapter saved;
- strict reload works;
- non-empty generation.

## Run B

```python
EXECUTION_PROFILE = "generator_probe_endurance"
```

PASS only if:
- 30 optimizer steps complete;
- no OOM/NaN/Inf;
- no clear unbounded VRAM growth;
- strict reload passes.

## Run C

```python
EXECUTION_PROFILE = "screen_fold0"
```

Required:
- R0G0
- R1G0
- R_SELECTED_G1
- identical sample hash;
- no mocks/fallbacks;
- `promotion_report.json`;
- `promoted_production_selection.yaml`;
- `screen_handoff.zip`;
- resulting config status = PROMOTED.

Audit the promotion before final training.

## Run D

Commit/use the audited PROMOTED config, rebuild the strict final package, then:

```python
EXECUTION_PROFILE = "final_train_and_submit"
ALLOW_UNVALIDATED_FINAL = False
```

Final submission must:
- contain exactly 1,000 official IDs;
- contain no missing/extra IDs;
- use only promoted components;
- pass parameter audit `< 4B`;
- have no model fallback.

---

# 9. Non-negotiable prohibitions

Do not:

- change Stack A model families without new measured evidence;
- reduce Qwen from 3B merely to avoid the old V15 OOM;
- re-enable `chunked_nll`;
- enable broad Liger kernels beyond the validated selective fused-linear CE config;
- use DataParallel/DDP/FSDP/ZeRO;
- train generator on both T4s;
- alter inherited data hashes;
- use external legal corpora or answer APIs;
- bypass `PROMOTED` with `ALLOW_UNVALIDATED_FINAL=True` for the score-max run;
- pick the first checkpoint found from an ambiguous mount;
- package an UNVALIDATED config as `final_training`;
- write secrets/tokens into logs, manifests, code, notebooks, or artifacts.

---

# 10. Definition of done

The fix is complete only when:

- latest unit/contract/CI tests pass;
- production generator config is validated for screen and final;
- final packaging rejects UNVALIDATED policies;
- checkpoint reuse is deterministic and provenance-checked;
- final/reuse revalidate promotion provenance;
- README/config authority matches V16;
- inherited V15 hashes remain unchanged;
- parameter budget remains `< 4B`;
- notebook default remains `generator_probe_worstcase`;
- no Kaggle protected CUDA/Torch packages are replaced;
- the repository is ready for the user to run **Run A** manually.

Do not claim final Kaggle readiness until the real T4 Run A and Run B pass.
