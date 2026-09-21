# 04 — Runbook: the Modal A100 run

Read [`03-implementation-plan.md`](03-implementation-plan.md) first. This file covers only the execution of the run
itself: commands, expected numbers, and when to kill it.

---

## 0. Preconditions — all must hold

```bash
cd "/Users/phucdang/Documents/LegalQA - Public Test"

./test.sh                                     # 252/252, working tree clean
git status --porcelain                        # empty
git rev-parse HEAD                            # note it

CID=$(ls -t artifacts/candidates | head -1)
.venv311/bin/python -c "
import json, subprocess
cid = '$CID'
m = json.load(open(f'artifacts/candidates/{cid}/candidate_manifest.json'))
head = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
assert m['git_commit_sha'] == head, f'STALE CANDIDATE: {m[\"git_commit_sha\"][:8]} vs HEAD {head[:8]}'
print('candidate', cid, 'pins HEAD OK')
"
modal app list                                # expect 0 running
```

If the candidate check fails, **stop** — that is bug B-01 and it means the release would
misattribute its own code. Re-mint (Task 9) before going further.

---

## 1. The gate chain

Governance is a DAG (`scripts/run_gpu_gate.py:81-85`); each stage needs its parent's PASS
report for the **same** candidate id.

```
kaggle_t4x2  ──▶  a100_micro_probe  ──▶  full
  (2 x T4)            (A100)            (A100)
```

**`colab_t4` is removed from the workflow** (bug B-19, plan Task 12). `a100_micro_probe`
accepts `kaggle_t4x2` directly, so no middle hop is needed and Colab is not involved at
any point.

| stage | GPU | expect | cost |
|---|---|---|---|
| `kaggle_t4x2` | 2×T4 | 8-15 min | ~$0.15 |
| `a100_micro_probe` | A100-40GB | 4-8 min | ~$0.25 |
| `full` | A100-40GB | **90-150 min** (see §3) | ~$5-9 |

```bash
modal run scripts/modal_app.py --stage kaggle_t4x2      # needs Task 10
modal run scripts/modal_app.py --stage micro_probe
# only then:
modal run scripts/modal_app.py --stage full --test-path private-official.json
```

Reports land in `artifacts/gates/$CID/` automatically
(`modal_app.py:735-746`).

---

## 2. Read the micro-probe before spending on the full stage

The probe exists to produce the numbers the full stage's deadline math depends on
(B-04). Check all four:

| check | expected | if it fails |
|---|---|---|
| `torch.cuda.get_device_name(0)` | contains `A100` | wrong GPU class — stop |
| peak train VRAM | **< 22 GB** of 40 | batch 4 × accum 2 is wrong for this image |
| `measured_tokens_per_second` | **> 400** aggregate | bf16 path (Task 1) did not take effect — **stop** |
| `strict_reload` | `pass` | adapter is corrupt; a full run would waste an hour |

> A decode throughput near ~100 tok/s means the model still loaded in NF4. That single
> number is the difference between a 30-minute inference stage and a 6-hour one. **Do not
> proceed on a failing throughput check** — that is exactly the mistake that burned the
> previous container.

---

## 3. Expected wall clock for the full stage

Derived from [M-1](00-evidence-and-measurements.md#m-1--dataset-shape), [M-2](00-evidence-and-measurements.md#m-2--gold-answer-length-real-qwen-tokenizer), [M-8](00-evidence-and-measurements.md#m-8--vram-budget-bf16-inference-on-a100-40gb) and the fixes in Tasks 1-6.

**Training.** ~4,372 deduplicated evidence-bearing examples (Task 4 + Q-7) ÷ effective
batch 8 = **~547 optimizer steps per epoch**. With `group_by_length` (Task 3) removing most
padding: **~20-30 min per epoch**. Under the max-score directive, budget **2-3 epochs →
40-90 min** (Task 13 decides the count by measured held-out METEOR, not by guesswork).

**Inference.** 1,918 questions − 81 known-QA overrides ([M-7](00-evidence-and-measurements.md#m-7--known-qa-override-coverage-on-the-private-test)) ≈ **1,837 generations**.
Mean generated length ~500 tokens — most sequences hit EOS well before the 1536 cap, so the
raised ceiling costs far less than 3× — → **~0.9-1.1 M tokens**.

| decode throughput | inference time |
|---|---|
| 500 tok/s (pessimistic bf16) | ~31 min |
| 1,200 tok/s (expected, batch 32-48) | ~13 min |
| ~100 tok/s (NF4 — the bug) | **~2.5 h** ❌ |

| stage | expected |
|---|---|
| image build + dataset pull (cold) | 8-15 min |
| dense index rebuild (**always cold the first time** — the shipped dataset has no dense index, [M-12](00-evidence-and-measurements.md#m-12--the-local-dek21-dense-index-is-a-mock-index-of-random-vectors)) | 15-25 min cold, 0 cached |
| training (2-3 epochs) | 40-90 min |
| inference @1536 cap | 20-45 min |
| bundle + HF upload | 3-6 min |
| **total (warm volume)** | **~90-150 min** |

Still comfortably inside the 17,100 s graceful budget — the speed work in Tasks 1-6 is what
creates the room to spend on extra epochs and a higher token ceiling.

Container timeout is 18,000 s (`modal_app.py:380`) with a graceful budget of 17,100 s
(`modal_app.py:195`). A healthy run finishes at roughly **⅓ of budget**.

---

## 4. Abort criteria — kill the app if any of these appear

```bash
modal app stop -y <APP_ID>
```

| signal | meaning |
|---|---|
| No new log line for **> 20 min** during inference | resume cache not flushing (Task 6 regression) |
| `[Inference] Generated …` advancing **< 200 queries / 10 min** | decode throughput collapsed → NF4 path |
| `PARAMETER BUDGET EXCEEDED` | stop; `audit_parameters` is correct and the stack is not |
| `completion_only_loss=True but the collator masked 0 tokens` | Task 4's assertion fired — TRL is not masking |
| `refusing release: trainer manifest lacks measured optimizer_steps` | telemetry missing; the bundle would be unevidenced |
| elapsed > **3.5 h** with no submission stage | deadline math wrong again; stop and re-derive |

Stopping is cheap; a killed container with an uncommitted volume is not. When in doubt, stop.

---

## 5. After the run

```bash
# 1. Submission arrives automatically at repo root and under artifacts/
ls -la submission.json.zip artifacts/submissions/$CID/

# 2. Verify the zip against the loose file (scorer reads the inner member)
.venv311/bin/python -c "
from src.task2.scorer_contract import verify_zip_inner_matches_loose
import json
print(json.dumps(verify_zip_inner_matches_loose('submission.json.zip', 'submission.json'), indent=2))
"

# 3. Sanity-check the answer distribution BEFORE uploading
.venv311/bin/python -c "
import json, zipfile, numpy as np
with zipfile.ZipFile('submission.json.zip') as z:
    d = json.loads(z.read('submission.json'))
L = np.array([len(v['answer'].split()) for v in d.values()])
print(f'n={len(d)}  mean={L.mean():.0f}  median={np.median(L):.0f}  p90={np.percentile(L,90):.0f}')
print('empty:', int((L == 0).sum()))
assert len(d) == 1918, f'expected 1918 answers, got {len(d)}'
assert (L == 0).sum() == 0, 'empty answers present — these score 0'
"
```

Expected shape if `dual_assembled` won Task 7's sweep: **mean ~470-870 words, zero empties.**
A mean near 300 means the citation block is not being appended; a mean near 1,200 means
more than one article is being stitched, which [M-5](00-evidence-and-measurements.md#m-5--multi-article-stitching-under-imperfect-retrieval-n--150) shows *loses* score.

Then score it locally against a held-out reference before trusting the leaderboard:

```bash
.venv311/bin/python scripts/score_official.py --pred submission.json --ref <held_out_reference.json>
```

---

## 6. Cost discipline

- One `modal app stop -y <id>` is cheaper than any hour of "let's see if it recovers".
- `modal app list` after every session; confirm **0 running**.
- The data and runs volumes persist, so a second attempt skips the 15-25 min cold index
  build. Never delete them to "start clean" without a reason.
