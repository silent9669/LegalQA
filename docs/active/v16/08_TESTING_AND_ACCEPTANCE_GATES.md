# Testing and Acceptance Gates

## Philosophy

No single test proves the pipeline is production-ready.

Use four layers:

```text
unit contracts
   ↓
CPU/integration CI
   ↓
release/package audit
   ↓
manual Kaggle GPU gates
```

## Gate 0 — data inheritance

Must pass before runtime work is trusted.

Checks:

- core SHA256 matches;
- legal chunks = 801,863;
- public queries = 1,000;
- BM25 layout complete;
- DEk21 matrix shape/dtype correct;
- fold assignment present;
- no regenerated artifact.

## Gate 1 — unit tests

### Generation config

Assert:

```text
Qwen2.5-3B
2048
r16/a32
batch1
gradaccum8
activation offloading
Liger fused CE
cuda:0
```

### Dataset

Assert:

- answer full;
- evidence trimmed first;
- deterministic fold exclusion;
- worst-case sample selection works;
- no silent empty answer.

### Liger backend

Assert:

- exact version;
- selective kernel flags;
- no `chunked_nll`;
- configuration unsupported -> fail loud.

### Memory policy

Mock CUDA and assert:

- both devices cleaned between stages;
- telemetry keys stable;
- callback frequency controlled.

## Gate 2 — integration/contract tests

### Runtime release

- API16 everywhere.
- stale API15 manifest rejected.
- dataset/root/nested code manifest parity.

### Notebook

Committed probe profile must be one of the explicitly allowed profiles, initially:

```text
generator_probe_worstcase
```

No:

```text
ALLOW_UNVALIDATED_FINAL=True
ALLOW_SINGLE_GPU_SMOKE=True
```

### Generator strict reload

The saved adapter must be loadable with:

```text
require_adapter=True
fail_on_fallback=True
4-bit base model
```

### Official scoring

Keep a test that compares internal METEOR implementation with the bundled official scoring program semantics.

## Gate 3 — dependency compatibility

CI lanes:

```text
Python 3.10 SFT API
Python 3.12 SFT API
protected bootstrap preservation
CPU unit/integration suite
```

Optional GPU tests are marked and skipped in CPU CI.

## Gate 4 — release package

Before uploading:

```text
clean git tree
exact final HEAD
API16 tripartite manifest parity
inherited data hashes unchanged
no secret files
no __pycache__
no local .venv
no raw token values
requirements include exact Liger pin
```

After upload, verify the remote dataset again.

## Gate 5 — worst-case T4 probe

This is the first authoritative GPU acceptance test.

Profile:

```text
generator_probe_worstcase
```

Data selection:

```text
top 12 total-token examples
+ top 12 completion-token examples
deduplicated
```

Training:

```text
3 optimizer steps
batch1
gradaccum8
2048
```

PASS only if:

```text
no OOM
finite loss
no fallback
trainer_n_gpu=1
Liger fused CE confirmed active
adapter saved
strict reload pass
nonempty generation
```

## Gate 6 — 30-step T4 endurance

Profile:

```text
generator_probe_endurance
```

PASS only if:

```text
30 optimizer steps
no OOM
no NaN/Inf
memory does not show clear unbounded growth
strict reload pass
```

Record:

- seconds/step;
- peak allocated;
- peak reserved;
- free VRAM;
- total elapsed.

Use this to estimate full-epoch time.

## Gate 7 — Protocol-8

Only after both generator probes pass.

Required systems:

```text
R0G0
R1G0
R_SELECTED_G1
```

Required outputs:

```text
promotion_report.json
promoted_production_selection.yaml
screen_handoff.zip
```

Promotion must use official METEOR.

## Gate 8 — final train

Only a `PROMOTED` config can run final training.

No emergency unvalidated override for the score-max attempt.

## Stop rules

Immediately stop and investigate when:

- exact API/SHA mismatch;
- data hash mismatch;
- Liger not active when expected;
- fallback generator used;
- first OOM;
- NaN/Inf loss;
- memory grows step after step;
- checkpoint reload fails;
- Protocol-8 report incomplete;
- production config remains `UNVALIDATED`.
