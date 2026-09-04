# Agent Operating Rules

## Kaggle ownership

Only the user may:

- choose Kaggle accelerator;
- attach Kaggle datasets/models;
- configure Kaggle Secrets;
- trigger notebook execution;
- trigger Save & Run All;
- start GPU sessions.

Coding agents must never run Kaggle GPU notebooks.

## Allowed agent work

Agents may:

- inspect GitHub;
- create branches/worktrees;
- edit source;
- write tests;
- run CPU/CI tests;
- run optional non-Kaggle GPU probes only when explicitly requested;
- package datasets;
- upload/verify dataset versions if authorized;
- inspect remote manifests;
- prepare notebook source;
- analyze user-provided Kaggle logs.

## Secrets

Never:

```text
print HF_TOKEN
write HF_TOKEN to disk
commit .env
include token in manifest/log
echo Kaggle credentials
```

## Development method

For runtime fixes:

```text
evidence
↓
single hypothesis
↓
failing test
↓
minimal implementation
↓
focused tests
↓
full CI
↓
release package
↓
manual Kaggle probe
```

Do not combine unrelated fixes.

## Source-of-truth rule

Active V16 docs override older V7–V15 implementation notes.

Old documents are historical evidence.

## Runtime changes

Any packaged runtime behavior change requires:

- runtime API review;
- notebook API literal review;
- runtime-integrity API review;
- tripartite manifest parity;
- stale-package rejection test.

## Data changes

Do not modify inherited data without explicit user approval and a dedicated dataset-generation plan.

## Score-max rule

Do not optimize for “pipeline finishes” by silently:

- using a smaller model;
- shortening all answers;
- lowering sequence length;
- reducing LoRA capacity;
- forcing extractive fallback;
- bypassing Protocol-8.

If a quality trade-off is required, measure it and present it to the user.

## Completion language

Allowed:

```text
READY FOR USER MANUAL KAGGLE GENERATOR PROBE
READY FOR USER MANUAL KAGGLE SCREEN
READY FOR USER MANUAL FINAL TRAIN
BLOCKED
```

Do not say “fixed” or “production-ready” without the required evidence gate.
