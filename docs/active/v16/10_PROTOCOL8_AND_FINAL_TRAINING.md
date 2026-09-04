# Protocol-8 and Final Training

## Why screening remains mandatory

The pipeline should not assume that a trained reranker or QLoRA generator improves official METEOR.

Protocol-8 makes the final component decision from held-out evidence.

## Screen systems

```text
R0G0
base reranker + base generator/candidate system

R1G0
task-tuned reranker + base generator/candidate system

R_SELECTED_G1
selected reranker + QLoRA generator/candidate system
```

The screen determines:

1. whether tuned reranking helps;
2. whether the generator helps;
3. which candidate policy is deployable;
4. which exact configuration is promoted.

## Fold discipline

For `screen_fold0`:

```text
fold 0 = validation
training = all other folds
```

No fold-0 examples may leak into:

- reranker training;
- QLoRA training;
- model-selection fitting.

## Metrics

Primary:

```text
official whitespace-tokenized METEOR
```

Secondary:

```text
ROUGE-L
```

Always use the repository's bundled official scorer semantics for final comparisons.

## Promotion artifact

Required report fields include:

```text
screen_protocol_version = 8
evaluated_systems includes R0G0, R1G0, R_SELECTED_G1
overall_deployable_winner present
overall_deployable_meteor present
```

The promoter writes:

```text
status: PROMOTED
```

and provenance hashes.

No hand-edited winner.

## Final training rules

After promotion:

### Reranker

If promoted:

```text
train on all allowed Task2 training pairs
val_fold = None
is_final_checkpoint = true
```

If not promoted, use the promoted base configuration.

### Generator

If promoted:

```text
Qwen2.5-3B-Instruct
NF4
LoRA r16/a32
2048
Liger fused-linear CE
all allowed Task2 training examples
val_fold = None
is_final_checkpoint = true
```

If QLoRA is not promoted, do not spend hours training it for final submission.

## Final inference

Use exactly the promoted candidate policy.

Do not infer `REQUIRES_GENERATOR` from an old unvalidated default.

## Submission validation

Require:

```text
1,000 IDs
same ID set as public-official
each value object has non-empty answer
no NaN/null
UTF-8 JSON
no extra debug fields
```

Run local scorer-compatible sanity checks before submission packaging.

## Reproducibility bundle

Save:

```text
final Git SHA
Runtime API
dataset manifest SHA
production config hash
reranker manifest
generator manifest
environment manifest
submission hash
```

This is the minimum needed to reproduce the final score path.
