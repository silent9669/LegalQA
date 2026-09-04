# Kaggle Dual-T4 Runbook

## Ownership

Only the user starts or configures Kaggle GPU runs.

The coding agent may prepare code/dataset versions and verify metadata but must not trigger a GPU notebook.

## Common Kaggle settings

```text
Accelerator: T4 x2
Internet: On
HF_TOKEN: enabled in Kaggle Secrets
Qwen model input:
qwen-lm/qwen2.5/transformers/3b-instruct/1
Dataset:
latest verified LegalQA API16 package
```

Restart the session after replacing notebook/dataset versions.

## Run A — worst-case generator probe

Notebook profile:

```text
generator_probe_worstcase
```

Expected startup:

```text
COMMITTED EXECUTION PROFILE: generator_probe_worstcase
CUDA GPUs Detected: 2
GPU0 Tesla T4
GPU1 Tesla T4
Runtime API v16 verified
Generator cuda:0
Retrieval/Reranker cuda:1
```

Expected dependency block:

```text
TRL 1.12.0
Liger 0.8.2
Transformers 5.0.0
PEFT 0.19.1
bitsandbytes 0.50.2
```

Expected generator block:

```text
Full fold-filtered source pool built
Worst-case probe selected
Liger selective config verified
fused_linear_cross_entropy=True
chunked_nll disabled
trainer_n_gpu=1
```

Successful end:

```text
3 optimizer steps complete
adapter saved
strict reload pass
generation pass
```

Download the generator manifest/log.

## Run B — endurance

Only change:

```text
EXECUTION_PROFILE = generator_probe_endurance
```

Keep API16 if runtime code did not change.

Expected:

```text
max_steps=30
max_train_examples=None
```

Successful end includes memory/timing telemetry.

## Run C — Protocol-8 screen

Only after Run A+B PASS.

Profile:

```text
screen_fold0
```

This run may take hours because it includes actual reranker/generator training and held-out evaluation.

Expected final outputs:

```text
/kaggle/working/promotion_report.json
/kaggle/working/promoted_production_selection.yaml
/kaggle/working/screen_handoff.zip
```

Download `screen_handoff.zip` immediately.

Do not proceed to final training in the same logical selection step.

## Run D — final training

After the promoted config is audited and committed:

```text
EXECUTION_PROFILE = final_train_and_submit
ALLOW_UNVALIDATED_FINAL = False
```

Train on all allowed Task2 training data according to the promoted component decisions.

Expected outputs:

```text
final reranker checkpoint if promoted
final QLoRA adapter if promoted
submission.json
runtime/release manifests
```

## Log capture rule

On failure, preserve:

- 100 lines before the first traceback;
- complete traceback;
- environment/version header;
- memory diagnostics;
- execution profile;
- API/SHA header.

Do not ask an agent to guess a fix from only the final exception line.

## Kaggle time-saving rule

Never repeat an expensive stage merely to test a downstream hypothesis.

Examples:

```text
generator memory bug -> generator-only probe
reranker bug -> reranker-only probe
screen selection bug -> use existing verified checkpoints when provenance permits
packaging bug -> CPU/static validation first
```
