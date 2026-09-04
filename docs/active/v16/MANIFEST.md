# Refresh Pack Manifest

## Pack identity

```text
Name: LegalQA_V16_Fresh_Start_Refresh_Pack
Created: 2026-09-04
V15 baseline: 151313fc3126615ec11c08ca68f154d5b0c5406f
Inherited runtime API: 15
Target runtime API: 16
Primary new backend: Liger fused-linear cross entropy
Pinned Liger target: 0.8.2
```

## Evidence sources used

- `silent9669/LegalQA` V15 HEAD `151313fc3126615ec11c08ca68f154d5b0c5406f`
- real Kaggle log `legalqa-training-5.log`
- verified Drive V15 package rooted at the user-provided folder
- V15 `dataset_manifest.json`
- V15 root/nested `code_manifest.json`
- V15 `requirements-kaggle.txt`
- repository official `Scoring-Program-Task-LegalQA/scoring.py`
- Hugging Face TRL chunked-CE implementation
- Hugging Face Transformers Liger integration
- Liger-Kernel Qwen2/Qwen2.5 support and v0.8.2 release metadata

## Pack files

The ZIP contains all active Markdown documents for the refresh plus a SHA256 checksum file.

This pack does not contain private model weights, user secrets, inherited dataset binaries, or font/binary dependencies.
