# Google Colab A100 Production Training Guide

Authoritative specification for **Google Colab A100 Production Training** (`B2.2 Task 2 — Colab A100 Real Run & Evidence`).

---

## 1. Purpose

Google Colab with NVIDIA A100 GPU (40GB or 80GB VRAM) serves as the **production training environment**. A100 compute is expensive and strictly reserved for fully verified, frozen versions that have achieved `PASS` status on the Kaggle Dual-T4 smoke gate.

---

## 2. Prerequisites & Authorization Gate

Before executing the Colab A100 notebook (`notebooks/colab_a100_train.ipynb`):
1. **GitHub Commit Frozen**: The exact Git commit SHA must be checked out.
2. **Kaggle Dataset Manifest Verified**: `phucdangg/legalqa-task2-clean-data` SHA256 hashes must match.
3. **Kaggle Smoke Pass Attached**: `kaggle_smoke_report.json` must be present and report `"status": "PASS"`.

---

## 3. Production Configuration (`configs/colab_train_a100.yaml`)

Unlike the Kaggle Dual-T4 smoke profile (which used INT4 quantization to fit into 16GB VRAM), Colab A100 leverages native high-throughput training:
- **Precision**: Native `bfloat16` (`torch_dtype: bfloat16`)
- **Kernel**: Liger Fused Linear Cross-Entropy loss enabled
- **LoRA Parameters**: Rank `r=32`, Alpha `alpha=64`, Dropout `0.05`
- **Batch Size**: `per_device_train_batch_size: 4`, `gradient_accumulation_steps: 4` (effective batch size = 16)
- **Epochs**: Full training (3 epochs)
- **Checkpoints**: Saved per epoch with `save_total_limit: 2` to conserve disk

---

## 4. Execution Workflow

### Interactive Colab Execution
Run `notebooks/colab_a100_train.ipynb` cell-by-cell in Google Colab with A100 runtime:
```bash
# Cell 5 executes the production runner:
python scripts/run_pipeline.py \
  --config configs/colab_train_a100.yaml \
  --data-dir /content/data/legalqa-task2-clean-data \
  --output-dir /content/runs/current \
  --require-smoke-pass kaggle_smoke_report.json \
  --allow-single-gpu
```

### Automated Headless / Colab CLI Execution (Recommended)
Run the automated session orchestrator from your local terminal:
```bash
./scripts/launch_colab_training.py --gpu A100
```
This automatically verifies local `.env` credentials and `kaggle_smoke_report.json`, provisions the A100 VM, uploads credentials, executes `notebooks/colab_a100_train.ipynb`, uploads all artifacts to Hugging Face, and releases the VM upon completion to prevent credit leakage.

---

## 5. Artifact Packaging & Hugging Face Upload

Upon completion, the pipeline evaluates the final model against the validation folds, computes official whitespace METEOR scores, and builds the Run Bundle:

```text
/content/runs/current/
├── run_manifest.json
├── config.yaml
├── dataset_manifest.json
├── kaggle_smoke_report.json
├── environment.txt
├── nvidia-smi.txt
├── train.log
├── metrics.json
├── trainer_state.json
├── tensorboard/
└── final_adapter/
    ├── adapter_model.safetensors
    ├── adapter_config.json
    └── README.md (Model Card)
```

The notebook uploads this bundle directly to the Hugging Face model repository (`dangphuc2109/legalqa-qwen2.5-3b-adapter`).
