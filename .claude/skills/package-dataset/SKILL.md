---
name: package-dataset
description: Package Kaggle dataset bundle and verify tripartite SHA manifest parity
disable-model-invocation: true
---

# Kaggle Dataset Packaging Workflow

Use this skill when preparing the staged dataset bundle for upload to Kaggle (`phucdangg/legalqa-task2-clean-data`).

## Packaging Steps:

1. **Clean Prior Staging State (if required)**:
   Verify or clean `kaggle_dataset/staged` to prevent stale artifacts from leaking into the archive.

2. **Run Packaging Script**:
   Execute packaging with the production profile:
   ```bash
   ./.venv/bin/python scripts/package_kaggle_dataset.py --profile final_training
   ```

3. **Verify Manifest Parity & SHA Integrity**:
   Run packaging validation unit tests:
   ```bash
   ./.venv/bin/pytest tests/test_production_config_and_manifest.py tests/test_kaggle_packaging.py -v
   ```

4. **Tripartite SHA Checksum Audit**:
   Confirm that SHA-256 digests match across all three manifests:
   - `kaggle_dataset/staged/dataset_manifest.json`
   - `kaggle_dataset/staged/code_manifest.json`
   - `kaggle_dataset/staged/code/LegalQA/code_manifest.json`

5. **Summarize Deliverable**:
   Output:
   - Staged dataset directory path and total byte size.
   - Core archive file hash (SHA-256).
   - Confirmation of zero untracked or orphan dependencies.
