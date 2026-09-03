---
name: preflight-check
description: Run Kaggle preflight checks, verify API 14 contracts, and run core runtime integrity tests
---

# Preflight Check Workflow

Use this skill to verify environment configuration, runtime API version contracts, and core test suites before staging, Colab smoke testing, or Kaggle submission.

## Steps:

1. **Verify Python Environment**:
   Ensure virtual environment `.venv` is active and available:
   ```bash
   ./.venv/bin/python --version
   ```

2. **Execute Preflight Script**:
   Run the preflight validation diagnostics:
   ```bash
   ./.venv/bin/python scripts/preflight_kaggle.py
   ```
   Verify parameter budget compliance and public data checks.

3. **Validate Runtime API Contracts & Integrity**:
   Run the core contract and integrity test suite:
   ```bash
   ./.venv/bin/pytest tests/test_v7_runtime_integrity.py tests/test_v10_runtime_release_binding.py tests/test_colab_smoke_contract.py -v
   ```

4. **Report Findings**:
   - Runtime API version reported (e.g. API 14).
   - Preflight check result: PASS / FAIL.
   - Pytest execution summary: tests passed / failed.
   - Ready status for dataset packaging or remote Colab/Kaggle execution.
