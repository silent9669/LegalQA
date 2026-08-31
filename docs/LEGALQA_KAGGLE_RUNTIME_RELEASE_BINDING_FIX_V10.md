# LegalQA Kaggle V10 — Runtime Release Binding Fix

## Baseline
Latest audited HEAD: `8bc7c5b441b0b3af4754dca8b20e938cb7fb1744`.
Fresh CI: run `33404561780`, all 3 jobs green, `144 passed, 1 skipped`.

## Confirmed
The latest commit fixes the exact nested Kaggle mount bug by recursively supporting:
`/kaggle/input/datasets/<owner>/<slug>`.

## Remaining blocker
Runtime API is still 9 and the notebook obtains its expected API from packaged dataset code. A stale V9 package can therefore define both the old `path_resolver` and the API used to validate itself. The notebook also calls `resolve_runtime_paths()` before manifest validation.

## V10 fix
1. Add notebook-owned literal:
```python
REQUIRED_RUNTIME_API_VERSION = 10
```
Do not derive it from packaged code.

2. After resolving the single packaged code root:
```python
resolved_code_root = resolve_packaged_code_root("/kaggle/input", strict=True)
runtime_root_from_code = str(Path(resolved_code_root).resolve().parents[1])
```

3. Validate manifests BEFORE using packaged path resolution:
```python
RUNTIME_PROVENANCE = validate_runtime_manifests(
    runtime_root=runtime_root_from_code,
    code_root=resolved_code_root,
    expected_api_version=REQUIRED_RUNTIME_API_VERSION,
)
```

4. Only then:
```python
paths = resolve_runtime_paths(
    "/kaggle/input",
    strict=True,
    allow_remote_model_download=False,
)
```

5. Cross-check:
```python
assert os.path.realpath(paths["runtime_root"]) == os.path.realpath(runtime_root_from_code)
```

6. Bump:
- `configs/runtime_api.yaml` -> API 10
- `src/task2/runtime_integrity.py` -> `EXPECTED_RUNTIME_API_VERSION = 10`

7. Add regressions:
- notebook owns literal API 10;
- manifest validation occurs before `resolve_runtime_paths`;
- matching-SHA stale API-9 package is rejected;
- API-10 package passes;
- preserve all nested-layout, ambiguity and no-root tests.

8. Keep recursive nested-root fix. Prefer `followlinks=False` unless a real Kaggle test proves links are required.

## Verification
Run:
```bash
pytest tests/test_v10_runtime_release_binding.py -v
pytest tests/test_path_resolver_kaggle_nested.py -v
pytest tests/test_v9_notebook_contract.py -v
pytest tests/test_v8_fail_loud_integration.py -v
pytest tests/ -v
```
Require all GitHub Actions jobs green on final V10 HEAD.

## Repackage
From a clean final HEAD:
```bash
python scripts/package_kaggle_dataset.py   --source artifacts/task2   --staging kaggle_dataset/staged   --profile final_training
```

Verify:
- dataset/code runtime API = 10;
- dataset/code Git SHA identical and equal final HEAD;
- SHA is real 40-char lowercase hex;
- `code/LegalQA/src/task2/path_resolver.py` exists;
- BM25 and DEk21 indexes exist.

## Kaggle deployment
Using Kaggle CLI, upload a NEW version of:
`phucdangg/legalqa-task2-clean-data`.

Then inspect/download the uploaded artifact itself and prove:
- API 10;
- dataset/code SHA == final V10 HEAD;
- nested resolver is actually uploaded;
- BM25/DEk21 are present.

Do not rely only on local staging.

## Smoke
Keep:
```python
EXECUTION_PROFILE = "smoke_only"
```
Use T4 x2, HF_TOKEN, mounted Qwen, newest V10 dataset, Restart Session -> Save & Run All.

Startup must show:
- 2 T4s;
- packaged code root;
- Runtime Release API 10;
- final V10 Git SHA;
- nested dataset root;
- manifest validation PASS.

Smoke only passes after dependency/import/index checks, 30 reranker steps + reload, 30 QLoRA steps + reload, and 5 real held-out predictions with no mock/fallback.

## Completion verdict
Return exactly `READY FOR KAGGLE SMOKE` or `BLOCKED`, with HEAD, tests/CI, Kaggle dataset version and uploaded-manifest evidence.
