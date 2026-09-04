# Fresh Workspace Structure

## Principle

This is a **clean runtime refresh**, not a blind rewrite of proven retrieval/evaluation code.

The new workspace should isolate:

- inherited immutable data;
- runtime/release contracts;
- generation training;
- evaluation/promotion;
- Kaggle entrypoints;
- historical material.

Avoid a single 30k-line training/orchestration file.

## Target repository tree

```text
LegalQA/
├── README.md
├── .gitignore
├── pytest.ini
│
├── configs/
│   ├── runtime_api.yaml
│   ├── models.yaml
│   ├── production_selection.yaml
│   ├── task2.yaml
│   ├── training_v16.yaml              # NEW: generator/reranker/probe settings
│   └── experiments.yaml
│
├── requirements/
│   ├── base.txt                       # NEW: CPU/common
│   ├── kaggle.txt                     # NEW: user-space Kaggle dependencies
│   └── gpu-test.txt                   # NEW: optional GPU validation deps
│
├── src/
│   ├── common/                        # INHERIT proven utilities
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── dense_dek21.py
│   │   ├── reranker.py
│   │   └── ...
│   │
│   └── task2/
│       ├── __init__.py
│       ├── data_contract.py           # NEW: inherited artifact verification
│       ├── candidates.py              # INHERIT
│       ├── evidence_packer.py         # INHERIT
│       ├── metrics.py                 # INHERIT official-equivalent scoring helpers
│       ├── predict.py                 # retain compatibility until new runner proven
│       │
│       ├── generation/                # NEW CLEAN GENERATOR BOUNDARY
│       │   ├── __init__.py
│       │   ├── config.py              # dataclass/config validation
│       │   ├── dataset.py             # answer-preserving SFT builder
│       │   ├── memory.py              # VRAM telemetry / cleanup
│       │   ├── liger_backend.py       # selective Liger config and assertions
│       │   ├── trainer.py             # QLoRA training entrypoint
│       │   └── inference.py           # strict adapter reload/inference
│       │
│       ├── reranking/
│       │   ├── __init__.py
│       │   └── trainer.py             # thin wrapper around proven reranker trainer
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── protocol8.py           # extracted from current evaluation flow
│       │   └── promotion.py           # deterministic production promotion
│       │
│       ├── runtime/
│       │   ├── __init__.py
│       │   ├── hardware.py            # dual-T4 policy
│       │   ├── environment.py         # dependency/runtime checks
│       │   └── release.py             # API/SHA manifest contract
│       │
│       └── pipeline/
│           ├── __init__.py
│           ├── profiles.py            # generator_probe_worstcase, endurance, screen, final
│           └── runner.py              # orchestration only
│
├── scripts/
│   ├── verify_inherited_dataset.py    # NEW
│   ├── verify_release.py              # NEW or extracted from preflight
│   ├── run_generator_probe.py         # NEW
│   ├── run_screen_fold0.py            # NEW
│   ├── run_final_train.py             # NEW
│   ├── package_kaggle_dataset.py      # INHERIT, update API16 package content
│   ├── preflight_kaggle.py            # INHERIT, simplify into reusable calls
│   └── promote_production_selection.py
│
├── kaggle_kernel/
│   ├── legalqa_gpu_pipeline.ipynb     # THIN notebook, calls scripts/modules
│   └── kernel-metadata.json
│
├── tests/
│   ├── unit/
│   │   ├── test_generation_config.py
│   │   ├── test_sft_dataset.py
│   │   ├── test_liger_backend.py
│   │   └── test_memory_policy.py
│   ├── contracts/
│   │   ├── test_runtime_api16.py
│   │   ├── test_dataset_inheritance.py
│   │   ├── test_notebook_profiles.py
│   │   └── test_dependency_lock.py
│   ├── integration/
│   │   ├── test_generator_reload.py
│   │   ├── test_protocol8_contract.py
│   │   └── test_release_packaging.py
│   └── gpu/
│       └── test_liger_qlora_gpu.py     # skipped when CUDA unavailable
│
├── artifacts/
│   ├── inherited/
│   │   └── dataset_manifest_v15.json  # small provenance copy only
│   ├── checkpoints/
│   ├── reports/
│   ├── submissions/
│   └── runtime/
│
├── kaggle_dataset/
│   └── staged/                        # generated, never authoritative source
│
├── docs/
│   ├── active/
│   │   └── v16/                       # this refresh pack
│   └── archive/
│       └── README.md                  # marks V7–V15 docs historical
│
└── Scoring-Program-Task-LegalQA/      # INHERIT untouched
```

## Migration strategy

Do not rename every stable module in one commit.

### Stage 1

Create the new generation/runtime/profile modules and keep existing imports working.

### Stage 2

Make `src/task2/training/train_generator.py` a compatibility wrapper that calls the new `src.task2.generation.trainer`.

### Stage 3

Move notebook orchestration into `src.task2.pipeline.runner`.

### Stage 4

After CI and Kaggle generator probes pass, remove duplicated V14/V15-specific chunk-guard logic from the active path.

### Stage 5

Archive old fix documents conceptually under `docs/archive`; do not delete Git history.

## Notebook rule

The notebook should become a thin launcher:

```text
Cell 1: constants/profile
Cell 2: secret/hardware initialization
Cell 3: runtime + package validation
Cell 4: dependency bootstrap
Cell 5: preflight
Cell 6: call pipeline runner
Cell 7: export handoff/results
```

Training algorithms should not live in notebook cells.
