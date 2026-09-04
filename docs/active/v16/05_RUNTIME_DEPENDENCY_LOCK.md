# Runtime and Dependency Lock

## Target Kaggle environment

The V16 package is designed around the environment observed in the real V15 Kaggle T4×2 run.

```text
Python                 3.12.13
PyTorch                2.10.0+cu128
CUDA                   12.8
Triton                 3.6.0
Transformers           5.0.0
Accelerate             1.13.0
Datasets               5.0.0
PEFT                   0.19.1
TRL                    1.12.0
bitsandbytes           0.50.2
sentence-transformers  5.4.1
bm25s                  0.3.11
scikit-learn           1.6.1
NLTK                   3.9.1
PyArrow                24.0.0
```

## New exact dependency

Pin:

```text
liger-kernel==0.8.2
```

Liger-Kernel v0.8.2 is the current release audited for this refresh and includes cross-entropy correctness/numerical-precision fixes.

## Kaggle requirements policy

Never install or replace:

```text
torch
torchvision
torchaudio
triton
cuda-*
nvidia-*
```

Kaggle's preinstalled GPU stack is protected.

Suggested `requirements/kaggle.txt`:

```text
transformers>=4.45.0
accelerate>=0.34.0
datasets>=2.20.0
peft>=0.10.0
trl==1.12.0
bitsandbytes>=0.43.0
liger-kernel==0.8.2
sentence-transformers>=3.0.0
bm25s>=0.2.5
scikit-learn>=1.4.0
nltk>=3.8.1
pyvi>=0.1.1
pyyaml>=6.0
pyarrow>=14.0.0
fastparquet>=2024.2.0
tqdm>=4.66.0
```

## Compatibility guard

At runtime, print and validate:

```text
torch version
CUDA runtime
GPU names / compute capability
transformers
trl
peft
bitsandbytes
liger_kernel
triton
```

For the production release:

```text
TRL must equal 1.12.0
Liger must equal 0.8.2
Transformers must expose use_liger_kernel and liger_kernel_config
SFTConfig must expose completion_only_loss and activation_offloading
```

Fail loud if Liger cannot be imported or the selective kernel config is unsupported.

## T4 caveat

Tesla T4 is compute capability 7.5. Liger v0.8.2 release notes include a guard that reserves one fused-linear CE addmm fast path for compute capability >=8.0. That must not be treated as a blocker: the implementation must prove the normal supported T4 path with the manual Kaggle probe before any full training.

## No hidden dependency upgrades

Bootstrap should use:

```bash
pip install --upgrade-strategy only-if-needed   -c <protected-constraints>   -r requirements/kaggle.txt
```

Then perform the existing baseline-aware `pip check` regression guard and protected-distribution hash/version check.

## Release lock artifact

Write:

```text
/kaggle/working/runtime_environment.json
```

with only non-secret versions/hardware metadata.

Never include environment-variable values, access tokens, or Kaggle secret content.
