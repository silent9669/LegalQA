#!/usr/bin/env bash
# ==============================================================================
# LegalQA Task 2 — Google Colab Automated Training Launcher (Notion DSC 2026)
# ==============================================================================
# Usage:
#   ./run_colab_a100.sh              # Production run on NVIDIA A100
#   ./run_colab_a100.sh --gpu T4     # Verification / smoke run on Tesla T4
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN=""
if [ -x ".venv-ml/bin/python" ]; then
    PYTHON_BIN=".venv-ml/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

exec "$PYTHON_BIN" scripts/launch_colab_training.py "$@"
