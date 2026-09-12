#!/bin/bash
set -e

# LegalQA Task 2 Main Test Execution Gate
# Executes syntax, schema, notebooks, and full test suite

PYTHON_BIN="python3"
if [ -x ".venv311/bin/python" ]; then
    PYTHON_BIN=".venv311/bin/python"
elif [ -x ".venv-ml/bin/python" ]; then
    PYTHON_BIN=".venv-ml/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

exec $PYTHON_BIN scripts/pre_push_check.py "$@"
