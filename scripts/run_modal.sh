#!/usr/bin/env bash
# One-click execution launcher for Modal A100 LegalQA Task 2
# Designed for seamless cross-machine & teammate handoff
set -e

STAGE="${1:-micro_probe}"
TEST_PATH="${2:-private-official.json}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 1. Resolve Modal CLI and Python executables
if [ -f "$ROOT_DIR/.venv311/bin/modal" ] && "$ROOT_DIR/.venv311/bin/modal" --version &>/dev/null; then
    MODAL_CMD="$ROOT_DIR/.venv311/bin/modal"
    PY_CMD="$ROOT_DIR/.venv311/bin/python"
elif command -v modal &>/dev/null; then
    MODAL_CMD="modal"
    PY_CMD="python3"
elif python3 -m modal --version &>/dev/null; then
    MODAL_CMD="python3 -m modal"
    PY_CMD="python3"
else
    echo "======================================================================="
    echo "[!] Modal CLI not found in environment."
    echo "    Please install it with: pip install modal"
    echo "    Then authenticate with: modal setup"
    echo "======================================================================="
    exit 1
fi

# 2. Check Modal Authentication
echo "[1/4] Checking Modal Authentication..."
CURRENT_PROFILE=$($MODAL_CMD profile current 2>/dev/null || echo "")
if [ -z "$CURRENT_PROFILE" ]; then
    echo "[!] Not authenticated to Modal! Please run: modal setup"
    exit 1
fi
echo "      Active Modal Profile: $CURRENT_PROFILE"

# 3. Bootstrap Secrets & Volumes from .env if missing in teammate's workspace
echo "[2/4] Ensuring Modal Secrets & Volumes in workspace '$CURRENT_PROFILE'..."
$PY_CMD -c "
import os
import subprocess
from pathlib import Path
from dotenv import dotenv_values

env_p = Path('.env')
if env_p.is_file():
    vals = dotenv_values(env_p)
    k_user = vals.get('KAGGLE_USERNAME')
    k_key = vals.get('KAGGLE_KEY')
    hf_tok = vals.get('HF_TOKEN')

    try:
        sec_out = subprocess.check_output(['$MODAL_CMD', 'secret', 'list'], text=True)
    except Exception:
        sec_out = ''

    if 'kaggle-secret' not in sec_out and k_user and k_key:
        print('      Creating missing kaggle-secret on Modal...')
        subprocess.run(['$MODAL_CMD', 'secret', 'create', 'kaggle-secret', f'KAGGLE_USERNAME={k_user}', f'KAGGLE_KEY={k_key}'], check=True)

    if 'huggingface-secret' not in sec_out and hf_tok:
        print('      Creating missing huggingface-secret on Modal...')
        subprocess.run(['$MODAL_CMD', 'secret', 'create', 'huggingface-secret', f'HF_TOKEN={hf_tok}'], check=True)

    try:
        vol_out = subprocess.check_output(['$MODAL_CMD', 'volume', 'list'], text=True)
    except Exception:
        vol_out = ''
    for vol in ['legalqa-data-vol', 'legalqa-runs-vol']:
        if vol not in vol_out:
            print(f'      Creating missing volume {vol} on Modal...')
            subprocess.run(['$MODAL_CMD', 'volume', 'create', vol], check=True)
" 2>/dev/null || true

# 4. Auto-resolve DAG prerequisites if parent reports are missing
CAND_ID=$($PY_CMD -c "
import json
from pathlib import Path
cands = sorted(Path('artifacts/candidates').glob('*/candidate_manifest.json'), key=lambda p: p.stat().st_mtime)
cands_with_gates = [p for p in cands if (Path('artifacts/gates') / p.parent.name / 'kaggle_t4x2_report.json').is_file()]
pick = cands_with_gates[-1] if cands_with_gates else (cands[-1] if cands else None)
if pick:
    print(json.loads(pick.read_text())['candidate_id'])
" 2>/dev/null || echo "")

if [ -n "$CAND_ID" ]; then
    MICRO_REPORT="artifacts/gates/$CAND_ID/a100_micro_probe_report.json"

    if [ "$STAGE" = "full" ]; then
        if [ ! -f "$MICRO_REPORT" ]; then
            echo "[*] a100_micro_probe_report.json missing for candidate $CAND_ID."
            echo "    Automatically running required DAG stage 'micro_probe' on Modal A100 (~1.5 min)..."
            $MODAL_CMD run scripts/modal_app.py --stage "micro_probe"
            echo "[+] a100_micro_probe stage completed and registered locally at $MICRO_REPORT!"
        fi
    fi
fi

# 5. Dispatch Modal Pipeline
echo "[3/4] Ready: stage=$STAGE | test=$TEST_PATH"
echo "[4/4] Dispatching to remote Modal container..."
echo "======================================================================="

$MODAL_CMD run scripts/modal_app.py --stage "$STAGE" --test-path "$TEST_PATH"
