#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python 3.10+ is required." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f"Python 3.10-3.12 required, found {sys.version.split()[0]}")
PY

VENV="${VENV:-.venv-cpu}"
if [[ ! -d "$VENV" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r runtime/cpu/requirements.txt

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"

bash runtime/cpu/run_phase0.sh

echo
echo "RockSoul Mind Phase 0 bootstrap complete."
echo "Evidence: ${OUTPUT_DIR:-benchmarks/cpu/production}"
