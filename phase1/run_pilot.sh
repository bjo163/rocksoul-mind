#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET_TOKENS="${TARGET_TOKENS:-20000000}"
ID_RATIO="${ID_RATIO:-0.70}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Phase 1 pilot requires a CUDA GPU. Use Kaggle/Colab or another GPU environment.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

python phase1/download_base.py
python phase1/prepare_iden.py \
  --target-train-tokens "$TARGET_TOKENS" \
  --id-ratio "$ID_RATIO"

bash phase1/train_iden.sh
python phase1/eval_iden.py --device cuda:0

python - <<'PY'
from pathlib import Path
result = Path("phase1/results/iden_eval.json")
print("\nRockSoul Mind Phase 1 pilot complete.")
print(f"Checkpoint: {Path('out/rocksoul_iden_pretrain_768.pth').resolve()}")
print(f"Evaluation: {result.resolve()}")
if result.exists():
    print(result.read_text(encoding="utf-8"))
PY
