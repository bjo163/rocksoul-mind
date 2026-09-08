#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-jingyaogong/minimind-3}"
THREADS="${THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc || echo 1)}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/cpu/production}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

mkdir -p "$OUTPUT_DIR"

python runtime/cpu/system_info.py > "$OUTPUT_DIR/system-info.json"
python runtime/cpu/benchmark.py \
  --model "$MODEL" \
  --threads "$THREADS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output "$OUTPUT_DIR/core.json"
python runtime/cpu/context_benchmark.py \
  --model "$MODEL" \
  --threads "$THREADS" \
  --contexts 128 512 1024 2048 \
  --output "$OUTPUT_DIR/context.json"

python - "$OUTPUT_DIR" "$MODEL" "$THREADS" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

out_dir, model, threads = sys.argv[1:]
manifest = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": model,
    "threads": int(threads),
    "files": {
        "system_info": "system-info.json",
        "core": "core.json",
        "context": "context.json",
    },
}
with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY

echo "Phase 0 CPU evidence written to: $OUTPUT_DIR"
