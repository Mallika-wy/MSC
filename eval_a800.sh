#!/usr/bin/env bash
# Repeatable held-out full-motion evaluation for the A800 experiments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MSC_PYTHON="${MSC_PYTHON:-$PWD/.runs/envs/omg/bin/python}"
export PATH="$(dirname "$MSC_PYTHON"):$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SPLIT=val SEED=42
export TASK="${TASK:-text}"
export NUM_MOTIONS=10
export OUTPUT_DIR="${OUTPUT_DIR:-.runs/a800-fixed-eval-$TASK}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RTC="${RTC:-false}"
bash eval.sh "$@"
"$MSC_PYTHON" - "$OUTPUT_DIR/selection.json" ".runs/fixed-eval/$TASK.json" <<'PY'
import json, sys
from pathlib import Path
actual, expected = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
for key in ('seed', 'dataset_identity', 'motions'):
    if actual[key] != expected[key]:
        raise ValueError(f'Fixed evaluation selection changed: {key}')
print('[MSC] Fixed validation motions and per-motion seeds verified.')
PY
