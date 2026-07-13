#!/usr/bin/env bash
# DexVLG dataset inference -> predictions JSON.
#
# Uses the local dexvlg env python by default; override with PYTHON_BIN=/path/to/python.
# Usage:
#   bash scripts/inference.sh \
#     --config configs/default.yaml \
#     --checkpoint outputs/checkpoints/best_eXXXX_sX.XXXX.pt \
#     --output outputs/predictions_test.json \
#     --split val --num-steps 10 --batch-size 256
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda/jiaxuan/miniconda3/envs/dexvlg/bin/python}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" infer_dataset.py "$@"
