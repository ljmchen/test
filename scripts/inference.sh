#!/usr/bin/env bash
# DexVLG dataset inference -> predictions JSON.
#
# Activate the dexvlg env first, or pass PYTHON_BIN=/path/to/envs/dexvlg/bin/python.
# Usage:
#   bash scripts/inference.sh \
#     --config configs/default.yaml \
#     --checkpoint outputs/checkpoints/best_eXXXX_sX.XXXX.pt \
#     --output outputs/predictions_test.json \
#     --split val --num-steps 10 --batch-size 256
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

source /mnt/afs/L202500241/miniconda/bin/activate dexvlg

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" infer_dataset.py "$@"
