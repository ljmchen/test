#!/usr/bin/env bash
# Predictions JSON -> pre/squeeze -> per-task_type dgbench graspdata
# (channels: left/right/lgbidex/bidex, each writes <exp-name>_<channel>_<hand_suffix>).
#
# Only needs numpy; any env works (default PYTHON_BIN=python).
# Usage:
#   bash scripts/prepare_bench.sh \
#     --input outputs_v3/predictions_test.json \
#     --mesh-root /home/jiaxuan/data/oakink_obj/processed_data \
#     --output-root /home/jiaxuan/slai/dgbench-pami/output \
#     --exp-name dexvlg_v3 --channels left,right,lgbidex,bidex --clean --numworker 16
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" prepare_bench_data.py "$@"
