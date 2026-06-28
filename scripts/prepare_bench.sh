#!/usr/bin/env bash
# Two-in-one: predictions JSON -> pre/squeeze -> dgbench-pami eval3 graspdata.
#
# Only needs numpy; any env works (default PYTHON_BIN=python).
# Usage:
#   bash scripts/prepare_bench.sh \
#     --input outputs/predictions_test.json \
#     --mesh-root /mnt/afs/L202500241/Dataset/MeshProcess/assets/object/oakink_obj/processed_data \
#     --output-root /mnt/afs/L202500241/temp/dgbench-pami/output \
#     --exp-name slaitest_RUNID_bestXX --hand-name shadow_left --numworker 16
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" prepare_bench_data.py "$@"
