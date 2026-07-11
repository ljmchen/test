#!/usr/bin/env bash
# Deduped multi-sample test inference.
#
# Runs ONE inference input per unique (obj_id, scale_id, pose_id, guidance)
# combination ('guidance' also matches the 'guidence' spelling), and samples N
# grasps for each — N controlled by --samples-per-combo. Each sample is emitted
# as a separate record with its own candidate_idx (0..N-1), so
# prepare_bench_data.py groups the N samples as N variants of the same scene.
# scale_id MUST stay in the dedup key: bidex has multiple scales per
# (obj, pose, guidance) — dropping it collapses ~1/3 of bidex scenes.
#
# Usage:
#   bash scripts/inference2.sh \
#     --config outputs/<run>/config.yaml \
#     --checkpoint outputs/<run>/checkpoints/best_eXXXX_sX.XXXX.pt \
#     --output outputs/predictions_test.json \
#     --split test --num-steps 10 --batch-size 256 \
#     --samples-per-combo 10
#
# Override the dedup keys by passing your own --dedup-by after the script's
# default (last one wins).
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda/jiaxuan/miniconda3/envs/dexvlg/bin/python}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" infer_dataset.py --dedup-by "obj_id,scale_id,pose_id,guidance" "$@"
