#!/usr/bin/env bash
# Deduped multi-sample test inference.
#
# Runs ONE inference input per unique (obj_id, pose_id, guidance) combination
# ('guidance' also matches the 'guidence' spelling), and samples N grasps for
# each — N controlled by --samples-per-combo. Each sample is emitted as a
# separate record with its own candidate_idx (0..N-1), so prepare_bench_data.py
# groups the N samples as N variants of the same object/pose.
#
# Usage:
#   bash scripts/inference2.sh \
#     --config configs/default.yaml \
#     --checkpoint outputs/checkpoints/best_eXXXX_sX.XXXX.pt \
#     --output outputs/predictions_test.json \
#     --split test --num-steps 10 --batch-size 256 \
#     --samples-per-combo 10
#
# Override the dedup keys by passing your own --dedup-by after the script's
# default (last one wins), e.g.  --dedup-by "obj_id,pose_id,action_id,guidance".
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

source /mnt/afs/L202500241/miniconda/bin/activate dexvlg

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" infer_dataset.py --dedup-by "obj_id,pose_id,guidance" "$@"
