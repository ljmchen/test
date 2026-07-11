#!/usr/bin/env bash
# Full eval3 + stat3 for an already-converted graspdata exp (run on 230/Alset).
# Usage: run_eval.sh <exp_name> [n_worker]
set -euo pipefail
EXP="${1:?usage: run_eval.sh <exp_name> [n_worker]}"
NW="${2:-120}"
cd /home/jiaxuan/slai/dgbench-pami
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
PY=/mnt/conda/jiaxuan/miniconda3/envs/DGBench/bin/python
SR=/home/jiaxuan/slai/dgbench-pami/output
"$PY" src/main.py task=eval3 setting=tabletop hand=shadow_left save_root="$SR" exp_name="$EXP" task.max_num=-1 n_worker="$NW" skip=True
"$PY" src/main.py task=stat3 setting=tabletop hand=shadow_left save_root="$SR" exp_name="$EXP"
echo "ALL_DONE_${EXP}"
