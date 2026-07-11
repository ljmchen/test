#!/usr/bin/env bash
# Full eval3 + stat3 for the e0115 predictions (run on the 230/Alset node).
set -euo pipefail
cd /home/jiaxuan/slai/dgbench-pami
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
PY=/mnt/conda/jiaxuan/miniconda3/envs/DGBench/bin/python
EXP=dexvlg_e0115
SR=/home/jiaxuan/slai/dgbench-pami/output
"$PY" src/main.py task=eval3 setting=tabletop hand=shadow_left save_root="$SR" exp_name="$EXP" task.max_num=-1 n_worker=160 skip=True
"$PY" src/main.py task=stat3 setting=tabletop hand=shadow_left save_root="$SR" exp_name="$EXP"
echo "ALL_DONE_e0115"
