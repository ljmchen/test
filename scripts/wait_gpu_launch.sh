#!/usr/bin/env bash
# 空闲卡守卫：轮询 nvidia-smi，某卡（排除 --exclude 列表）连续 3 次（间隔 60s）
# 满足 mem<1000MiB 且 util<10% 即认定空闲，用它 tmux 启动指定训练后自我退出。
#
# 用法: bash scripts/wait_gpu_launch.sh --config configs/p3a_rot_seq.yaml \
#         --session p3a_rot_seq [--exclude 0] [--log PATH]
set -uo pipefail
TEST=/home/jiaxuan/exp-pami/test
PY=/mnt/conda/jiaxuan/miniconda3/envs/dexvlg/bin/python
CONFIG=""; SESSION=""; EXCLUDE="0"; LOGF=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2;;
    --session) SESSION=$2; shift 2;;
    --exclude) EXCLUDE=$2; shift 2;;
    --log) LOGF=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
[ -z "$CONFIG" ] || [ -z "$SESSION" ] && { echo "必填 --config --session"; exit 2; }
LOGF=${LOGF:-$TEST/outputs/wait_gpu_launch_${SESSION}.log}
DRY=${DRY:-0}
log() { echo "$(date '+%F %T') $*" | tee -a "$LOGF"; }

declare -A streak
log "守卫启动: config=$CONFIG session=$SESSION exclude=[$EXCLUDE] dry=$DRY (host=$(hostname))"
while true; do
  picked=""
  while IFS=', ' read -r idx mem util; do
    skip=0
    for e in $(echo "$EXCLUDE" | tr ',' ' '); do [ "$idx" = "$e" ] && skip=1; done
    [ $skip -eq 1 ] && { streak[$idx]=0; continue; }
    if [ "$mem" -lt 1000 ] && [ "$util" -lt 10 ]; then
      streak[$idx]=$(( ${streak[$idx]:-0} + 1 ))
      if [ "${streak[$idx]}" -ge 3 ]; then picked=$idx; break; fi
    else
      streak[$idx]=0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  if [ -n "$picked" ]; then
    log "GPU$picked 连续3次空闲，启动训练"
    if [ "$DRY" = "1" ]; then
      log "[dry-run] 将执行: tmux new -d -s $SESSION; CUDA_VISIBLE_DEVICES=$picked $PY train.py --config $CONFIG"
      exit 0
    fi
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION"
    tmux send-keys -t "$SESSION" "cd $TEST && CUDA_VISIBLE_DEVICES=$picked $PY train.py --config $CONFIG" Enter
    sleep 10
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      log "已启动 tmux:$SESSION @ GPU$picked，守卫退出"
      exit 0
    else
      log "tmux 启动失败，继续等待"; streak[$picked]=0
    fi
  fi
  sleep 60
done