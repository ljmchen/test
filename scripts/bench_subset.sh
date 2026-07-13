#!/usr/bin/env bash
# 训练期异步 bench 子集钩子 —— 由 train.py 的 training.bench_eval 每 N epoch 触发
# （subprocess.Popen detached），也可手动运行。对训练 0 影响原则：
#   1. flock 重入锁：上一次 bench 还在跑 → 直接退出（不排队）。锁按用户全节点共享，
#      同节点多个训练 run 的 bench 也互斥（保护节点 CPU / 空闲卡）。
#   2. 不占训练卡：自动选卡时排除继承自训练进程的 CUDA_VISIBLE_DEVICES，
#      且只选 memory.used<1000MiB 并 util<10% 的卡；找不到 → 直接退出 0。
#   3. 任何失败只终止本脚本；训练侧 Popen 后不 wait，不受影响。
#
# 用法:
#   bash scripts/bench_subset.sh <checkpoint> <config> <exp_tag> \
#     [channels=lgbidex] [max_num=6000] [gpu=] [n_worker=48] [--dry-run]
#   gpu 为空 = 自动挑空闲卡；非空 = 直接用该物理卡号。
#   --dry-run（或 env BENCH_SUBSET_DRY_RUN=1）：只打印参数与选卡结果后退出；
#     配合 BENCH_SUBSET_HOLD=<sec> 持锁 sleep，用于 flock 重入测试。
#
# 评测走现成 scripts/run_full_eval.sh（推理→转换→eval→stat→全局 评测记录.md）。
# 注意 run_full_eval 内部 `CUDA_VISIBLE_DEVICES=$GPU` 直接用物理卡号，
# 所以这里把选中的物理卡号原样传给 --gpu，绝不做二次映射。
# 结束后把本次 exp_tag 的成功率行（来自全局 评测记录.md）追加到
# <训练output_dir>/bench_subset/history.md 汇总。
set -uo pipefail

TEST=/home/jiaxuan/exp-pami/test
RECORD_DOC=$TEST/评测记录.md

log() { echo "[bench_subset $(date '+%F %T')] $*"; }

# ---- 参数 ----
DRY_RUN=${BENCH_SUBSET_DRY_RUN:-0}
POS=()
for a in "$@"; do
  if [ "$a" = "--dry-run" ]; then DRY_RUN=1; else POS+=("$a"); fi
done
CKPT=${POS[0]:-}; CONFIG=${POS[1]:-}; EXP_TAG=${POS[2]:-}
CHANNELS=${POS[3]:-lgbidex}; MAXNUM=${POS[4]:-6000}; GPU_REQ=${POS[5]:-}; NWORKER=${POS[6]:-48}
if [ -z "$CKPT" ] || [ -z "$CONFIG" ] || [ -z "$EXP_TAG" ]; then
  log "usage: bench_subset.sh <checkpoint> <config> <exp_tag> [channels] [max_num] [gpu] [n_worker] [--dry-run]"
  exit 2
fi
log "start host=$(hostname) pid=$$ ckpt=$CKPT config=$CONFIG exp_tag=$EXP_TAG channels=$CHANNELS max_num=$MAXNUM gpu_req='${GPU_REQ}' n_worker=$NWORKER dry_run=$DRY_RUN"

# ---- 1. flock 重入锁（non-blocking：拿不到锁直接退出，不排队）----
LOCKFILE=/tmp/bench_subset_${USER:-$(id -un)}.lock
if ! exec 9>>"$LOCKFILE"; then
  log "skip: cannot open lock file $LOCKFILE"
  exit 0
fi
if ! flock -n 9; then
  log "skip: previous bench still running (lock: $LOCKFILE)"
  exit 0
fi

# ---- 2. GPU 选择（排除训练卡 = 继承的 CUDA_VISIBLE_DEVICES）----
TRAIN_GPUS=${CUDA_VISIBLE_DEVICES:-}
if [ -n "$GPU_REQ" ]; then
  GPU=$GPU_REQ
  log "using requested gpu=$GPU (physical index)"
else
  GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
    awk -F',' -v excl="$TRAIN_GPUS" '
      BEGIN {
        n = split(excl, e, ",")
        for (i = 1; i <= n; i++) { gsub(/[^0-9]/, "", e[i]); if (e[i] != "") bad[e[i]] = 1 }
      }
      {
        idx = $1; mem = $2; util = $3
        gsub(/[^0-9]/, "", idx); gsub(/[^0-9]/, "", mem); gsub(/[^0-9]/, "", util)
        if (idx != "" && !(idx in bad) && mem + 0 < 1000 && util + 0 < 10) { print idx; exit }
      }')
  if [ -z "${GPU:-}" ]; then
    log "skip: no idle gpu (need mem<1000MiB & util<10%, excluded train gpus='${TRAIN_GPUS}')"
    exit 0
  fi
  log "selected idle gpu=$GPU (excluded train gpus='${TRAIN_GPUS}')"
fi

RUN_CMD=(bash "$TEST/scripts/run_full_eval.sh"
  --checkpoint "$CKPT" --config "$CONFIG" --exp-name "$EXP_TAG"
  --channels "$CHANNELS" --max-num "$MAXNUM" --gpu "$GPU" --n-worker "$NWORKER"
  --notes "训练中bench钩子")

if [ "$DRY_RUN" = "1" ]; then
  log "DRY-RUN would exec: ${RUN_CMD[*]}"
  if [ -n "${BENCH_SUBSET_HOLD:-}" ]; then
    log "DRY-RUN holding lock for ${BENCH_SUBSET_HOLD}s (flock re-entry test)"
    sleep "$BENCH_SUBSET_HOLD"
  fi
  log "DRY-RUN done"
  exit 0
fi

# ---- 3. 评测（传物理卡号给 --gpu，见文件头注释）----
export CUDA_VISIBLE_DEVICES=$GPU
"${RUN_CMD[@]}"
rc=$?
log "run_full_eval exit=$rc"

# ---- 4. 成功率行写回训练 output_dir/bench_subset/history.md ----
# ckpt 位于 <output_dir>/checkpoints/epoch_XXXX.pt 或 <output_dir>/bench_snap_eN.pt
CKPT_PARENT=$(dirname "$CKPT")
if [ "$(basename "$CKPT_PARENT")" = "checkpoints" ]; then
  OUT_DIR=$(dirname "$CKPT_PARENT")
else
  OUT_DIR=$CKPT_PARENT
fi
HIST_DIR=$OUT_DIR/bench_subset
mkdir -p "$HIST_DIR" 2>/dev/null || true
HIST=$HIST_DIR/history.md
if [ ! -s "$HIST" ]; then
  {
    echo "# 训练期 bench 子集成功率（bench_subset.sh 自动追加；行取自全局 评测记录.md）"
    echo
    echo "| 时间 | exp_name | checkpoint | 通道 | eval数 | succ数 | 成功率 | 物体成功率 | coverage | 覆盖调整成功率 | 备注 |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|"
  } >> "$HIST"
fi
ROWS=$(grep -F "| $EXP_TAG |" "$RECORD_DOC" 2>/dev/null || true)
if [ -n "$ROWS" ]; then
  printf '%s\n' "$ROWS" >> "$HIST"
  log "history += $(printf '%s\n' "$ROWS" | wc -l) row(s) -> $HIST"
else
  printf '| %s | %s | %s | %s | - | - | 未获取(run_full_eval exit=%s) | - | - | - | 训练中bench钩子 |\n' \
    "$(date '+%F %H:%M')" "$EXP_TAG" "$(basename "$CKPT")" "$CHANNELS" "$rc" >> "$HIST"
  log "history += fallback row (评测记录.md 中未找到 $EXP_TAG 行) -> $HIST"
fi
log "DONE exp_tag=$EXP_TAG"
