#!/usr/bin/env bash
# 一键四通道推理评测：inference → 四通道转换 → eval/eval3/eval4 → stat → 成功率记录
#
# 每个通道评测+统计完成后，自动从该通道的 stat summary JSON（stat_summary.json /
# stat3_summary.json / stat4_summary.json）读取评测数与成功数，计算成功率并追加
# 一行到 test 项目的《评测记录.md》（并发安全，flock）。
#
# 用法:
#   bash scripts/run_full_eval.sh \
#     --checkpoint outputs/.../best_eXXXX.pt --config configs/xxx.yaml --exp-name my_exp \
#     [--gpu 0] [--samples-per-combo 5] [--channels left,right,lgbidex,bidex] \
#     [--guidance-scale 2.0] [--max-num -1] [--n-worker 64] [--notes "备注"]
#
# 说明:
#   - 推理单卡（--gpu），转换/评测纯 CPU；评旧 ckpt 必须配旧行为 config（见 CLAUDE.md 警告）。
#   - --channels 可只跑子集（如 lgbidex 做 guidance 扫描）；推理会按通道过滤记录。
#   - 长任务请在 tmux 里跑。
set -uo pipefail

TEST=/home/jiaxuan/exp-pami/test
DG=/home/jiaxuan/slai/dgbench-pami
SAVE_ROOT=$DG/output
PY_DEXVLG=/mnt/conda/jiaxuan/miniconda3/envs/dexvlg/bin/python
PY_DG=/mnt/conda/jiaxuan/miniconda3/envs/DGBench/bin/python
RECORD_DOC=$TEST/评测记录.md

GPU=0; SPC=5; CHANNELS=left,right,lgbidex,bidex; MAXNUM=-1; NWORKER=64
GS=""; NOTES=""; CONFIG=""; CKPT=""; EXP=""; NUMSTEPS=10; BATCH=256
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CKPT=$2; shift 2;;
    --config) CONFIG=$2; shift 2;;
    --exp-name) EXP=$2; shift 2;;
    --gpu) GPU=$2; shift 2;;
    --samples-per-combo) SPC=$2; shift 2;;
    --channels) CHANNELS=$2; shift 2;;
    --guidance-scale) GS=$2; shift 2;;
    --max-num) MAXNUM=$2; shift 2;;
    --n-worker) NWORKER=$2; shift 2;;
    --num-steps) NUMSTEPS=$2; shift 2;;
    --batch-size) BATCH=$2; shift 2;;
    --notes) NOTES=$2; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
[ -z "$CKPT" ] || [ -z "$CONFIG" ] || [ -z "$EXP" ] && { echo "必填: --checkpoint --config --exp-name"; exit 2; }

PRED=$TEST/outputs_v3/predictions_${EXP}.json
LOG=$TEST/outputs_v3/run_${EXP}.log
[ -n "$GS" ] && NOTES="${NOTES:+$NOTES; }guidance_scale=$GS"
: > "$LOG"
echo "#### run_full_eval $EXP | node=$(hostname) gpu=$GPU spc=$SPC channels=$CHANNELS | $(date) ####" | tee -a "$LOG"

record_channel() {
  # $1=summary_json $2=channel
  "$PY_DEXVLG" - "$1" "$2" "$EXP" "$(basename "$CKPT")" "$NOTES" "$RECORD_DOC" <<'PY'
import fcntl, json, os, sys
from datetime import datetime

summary_path, channel, exp, ckpt, notes, doc = sys.argv[1:7]
try:
    d = json.load(open(summary_path))
except Exception as e:
    print(f"[record] READ FAIL {summary_path}: {e}")
    sys.exit(0)

num_eval = d.get("num_eval")
num_succ = d.get("num_succ", d.get("num_succ_by_eval_flag"))
rate = d.get("grasp_success_rate", d.get("grasp_success_rate_by_eval_flag"))
obj_rate = d.get("object_success_rate")
coverage = d.get("coverage")
adj = d.get("grasp_success_rate_coverage_adjusted")
fmt = lambda v, p=4: (f"{v:.{p}f}" if isinstance(v, (int, float)) else "-")
row = (f"| {datetime.now().strftime('%Y-%m-%d %H:%M')} | {exp} | {ckpt} | {channel} "
       f"| {num_eval if num_eval is not None else '-'} | {num_succ if num_succ is not None else '-'} "
       f"| {fmt(rate)} | {fmt(obj_rate)} | {fmt(coverage)} | {fmt(adj)} | {notes or '-'} |\n")

header = (
    "# Bench 评测记录（自动维护）\n\n"
    "由 `scripts/run_full_eval.sh` 每通道评测完成后自动追加；成功率取自各通道 stat summary JSON\n"
    "（left/right→stat_summary.json，lgbidex→stat3_summary.json，bidex→stat4_summary.json）。\n"
    "覆盖调整成功率 = succ/(eval+missing_hand)，缺手记录计为失败（见 convert_summary.json 边车）。\n\n"
    "| 时间 | exp_name | checkpoint | 通道 | eval数 | succ数 | 成功率 | 物体成功率 | coverage | 覆盖调整成功率 | 备注 |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|\n"
)
with open(doc, "a+", encoding="utf-8") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    if not f.read(1):
        f.write(header)
    f.seek(0, os.SEEK_END)
    f.write(row)
    fcntl.flock(f, fcntl.LOCK_UN)
print(f"[record] {channel}: eval={num_eval} succ={num_succ} rate={fmt(rate)} -> {doc}")
PY
}

# ===== [1/3] 推理（单卡）=====
echo "==== [1/3] inference $(date) ====" | tee -a "$LOG"
cd "$TEST"
GS_ARGS=(); [ -n "$GS" ] && GS_ARGS=(--guidance-scale "$GS")
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$TEST TOKENIZERS_PARALLELISM=false \
  "$PY_DEXVLG" infer_dataset.py \
    --config "$CONFIG" --checkpoint "$CKPT" \
    --split test --output "$PRED" \
    --task-type-filter "$CHANNELS" \
    --dedup-by "obj_id,scale_id,pose_id,guidance" --samples-per-combo "$SPC" \
    --num-steps "$NUMSTEPS" --batch-size "$BATCH" "${GS_ARGS[@]}" >>"$LOG" 2>&1
rc=$?; echo "infer exit=$rc $(date)" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "ABORT: inference failed（详见 $LOG）" | tee -a "$LOG"; exit 1; }

# ===== [2/3] 四通道转换 =====
echo "==== [2/3] convert $(date) ====" | tee -a "$LOG"
"$PY_DEXVLG" prepare_bench_data.py -i "$PRED" \
  --mesh-root /home/jiaxuan/data/oakink_obj/processed_data \
  --output-root "$SAVE_ROOT" --exp-name "$EXP" \
  --channels "$CHANNELS" --clean --num-workers 32 >>"$LOG" 2>&1
rc=$?; echo "convert exit=$rc $(date)" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "ABORT: convert failed（详见 $LOG）" | tee -a "$LOG"; exit 1; }

# ===== [3/3] 逐通道 eval + stat + 记录 =====
cd "$DG"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
IFS=',' read -ra CH_ARR <<< "$CHANNELS"
for ch in "${CH_ARR[@]}"; do
  case "$ch" in
    left)    T=eval;  H=shadow_left; ST=stat;  SUM=stat_summary.json;  SUF=shadow_left;;
    right)   T=eval;  H=shadow;      ST=stat;  SUM=stat_summary.json;  SUF=shadow;;
    lgbidex) T=eval3; H=shadow_left; ST=stat3; SUM=stat3_summary.json; SUF=shadow_left;;
    bidex)   T=eval4; H=shadow_left; ST=stat4; SUM=stat4_summary.json; SUF=shadow_left;;
    *) echo "unknown channel: $ch" | tee -a "$LOG"; continue;;
  esac
  echo "==== [$ch] $T $(date) ====" | tee -a "$LOG"
  "$PY_DG" src/main.py task=$T setting=tabletop hand=$H save_root="$SAVE_ROOT" \
    exp_name="${EXP}_${ch}" task.max_num="$MAXNUM" n_worker="$NWORKER" >>"$LOG" 2>&1
  echo "[$ch] eval exit=$? $(date)" | tee -a "$LOG"
  "$PY_DG" src/main.py task=$ST hand=$H save_root="$SAVE_ROOT" \
    exp_name="${EXP}_${ch}" >>"$LOG" 2>&1
  echo "[$ch] stat exit=$? $(date)" | tee -a "$LOG"
  record_channel "$SAVE_ROOT/${EXP}_${ch}_${SUF}/$SUM" "$ch" | tee -a "$LOG"
done

echo "#### DONE $EXP $(date)（记录见 $RECORD_DOC）####" | tee -a "$LOG"