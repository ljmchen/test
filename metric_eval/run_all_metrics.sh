#!/usr/bin/env bash
# run_all_metrics.sh — exp-pami/test 一键指标评测：转换 + 多样性 + Q1 + FID，汇总到一个累积记录文件。
#
# 用法:
#   bash run_all_metrics.sh [predictions.json] [tag] [gpu] [metrics]
#     predictions.json : 预测 JSON（默认 ../outputs/predictions_e0115_test.json）
#     tag              : 结果标签（默认从文件名去掉 predictions_ 前缀，如 e0115_test）
#     gpu              : Q1/FID 用的 GPU 编号；**需 sm<=89（L40/3090/A100 可，RTX 5090 不行）** 默认 2
#     metrics          : 逗号分隔要跑哪些项，默认 convert,div,q1,fid（可只跑部分，如 convert,div）
#
# 例:
#   bash run_all_metrics.sh ../outputs/predictions_e0148_test.json e0148 2
#   bash run_all_metrics.sh ../outputs/pred.json mytag 0 convert,div    # 只转换+多样性(纯CPU)
#   METRICS 已跑过想只汇总: bash run_all_metrics.sh <json> <tag> - collect  # 见 collect-only 说明
#
# 耗时(全量 8w 条/手): 转换~40s；多样性~1min(CPU)；Q1~5h；FID~2.5h。**长任务建议放 tmux**:
#   tmux new-session -d -s metrics 'bash run_all_metrics.sh ../outputs/predictions_e0148_test.json e0148 2'
#
# 各步输出: results/{convert,diversity,q1,fid}_<tag>.log、results/{diversity,q1,fid}_<tag>.json
# 汇总记录(累积，多次追加): results/metrics_record.md
#
# 指标覆盖范围（v3 四通道 task_type = left/right/lgbidex/bidex）:
#   diversity : 全部四类（convert 保留全部记录，bidex 组键用 task_type+scale_id 兜底）
#   Q1 / FID  : 仅 left/lgbidex —— mesh 经 OakInk 码 obj_id_old 解析，right(YCB)/
#               bidex(sem_*) 无 OakInk mesh，eval 脚本按 obj_id_old 过滤并显式打印跳过数
set -uo pipefail

DIR="/home/jiaxuan/exp-pami/test/metric_eval"
R="$DIR/results"
cd "$DIR"
mkdir -p "$R"

INPUT="${1:-../outputs/predictions_e0115_test.json}"
TAG="${2:-$(basename "$INPUT" .json | sed 's/^predictions_//')}"
GPU="${3:-2}"
METRICS="${4:-convert,div,q1,fid}"
INPUT_ABS="$(readlink -f "$INPUT")"
RECORD="$R/metrics_record.md"
# 中间产物带 tag，避免跨 tag 分步/并发时 div/q1/fid 读到别的 run 的转换结果
CONV_L="$R/results_left_${TAG}.json"
CONV_R="$R/results_right_${TAG}.json"

require_convert() {
  if [ ! -f "$CONV_L" ] || [ ! -f "$CONV_R" ]; then
    echo "错误: 缺少 $CONV_L 或 $CONV_R —— 请先对 tag=$TAG 跑 metrics=convert" >&2
    exit 1
  fi
}

source /mnt/conda/jiaxuan/miniconda3/etc/profile.d/conda.sh
conda activate dexgys

has() { [[ ",$METRICS," == *",$1,"* ]]; }
log() { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }

log "run_all_metrics: tag=$TAG gpu=$GPU metrics=$METRICS"
log "input=$INPUT_ABS"
[ -f "$INPUT_ABS" ] || { echo "输入文件不存在: $INPUT_ABS"; exit 1; }

if has convert; then
  log "1/4 转换 world->物体系 + 分组 (convert_to_gays.py)"
  python convert_to_gays.py -i "$INPUT_ABS" -o "$R" 2>&1 | tee "$R/convert_${TAG}.log"
  # 立即固化为带 tag 的副本，后续步骤只读带 tag 文件
  cp -f "$R/results_left.json" "$CONV_L"
  cp -f "$R/results_right.json" "$CONV_R"
  log "转换结果已固化: $CONV_L / $CONV_R"
fi

if has div; then
  log "2/4 多样性 DGTR+GAYS (CPU)"
  require_convert
  CUDA_VISIBLE_DEVICES="" python eval_dgtr_diversity.py \
    -l "$CONV_L" -r "$CONV_R" \
    -o "$R/diversity_${TAG}.json" 2>&1 | tee "$R/diversity_${TAG}.log"
fi

if has q1; then
  log "3/4 Q1/pen/valid_q1 GAYS (GPU $GPU, 约5h)"
  require_convert
  CUDA_VISIBLE_DEVICES="$GPU" python eval_gays_q1.py \
    -l "$CONV_L" -r "$CONV_R" \
    -o "$R/q1_${TAG}.json" 2>&1 | tee "$R/q1_${TAG}.log"
fi

if has fid; then
  log "4/4 深度 FID GAYS/PointNet++ (GPU $GPU, 约2.5h)"
  require_convert
  CUDA_VISIBLE_DEVICES="$GPU" python eval_gays_fid.py \
    -l "$CONV_L" -r "$CONV_R" \
    -o "$R/fid_${TAG}.json" --batch-size 64 2>&1 | tee "$R/fid_${TAG}.log"
fi

log "汇总三项结果 -> $RECORD"
python collect_summary.py --tag "$TAG" --input "$INPUT_ABS" \
  --div "$R/diversity_${TAG}.json" --q1 "$R/q1_${TAG}.json" --fid "$R/fid_${TAG}.json" \
  --out "$RECORD"

log "指标覆盖范围: diversity=全部四类 task_type(left/right/lgbidex/bidex)；Q1/FID=仅 left/lgbidex（right/bidex 无 OakInk mesh，已在各自日志中显式跳过，见 results/{q1,fid}_${TAG}.log 开头的覆盖统计）"
log "完成。累积记录: $RECORD"
