#!/usr/bin/env bash
# 把服务器上的数据集/权重同步到本机磁盘，供 configs/local.yaml 使用。
#
# 背景：本机通过 rclone-SFTP 把远端 /mnt/afs/L202500241 挂载在
# /home/jiaxuan/wowowowo/slai，但该挂载吞吐很低（约 4-5MB/s），直接从挂载点
# 读 1.4GB 的训练集会在 DataLoader 初始化阶段卡住甚至超时。本脚本把训练用到
# 的文件一次性拷贝到本地 JuiceFS 磁盘，之后训练/多次调试都走本地读取。
#
# 用法：
#   bash scripts/sync_local_data.sh
#
# 前置条件：/home/jiaxuan/wowowowo/slai 已挂载，参见
# /home/jiaxuan/wowowowo/挂载命令.md。

set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/home/jiaxuan/wowowowo/slai}"
cd "$(dirname "$0")/.."

mkdir -p local_data/oakink_obj pretrained_models

echo "==> 同步 train/test json (v4new)"
rsync -ah --info=progress2 \
  "${REMOTE_ROOT}/asserts/lgbidex/pose_data/train_v4new.json" \
  "${REMOTE_ROOT}/asserts/lgbidex/pose_data/test_v4new.json" \
  local_data/

echo "==> 同步 mesh_root (oakink_obj processed_data)"
rsync -ah --info=progress2 \
  "${REMOTE_ROOT}/asserts/lgbidex/oakink_obj/processed_data/" \
  local_data/oakink_obj/processed_data/

echo "==> 同步 ModernBERT-base 权重"
rsync -ah --info=progress2 \
  "${REMOTE_ROOT}/test/pretrained_models/ModernBERT-base/" \
  pretrained_models/ModernBERT-base/

echo "==> 完成。用 configs/local.yaml 训练："
echo "    /mnt/conda/jiaxuan/miniconda3/envs/dexvlg/bin/python train.py --config configs/local.yaml"
