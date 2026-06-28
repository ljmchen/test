#!/usr/bin/env bash
# 在 H100 远端主机上创建 dexvlg conda 环境（与本机 5090 上已验证环境一致）。
#
# 背景：本机看到的 /home/jiaxuan/wowowowo/slai 是远端 H100 主机经 rclone-SFTP 挂载，
# 该挂载不能执行二进制、chmod 也不生效，所以环境无法从挂载端创建——必须在 H100
# 主机本地执行本脚本。H100(sm_90) 与 RTX 5090(sm_120) 共用同一套 cu128 wheel。
#
# 用法（在 H100 主机上）：
#   bash setup_env_h100.sh
#
# 如 H100 集群与 5090 机共享 /mnt/conda(JuiceFS)，则无需本脚本，直接：
#   source /mnt/conda/jiaxuan/miniconda3/bin/activate dexvlg

set -euo pipefail

# H100 远端主机上的 conda 根目录（= 本机挂载点 slai/miniconda 对应的真实本地盘）。
CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/L202500241/miniconda}"
ENV_NAME="${ENV_NAME:-dexvlg}"
PY_VER="${PY_VER:-3.13}"

CONDA_BIN="${CONDA_ROOT}/bin/conda"
if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "ERROR: 找不到可执行的 conda: ${CONDA_BIN}" >&2
  echo "       请在 H100 主机本地运行本脚本，并按需用 CONDA_ROOT=... 指定其 miniconda 根目录。" >&2
  exit 1
fi

echo "==> 创建环境 ${ENV_NAME} (python=${PY_VER}) at ${CONDA_ROOT}/envs/${ENV_NAME}"
"${CONDA_BIN}" create -n "${ENV_NAME}" "python=${PY_VER}" -y

# 用该环境的 pip 安装，避免 shell hook 依赖；代理 403 时用 NO_PROXY="*" 绕过。
PIP=("${CONDA_ROOT}/envs/${ENV_NAME}/bin/python" -m pip)

echo "==> 安装 PyTorch (cu128，支持 H100 sm_90 / 5090 sm_120)"
NO_PROXY="*" "${PIP[@]}" install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0" "torchvision==0.26.0"

echo "==> 安装项目其余依赖（版本对齐已验证环境）"
NO_PROXY="*" "${PIP[@]}" install \
  "transformers==5.12.1" \
  "numpy==2.4.4" \
  "scipy==1.18.0" \
  "trimesh==4.12.2" \
  "pyyaml==6.0.3" \
  "tqdm==4.68.3" \
  "tensorboard==2.20.0"
# 可选：inference 可视化用，训练不需要；在 numpy 2.x 下可能需较新 open3d。
# NO_PROXY="*" "${PIP[@]}" install open3d || echo "open3d 安装失败(可忽略，仅可视化用)"

echo "==> 验证"
"${CONDA_ROOT}/envs/${ENV_NAME}/bin/python" - <<'PY'
import torch, transformers, numpy, yaml, tqdm, tensorboard, scipy, trimesh
print("python deps ok | transformers", transformers.__version__)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| archs", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "==> 完成。训练命令："
echo "    source ${CONDA_ROOT}/bin/activate ${ENV_NAME}"
echo "    python train.py --config configs/default.yaml"
