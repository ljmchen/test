# DexVLG: Language-Aligned Bimanual Dexterous Grasp Generation

基于 [DexVLG (ICCV 2025)](https://arxiv.org/abs/2507.02747) 论文复现的双手灵巧抓取位姿生成模型。

## 架构概览

```
                    ┌────────────────┐
                    │  Language Inst. │
                    └───────┬────────┘
                            │
     ┌──────────┐    ┌──────▼──────┐
     │ Colored  │    │    BERT     │
     │ Point    │    │  Encoder   │
     │ Cloud    │    └──────┬──────┘
     └────┬─────┘           │
          │                 │
   ┌──────▼──────┐          │
   │  PointNet++ │          │
   │  Encoder    │          │
   └──────┬──────┘          │
          │                 │
   ┌──────▼──────┐          │
   │  Projector  │          │
   └──────┬──────┘          │
          │    ┌────────────┘
          │    │
   ┌──────▼────▼──────┐
   │     Fusion       │
   │   Transformer    │
   └───────┬──────────┘
           │
   ┌───────▼──────────┐
   │  Flow-Matching   │
   │  Transformer     │
   │  (AdaLN + xAttn) │
   └───────┬──────────┘
           │
    ┌──────▼──────┐
    │ Left  Hand  │  T(3) + R(6D) + θ(22)
    │ Right Hand  │  T(3) + R(6D) + θ(22)
    └─────────────┘
```

### 与原文的主要差异

| 组件 | 原文 (DexVLG) | 本实现 |
|------|-------------|--------|
| 点云编码器 | Uni3D | **PointNet++** (纯 PyTorch，无需 CUDA 编译) |
| 语言模型 | Florence-2 LLM | **BERT** (轻量高效，encoder-only) |
| 多模态融合 | Florence-2 内部融合 | **Fusion Transformer** (独立 4 层 Transformer) |
| 抓取目标 | 单手 | **双手** (左右手各 31 维：T3 + R6 + θ22) |

## 项目结构

```
.
├── configs/
│   └── default.yaml          # 模型、数据、训练的完整配置
├── data/
│   ├── __init__.py
│   ├── dataset.py            # 双手抓取数据集 (LGBiDex 格式)
│   └── lgbidex_io.py         # 网格/点云加载，旋转转换，坐标变换
├── models/
│   ├── __init__.py
│   ├── dexvlg.py             # DexVLG 主模型
│   ├── flow_matching.py      # Flow-Matching Transformer (DiT-style)
│   ├── pointnet2.py          # PointNet++ 编码器
│   └── pose_decoder.py       # 位姿解码器 (辅助工具)
├── utils/
│   ├── __init__.py
│   ├── misc.py               # 种子设置、参数统计、AverageMeter
│   ├── rotation.py           # 四元数 ↔ 旋转矩阵 ↔ 6D 旋转表示
│   └── visualization.py      # Open3D 点云 + 抓取可视化
├── train.py                  # 训练脚本 (DDP + 混合精度)
├── test.py                   # 评估脚本 (多指标)
├── inference.py              # 单样本推理 Demo
├── test_sanity.py            # 模型完整性检查
├── requirements.txt          # Python 依赖
└── README.md
```

## 数据格式

训练数据为 LGBiDex 格式 JSON 列表，每条记录包含：

```json
{
    "obj_id": "core-bottle-1a7ba1f4c82e2b4cec1c11c2c1a9a5d3",
    "obj_pose": [tx, ty, tz, qw, qx, qy, qz],
    "obj_scale": 0.1,
    "cate_id": "bottle",
    "guidance": "Grasp the bottle with both hands to pour",
    "dex_grasp_left":  [tx, ty, tz, ax, ay, az, j0, j1, ..., j21],
    "dex_grasp_right": [tx, ty, tz, ax, ay, az, j0, j1, ..., j21]
}
```

- `obj_pose`: 物体位姿 `[x, y, z, qw, qx, qy, qz]`（位置 + 四元数 wxyz）
- `dex_grasp_*`: 每只手 `[平移3D, 轴角3D, 关节角22D]` = 28 维
- 点云路径: `{mesh_root}/{obj_id}/pc_part/simplified_part_{N}.ply`
- 网格路径: `{mesh_root}/{obj_id}/mesh/simplified.obj`

## 模型细节

### 位姿表示 (每只手 31 维)

| 成分 | 维度 | 说明 |
|------|------|------|
| Translation T | 3 | 手腕平移 (物体中心坐标系) |
| Rotation R | 6 | 6D 连续旋转表示 (Gram-Schmidt 正交化) |
| Joint angles θ | 22 | Shadow Hand 22 自由度关节角 |

### Conditional Flow Matching (CFM)

- **训练**: 采样 t ~ U(0,1)，插值 x_t = (1-t)·ε + t·x_1，预测速度场 v = x_1 - ε
- **推理**: 从高斯噪声出发，Euler ODE 积分 dx/dt = v(x,t,c)，t: 0→1，50 步

## 评估指标

| 指标 | 说明 |
|------|------|
| `trans_err_cm` | 平移误差 (厘米) |
| `rot_err_deg` | 旋转测地距离误差 (度) |
| `joint_err_deg` | 关节角误差 (度) |
| `trans_success_2cm` | 平移误差 < 2cm 的成功率 |
| `rot_success_15deg` | 旋转误差 < 15° 的成功率 |

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- CUDA (推荐，CPU 可运行但较慢)

## 许可证

本项目仅用于学术研究。DexVLG 原文版权归原作者所有。
