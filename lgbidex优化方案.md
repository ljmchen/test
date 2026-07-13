# lgbidex 通道优化方案：诊断与正规训练/架构改进

> 目标：把 lgbidex（双手小物体协同交接）的 bench 成功率从 ~23.6%（best_e0029，per-grasp）实质性提高。
> 本文 = 定量诊断（已完成，全部有数）→ 训练侧优化（T1-T6，config/小代码）→ 架构侧优化（A1-A3，中期）→ 实验协议与红线。
> 基调：**只做模型架构与训练侧的正规方法**；不采用推理端 best-of-N 选优类技巧（用户明确排除）。
> 数据源：predictions_v2e29_s5_test.json 全量 40.3 万条误差分解、dgbench e24/e59 仿真 npy 抽样、两次 run 的 train.log、v3_multitask 配置与模型代码逐行核对（2026-07-11）。

---

## 0. TL;DR

lgbidex 差的**不是**平移、关节、数据量、采样多样性或评测链（GT 同链 91-100%），而是：

1. **旋转预测质量是硬约束**：左手旋转测地误差均值 98.4°（中位 96.5°，52.8% 候选 >90°）；即使 5 采样取最好仍 44°，远超 eval3 的 15° 判据。几何合格率（<5cm ∧ <15°）仅 **3%**。
2. **训练已平台化**：val 自 e39 起在 8.13-8.24 振荡；lgbidex 分任务 rot 75 个 epoch 只从 1.28 降到 1.20。**继续训到 200 epoch 救不了。**
3. **双手无交互建模**：手间相对旋转 rel_r≈1.97，是 bidex（0.67）的 2.9 倍 → right_release_failure 占失败 25.8%。
4. **条件信号太弱**：CFG 实际关闭（drop_prob=0 + guidance_scale=1）；v2 数据把 lgbidex 唯一 guidance 从 11370 种塌缩到 1671 个模板 → 一条指令对应多个旋转模态，flow 把概率质量摊在多个模态间，单次采样命中率低（多样性健康 divTrans 0.056m，object-level 成功 0.81 ≫ per-grasp 0.226——这组差距在此仅作为"分布摊开"的证据）。

对应处置：**T1 CFG + T4 guidance 修复**（攻条件弱）、**T2/T3 旋转损失**（攻旋转精度）、**T5 曝光偏差 + A1 联合双手去噪**（攻 rel_r）、**A2 旋转多模态结构化**（若训练侧压不动 rot 的后手）。

---

## 1. 诊断（数据）

### 1.1 误差分解（world 系，40.3 万条预测 vs 各自 GT）

| 通道/手 | 平移L2均值(m) | 旋转均值(°) | 关节MAE(rad) | best-of-5 旋转(°) | 旋转>90° 占比 | 几何合格率 |
|---|---|---|---|---|---|---|
| left/left | 0.131 | 70.4 | 0.100 | 34.7 | 29% | — |
| right/right | 0.158 | 89.7 | 0.204 | 44.3 | 47% | — |
| **lgbidex/left** | **0.164** | **98.4** | 0.102 | **44.4** | **52.8%** | **2.97%** |
| lgbidex/right | 0.155 | 84.5 | 0.197 | 42.8 | 42.5% | 3.66% |
| bidex/left | 0.107 | **29.3** | 0.154 | 10.2 | 1.0% | ~12-14% |

- 旋转差 4 倍于 bidex；关节反而优于 bidex；平移 best-of-5 已接近判据（0.083 vs 0.05m）。
- **左手（交接第一棒、失败即终止）比右手更差**（98.4° vs 84.5°）。
- 均值/中位都在 90° 左右且非 180° 尖峰 → 是**弱旋转信号/模态摊开**，不是系统性左右翻转（GT 自检 91-100% 也排除了约定错误）。

### 1.2 仿真侧失败机制（e24/e59 各 ~1.5 万条）

| 指标（中位） | SUCCESS | left_lift_failure | right_release_failure |
|---|---|---|---|
| left_lift_height_gain (m) | 0.295 | **6e-6（没抬起来）** | 0.295 |
| left_lift_delta_pos (m) | 0.012 | **0.301（物体原地留下）** | 0.013 |
| left_object_pene (m) | 0 | 0.0024（远小于 0.015 判据） | 0.0125 |
| 指-物间隙 (m) | — | **0.004（没真正贴合）** | 0.000 |

失败=「抓到了但没夹紧」：手指闭合位姿与小物体几何没对准（与 98° 旋转误差因果一致），不是穿透、不是零接触。init_collision 7.4%（初始位姿扎进物体，init_pene 0.017>0.015）与旋转误差同源。right_release_failure 组左手抬升是好的——失败在右手接管，对应 rel_r。

### 1.3 平台化与排除项

- 第二次 run（修复版代码 + held-out val）：val 8.66(e4)→8.19(e39)→8.14(e69,best)→8.16(e79) 振荡；train loss 仍在降（0.60→0.48）= 收益递减/轻微过拟合。lgbidex 分任务 val：trans 0.141→0.134、rot 1.276→1.204、rel_r 2.033→1.972（75 epoch 几乎平的）。
- bench 侧 e24→e59（35 epoch）仅 22.6%→27.2%，主失败模式不变。
- **排除**：数据量（lgbidex 占训练 37% 且被上采样到 35%，占比最高却效果最差）；小物体归一化尺度（换算成米后 lgbidex 平移 val ≈0.025m 与 bidex 0.033m 同量级；且旋转 6D 直通与 L_obj 无关）；采样塌缩（divTrans 0.056m 与 bidex 相当，best-of-5 能腰斩误差）。

---

## 2. 训练侧优化（T1-T6，按预期 ROI 排序）

> 前置工程项（使所有实验降本）：`train.py` 加 `--init-weights PATH`——`load_state_dict(ckpt["model_state_dict"], strict=False)`（打印 missing/unexpected keys），**不**恢复 optimizer/scheduler/epoch；落点在 resume 块（train.py:1198-1216）旁另立分支。用途：从 `epoch_0079.pt` 微调 15-20 epoch（单卡 epoch≈20min，约 5-7h）快速验证，替代从零 40-60 epoch；**微调阳性可信，阴性需从零复验**（平台化模型可能微调撬不动，尤其 CFG 的 null 分支是新参数）。

### T1 启用 CFG（攻"条件弱"，ROI 最高）

- **机制**：诊断 4 的直接解——一指令对多旋转模态、flow 摊开；guidance_scale>1 把概率质量压向条件模态，提升单次采样命中率（有健康的多样性可供压缩）。
- **改动**：纯 config——训练 `model.cfg.drop_prob: 0.1`；推理扫描 `model.cfg.guidance_scale ∈ {1.5, 2, 3, 4}`（4 份 yaml 副本，或给 infer_dataset.py 加 3 行 `--guidance-scale` override）。
- **代码事实（已核）**：链路完整——`null_cond` 仅当 drop_prob>0 时创建（dexvlg.py:387-393）；`use_cfg = null_cond is not None and guidance_scale != 1.0`（:552/:777）；drop 分支把 memory+z+h+prev 整段换 null token（:639-677）；采样在 per-hand Euler 内环做两次 forward（:831-845）。**当前 ckpt 无 null_cond → 不能直接扫描，必须带 drop 重训或 --init-weights 微调**（strict=False 恰好覆盖新参数）。
- **回归控制**：CFG 只作用于 flow，不影响 presence/side；推理分通道跑，**只对 lgbidex 用 >1 的 scale**，其它三通道 1.0 → 推理侧零回归。监控大 scale 下归一化姿态是否被推出 [-1,1]（denormalize 外推）。
- **验证**：val/lgbidex/rot + 固定 bench 子集 per-grasp 随 scale 的倒 U 曲线取峰。

### T2 per-task 旋转损失缩放（攻旋转，精准不伤别人）

- **机制**：只把 lgbidex 的 rotation 损失项加倍（bidex rot 0.44 已好，全局上调会殃及）。
- **改动**（全在 models/dexvlg.py，train.py 不动——`task_type_ids` 已是 `compute_loss_latent_ar` 入参）：
  - `__init__`（flow_loss_weight buffer 旁）：解析新键 `model.flow_loss_task_scales: {lgbidex: {rotation: 2.0}}` → 构建 `(num_task_types, pose_dim)` 的 `task_scale_table` buffer（默认全 1；task 索引按 utils/metrics.py:28 TASK_TYPES 序）。
  - `compute_loss_latent_ar:646`：`weight = self.flow_loss_weight.view(1, pose_dim) * self.task_scale_table[task_type_ids]`（ids 为 None 回退全局）。per-hand 版本再乘 slot 掩码（循环内 `hand_side_ids[:, s]` 在作用域）。
- **验证**：val 四任务 rot/trans/joint 全表；红线见 §4。

### T3 旋转测地辅助损失（直接优化评测口径）

- **机制**：6D 上的 MSE 与测地角非单调对应，大角度误差的梯度信号弱；直接加测地角损失。
- **合法性（已核）**：relquantile11 下 **rot6d 归一化直通**（v3_multitask.yaml:117 + pose_normalizer.py）→ 归一化空间的 rot6d 就是真实旋转；由 flow 的 `x̂1 = x_t + (1-t)·pred_v` 恢复终点姿态即可监督。
- **改动**（compute_loss_latent_ar per-slot 循环内，~:686 后）：
  ```python
  x1_hat = x_t + (1.0 - t[:, None]) * pred_v
  R_pred = rotation_6d_to_matrix(x1_hat[:, 3:9].float())   # utils/rotation.py:47
  R_gt   = rotation_6d_to_matrix(gt_s[:, 3:9].float())
  cos = ((R_pred.transpose(-1,-2) @ R_gt).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
  geo = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
  ```
  掩码平均后乘 `model.rot_geodesic_weight`（建议 0.5）计入 total；可与 T2 的 task 掩码共用**只对 lgbidex 开**。bf16 下强制 float()；t→0 时 x̂1 噪声大，loss 不稳可加 `t > 0.3` 门控。
- **与 T2 的关系**：同属"旋转损失包"，可合并为一个 run 先验证组合收益，消融后置。

### T4 v1 丰富 guidance 混回（根因修复，中期最重要）

- **机制**：1671 个模板无法区分 11370 种抓法的旋转模态；v1 文本含接触/指位线索（如 *"Grasp the cylinder bottle with all five fingers making contact with its body."*），恢复"一指令 ↔ 少数模态"的对应，让 flow 有条件可依。**与 T1 相乘生效：先有可分条件，再放大条件。**
- **改动**：新工具 `tools/mix_guidance_v1.py`——流式对齐 train_fuse_v1.json 与 train_fuse_v2_split.json（v2 由 v1 仅改 guidance 生成、记录一一对应，按 obj_id+grasp_id+cand_idx 对齐校验），对 task_type==lgbidex 的记录以 **p=0.5 按 grasp_id 哈希确定性替换**回 v1 guidance，输出 `train_fuse_v2v1mix_split.json`；config 只改 `data.train_data`。
- **两个陷阱（已核）**：① `epoch_sampling.group_keys` 含 guidance（yaml:180）→ 必须**逐条替换**而非双份并存，否则 lgbidex 组数翻倍、打破 35% 配比；② **val/test 保持 v2 不动**（测试集是 v2 模板，训练混入而非替换保证测试文本仍在分布内）。
- **健康信号判读**：若 val（v2 文本）不动而 bench 上升 → 收益来自表征而非文本记忆。

### T5 手2条件加噪对齐曝光偏差（攻 rel_r）

- **机制**：训练时手2条件在"GT手1 + 0.1 噪声"，推理时条件在"采样手1"（偏差远大于 0.1）——曝光偏差直接推高 rel_r 与 right_release_failure。
- **改动**：现成 config 键 `reasoner.hand_cond_noise_std: 0.1 → 0.3`（v3_multitask.yaml:63）。
- **验证**：val lgbidex 行的 rel_r + bench right_release_failure 占比；观察手2 绝对 rot 是否被噪声损伤。

### T6 辅助探针（低成本搭车项）

- `reasoner.aux_probe.enabled: true`（零代码；loss_weights.probe=0.1 已配好，train.py:232-235 已传 task_type_ids/log_l_obj）。强制 thinking tokens 编码任务/尺度，增强条件表征区分度。预期小，**与 T5 合并为一个 run**（同作用于 reasoner、机制正交）。

---

## 3. 架构侧优化（A1-A3，中期正规改造）

### A1 联合双手 flow 去噪（核心架构建议，对症 rel_r）

**现状与死键警告（已逐行核实）**：
- latent_ar 分支构造 flow transformer 时**硬编码** `hand_interaction={"enabled": True, "cross_hand_attn": False}`（dexvlg.py:366-374），config 里的 `model.hand_interaction.cross_hand_attn` 被忽略——**它是死键，翻了也没用**。
- 且 `cross_hand_attn` 的实现是 query 维度自注意力（flow_matching.py:192-193）；latent_ar 按手采样 num_queries=1，即使打开也是单 token 自注意 = no-op。
- 现有手间耦合 = 手1 的 (side+pose) prev-token 进手2 flow 的 cross-attn memory（dexvlg.py:656-659/:813-819）。按手自回归下这是**因果正确**的设计（手1 采样时手2 不存在，双向 attention 无对象），但表达能力有限——rel_r 2.9× 的差距说明单向 token 耦合不足以建模小物体交接的紧密手间约束。

**正解：reasoner 决策 + 联合去噪**。保留 reasoner 的手集合决策（presence/side，变手数能力不丢），flow 阶段改为**对"在场手"做集合式联合去噪**：两只手的 x_t 同时作为 2 个 query token 进同一次 flow forward，恢复启用 query 间 cross-hand attention，每个 ODE 步两手的含噪状态互相 attend——联合分布建模、无因果问题（这是双手扩散/流生成的标准做法）。

**改动面**：`sample_latent_ar` 的按手循环 → 集合式一次去噪；`flow_matching.py` 恢复多 query 路径（代码骨架现成）；`compute_loss_latent_ar` 对应改为联合监督（两手同一 t、同一 forward）；presence/side/thinking 链路不动。训练需从零重训。

**预期**：直接压 rel_r（1.97→bidex 量级 0.7 为目标），连带 right_release_failure（25.8%）与交接稳定性。

### A2 旋转多模态的结构化处理（若 §2 压不动 rot 的后手）

把"一对多回归"改为"分类 + 单模态回归"：
- **SO(3) anchor 分类 + 锚系残差 flow**：对 GT 旋转聚类 24-60 个锚（可按 cate/action 分层），reasoner 增加 anchor 分类头（CE 监督），flow 在选中锚的坐标系里出残差旋转。机制上最对症"98° 模态错误"——分类选模态、回归管精度。改动面：数据管线（锚标签离线预计算）、reasoner 头、flow 条件与采样四处。
- 或 **SO(3) 流形上的 flow matching**（旋转的测地插值路径替代 6D 欧氏路径），几何上更正确，实现参考 FoldFlow/SO(3)-diffusion 一族。
- **启动条件**：T1-T4 组合后 val/lgbidex/rot 仍压不进 ~0.8 rad（≈46°）。

### A3 接触感知条件（与 A2 同级备选）

- `contact_area` 字段**全为 null**（已核 val/train 抽样），不可直接用。
- 正规路线：离线用 GT 手 FK + mesh 重算"接触部件标签"（哪些手指链接触物体哪个部件），作为 reasoner 的辅助监督 token（DextER 式具身 CoT 的落地）。语义→物理约束的桥，对旋转模态选择有直接信息量；工程量大（离线标注 + 数据管线 + 头），列为中期。

---

## 4. 实验协议与红线

- **双基线**：epoch 匹配对比用 `epoch_0039.pt / epoch_0059.pt`（train.log 有同 epoch per-task 表，e84 参照：lgbidex rot 1.1986 / rel_r 1.9983）；bench 基线用 `best_e0069_s8.1382.pt`（val 已平台，当前 run 跑完后用最终 best 复核一次）。
- **早停 kill 规则**（利用"lgbidex val rot 前 40 epoch 定型"）：短训 60 epoch、val_interval=5；**e40 时 val/lgbidex/rot 未比基线同 epoch 低 ≥5%（1.20→≤1.14）且 rel_r 无改善 → 杀 run 释放卡**。T1（CFG）例外：收益在推理 guidance>1 才显现，e40 先做子集推理扫描再判。
- **bench 子集协议**：`infer_dataset.py --task-type-filter lgbidex --dedup-by obj_id,scale_id,pose_id,guidance --samples-per-combo 5` 固定前 1200 组（≈6000 grasps）→ 现成转换/eval3/stat3 链（CPU n_worker 高并行，1-2h）。报告：per-grasp 成功率、fail-reason 分布、离线几何合格率（<5cm∧<15°，从 predictions 直接算不进仿真）。胜者才跑全量 + 四通道回归。
- **四通道回归红线**：任何训练侧改动使 left/right/bidex 的 val rot/trans 劣化 >5% → 该改动降级为仅 lgbidex 通道推理侧启用（CFG 天然支持）或调低强度（T2 缩放系数）。
- **资源纪律**：基线 run 在 GPU0 继续不动；实验用空闲卡（1/5/7），每 run 独立 output_dir + config 副本 + tmux 启动；单卡 epoch ≈20min。

### 建议执行序（获批后）

| 阶段 | 内容 | 卡 | 时长 | 验收 |
|---|---|---|---|---|
| P1 | `--init-weights` 工程项 + T1（CFG）/ T5+T6 两个并行短训 | 2 | 微调 ~7h 或从零 ~20h | e40 kill 规则；CFG 扫描峰值 bench 子集 ≥0.30 |
| P2 | T2+T3（旋转损失包）短训；T4（mix 工具+数据+短训） | 2 | 同上 | val/lgbidex/rot ≤1.0 @e40-60 |
| P3 | 胜者组合从零 200 epoch 正式 run + 全量 bench + 四通道回归 | 1-2 | ~3 天 | lgbidex per-grasp ≥0.40、rel_r ≤1.2、init_collision <4%、right_release <15%、三通道不回归 |
| P4 | A1 联合去噪（若 rel_r 仍高）/ A2（若 rot 仍压不动） | — | 设计先行 | 文档评审后实施 |

---

## 5. 明确不做（诊断已排除 / 用户已排除）

- ✗ 加数据 / 再上采样 lgbidex（占比 37% 已最高、上采样已 35%）。
- ✗ 调平移归一化 / 小物体尺度补偿（平移非瓶颈；rot6d 与 L_obj 无关）。
- ✗ 拉长训练到 200+（e39 起平台，只会过拟合）。
- ✗ contact_area 直接作条件（字段全 null）。
- ✗ 翻 `hand_interaction.cross_hand_attn` config（死键 + 单 query 下 no-op；正确路线是 A1）。
- ✗ **推理端 best-of-N / top-k 选优**（用户明确排除；object-succ 0.81 vs per-grasp 0.226 的差距只作为诊断证据，不作为方案）。
- ✗ 动关节损失/归一化（lgbidex 关节 0.102 rad 优于 bidex）。

---

## 附录：关键代码事实索引（2026-07-11 逐行核对）

| 事实 | 位置 |
|---|---|
| latent_ar 硬编码 cross_hand_attn=False（死键） | models/dexvlg.py:366-374；flow_matching.py:192-193 |
| 手间耦合现状（prev-hand token） | dexvlg.py:656-659、:813-819；latent_reasoner.py:226-237 |
| CFG 链路（null_cond/use_cfg/drop/双 forward） | dexvlg.py:387-393、:552、:777、:639-677、:831-845 |
| flow 损失权重落点（task_type_ids 已入参） | dexvlg.py:646、:686；train.py:232 |
| rot6d 归一化直通（测地损失合法） | v3_multitask.yaml:117；data/pose_normalizer.py；utils/rotation.py:47 |
| 手2条件噪声 / aux_probe 现成键 | v3_multitask.yaml:63（hand_cond_noise_std）、aux_probe 块 |
| group_keys 含 guidance（T4 陷阱） | v3_multitask.yaml:180 |
| v1/v2 记录一一对应、仅 guidance 不同 | /home/jiaxuan/data/pose_data/fuse_v2_guidance_summary.json |
| contact_area 全 null | train/val split 抽样核对 |
| e84 val 参照数 | outputs/test_v2data_scratch/logs/train.log |
