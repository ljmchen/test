# GRACE：把「affordance / thinking」做成可信、可视化、可消融的抓取思维链

> **GRACE = Grounded Reasoning over Approach & Contact for grasp gEneration.**
> 一句话：让 reasoner 在生成位姿之前，先吐出三个**有监督、可画进论文、可消融**的「想法」——
> **WHERE**（物体上的接触热力图）、**HOW**（S² 接近方向的离散分布）、**RELATE**（双手相对旋转），
> 用它们去**条件化** flow；而 flow 的**输出空间与整条 `infer→prepare_bench_data→dgbench` 转换链原封不动**。
>
> 目标读者：写论文的你 + 复现实验的你。落地代码已提交，`configs/v3_reason.yaml` 一键从零训练。
> 设计经 4 方案对抗评审 + 综合 + 一次性正确性/约定安全审计（4 处必修项已全部并入代码）。

---

## 0. TL;DR

**病根（不是玄学，是数据）**：`lgbidex优化方案.md` 已经证明，主瓶颈是**旋转多模态**——lgbidex 左手旋转
测地误差 98.4°（中位 96.5°、52.8% 样本 >90°），几何合格率（<5cm ∧ <15°）仅 **2.97%**；仿真失败就是
"抓到了但没夹紧、手指闭合与几何没对齐"，与那 98° 因果一致。次瓶颈是**双手交互**（rel_r 1.97 ≈ bidex 的
2.9×，right_release 占失败 25.8%）。

**为什么现在的 affordance/thinking「不让人信服」**：
- `affordance`（`dexvlg.py:encode_condition`）= 对 64 个 token 做**无监督 softmax 池化**、产 1 个 summary
  token。无监督、不接触地、不可视化、不可消融——审稿人问"它 afford 了什么"你无图可给，删了指标大概率不动。
- `thinking`（`LatentReasoner` 的 6 个 Coconut 潜 token）= 只有一个可选的弱 aux probe。从没被证明编码了
  任务需要的东西，K→0 消融大概率也不掉分。是"为思考而思考"。

**GRACE 的解法**：把这两个模块**换成一条有监督的抓取思维链**，且每一步都精准打瓶颈——
它正是诊断文档自己开的两味药 **A2（结构化旋转多模态）+ A3（接触感知条件）**，但改造成**一次性安全**的形态：
A3 的"离线 FK 接触标签"→ **自监督**的腕部几何高斯（不需 mesh/FK，也绕开全为 null 的 `contact_area`）；
A2 的"SO(3) k-means + 锚系残差 flow（4 个约定敏感点）"→ **无需数据、无需聚类、无需改输出空间**的
S² 接近方向码本，仅**条件化** flow。**每一条箭头都是一行消融，每一个模块都是一张图。**

---

## 1. 病理：为什么旧 affordance / thinking 无法写进论文

| 维度 | 旧 affordance | 旧 thinking(K=6) | 一个可信的组件应该满足 |
|---|---|---|---|
| 监督 | ✗ 无 | ✗ 仅弱 aux probe | 有明确的自监督/GT 目标 |
| 可视化 | ✗（1 个抽象 token） | ✗（潜向量） | 能画成图（热力图 / 箭头 / 相对坐标系）|
| 可消融 | ✗（删了不掉分） | ✗（K→0 不掉分） | 消融能移动一个**具名**指标 |
| 对症 | ✗ 与旋转/交互无关 | ✗ 与旋转/交互无关 | 机制上直击已诊断的瓶颈 |

一个组件如果不满足右列，它在论文里就是**装饰**——reviewer 的第一刀就砍它。GRACE 的设计原则就是让
affordance 与 thinking 同时满足右列四条。

---

## 2. 故事（论文叙事）

**方法名**：GRACE。reasoner 的每只手在去噪任何位姿**之前**，必须先产出两个接地的"想法"——
**WHERE**（点云上的接触场）与 **HOW**（接近方向码本上的类别分布）——以及一个手间的想法 **RELATE**
（两手腕的相对旋转）。三者都从 GT 几何**自监督**、都能直接画成图、都反过来**条件化** flow。

**一段话 pitch（为什么可信）**：诊断已证明主瓶颈是**旋转多模态**而非平移/关节/数据/约定
（GT 同链 91–100%）。审稿人最会质询的两个模块——无监督 affordance 池 与自由潜 thinking——恰好
**没有任何有监督、可视化、可消融的输出**，删掉不动分。GRACE 用诊断自己推荐的 A2+A3 取而代之，但工程上
按"一次性安全"改造：接触目标是自监督高斯（无 mesh/FK/标签），接近方向是无数据的 S² 码本、仅做条件而
**不重参数化输出**（零约定风险）。于是——**故事里每一步都对应一行会动某个具名指标的消融，每个模块都对应
一张图。** 这正是把"reasoning"从不可证伪的装饰，变成**被测量、被消融、与具体失败绑定**的量。

**因果弧（WHERE → HOW → REFINE，双手关系并行）**：
语言 + 点云 → reasoner 潜思维 → 每只手产出
（WHERE）64 个 PointNet++ token 中心上的接触热力图 和
（HOW）S² 上接近方向的类别后验。**接近方向的 argmax 钉死 3 个旋转自由度中的 2 个**
（S² 上一点 = R_wrist·ê），把只剩 roll 留给 flow——于是把多模态的 `p(rot|text)` 变成近单模态的
`p(rot|text, approach-mode)`。已提交的接触质心与接近方向作为 flow 的条件 token 反馈回去，所以增益是
**因果的、不是化妆的**。双手之间，一个有监督的相对旋转头预测 R0ᵀR1（**正是 rel_r 指标**）并条件化第二只手，
取代原来那个单薄的 prev-hand token。

**新颖性声明（可在 CoRL/RSS/CVPR 立住）**：*我们把语言条件灵巧抓取生成器里两个最常见却最不可问责的模块——
无监督 affordance 池 与自由潜"thinking"——重铸为一条**有监督、可视化的抓取思维链**，其中间量是接地的接触场
与离散的接近方向后验。我们证明：把**接近方向**（旋转多模态经验上的主轴）离散成一个无数据的球面码本、并让
flow-matching 生成器条件于已提交的模式，能把一对多的旋转回归变成**先分类再精修**。据我们所知，这是第一个把
抓取生成器里的"推理"做成**被测量、可消融、且与具体失败（旋转多模态、手间 rel_r）绑定**的量的方法，并自带
oracle 上界与单模态坍缩的证伪实验。*

**每个机制对哪个瓶颈**：
- **瓶颈 1（旋转多模态，98°/2.97%）**：HOW 的接近-锚 CE 是**直接主攻**——在多模态真正存在的 S² 轴上做离散
  模式选择、反馈为 flow 条件 → 概率质量向单模态集中；WHERE 向共享编码器注入**旋转判别**梯度（接触质心随
  R_wrist·offset 移动）。二者与既有 `rot_geodesic` 辅助损失叠加以磨细残余 roll。**M=1 坍缩**消融把旋转打回
  ~98°（证因果）；**oracle**（推理时喂 GT 接近方向）给出可达上界、把剩余 roll 量化成一张 headroom 图。
- **瓶颈 2（rel_r 1.97 vs 0.67，right_release 25.8%）**：RELATE 监督 R0ᵀR1（因 rot6d 归一化直通而合法），
  并把它条件化到第二只手 flow——比 prev-hand token 富得多的耦合通道。

---

## 3. 架构（组件 · 监督输出 · 张量流）

全部由 `model.grace.enabled` 门控，**默认 false → 零新增参数/token/损失/必需字段，每一次默认前向逐位一致**。
下用真实 config 尺寸：`reasoner.dim=512`(R)、`bert_dim=768`(Dc)、`num_pc_tokens=64`(P)、`M`(接近锚，默认 64)、
`pose_dim=31`。

### 3.1 暴露 token 中心（`models/pointnet2.py`）
`PointNet2Encoder.forward(..., return_centers=False)`：把被丢弃的 `xyz3` 一并返回（`(feat3, xyz3)`）。
`xyz3 (B,P,3)` 是 sa3 的 FPS 中心，处于**与输入 xyz 同一坐标系（物体中心化、米）**，且与 `feat3` 位置对齐——
这就是接触热力图的可视化底座。默认 False → 单张量返回逐位一致（`test_pointnet2` 不变）。

### 3.2 GRACE 头（`models/dexvlg.py`，仅 latent_ar + use_grace 构造）
共享 helper `_grace_heads` / `_grace_rel`，**训练与两条采样路径复用同一份**（这是训练/采样一致性的根本保证）：
- `grace_q: Linear(R,Dc)` — 接触 query。`contact_logits = einsum('bhd,bnd->bhn', q, pc_tokens)/√Dc → (B,H,P)`；
  **有监督**（自监督高斯）；**可视化**为 xyz3 上的热力图。`contact_tok = softmax·pc_tokens → (B,H,Dc)`
  （有监督地取代旧 affordance 池）。
- `grace_approach: MLP(R→hidden→M)` — `approach_logits (B,H,M)`；**有监督**（锚 CE）；**可视化**为 S² 上的后验/箭头。
  `grace_anchors (M,3)` = 确定性 **Fibonacci 球**单位向量（无数据）。`a_hat = normalize(softmax·anchors) (B,H,3)`；
  `approach_tok = grace_approach_embed(a_hat) (B,H,Dc)`。
- `grace_rel_head: Linear(2R,6)` + `grace_rel_embed: Linear(6,Dc)` — `rel_rot6d (B,6)`；**有监督**（对 R0ᵀR1 测地）；
  **可视化**为相对腕坐标系。

### 3.3 训练数据流（`compute_loss_latent_ar`）
`encode_condition(...,return_centers=True) → memory (B,P+L,Dc), mem_mask, xyz3`。`pc_tokens = memory[:,:num_pc]`
（fusion 不重排，前 P 个恒为 PC token，与 xyz3 逐位对齐；`affordance` 关闭时更纯净）。
`ro = rollout_teacher_forced(...) → hand_hidden (B,2,R)` → `g = _grace_heads(hand_hidden, pc_tokens)`、
`rel6, rel_tok = _grace_rel(hand_hidden)` → 计算 4 个损失（§4）。条件束
`grace_cond = {contact_tok, approach_tok, rel_tok}`（`condition:false` 则为 None；`detach_condition:true` 则 detach）
传入 `_sequential_flow_terms`，**固定顺序**逐手追加：`[…, contact_s, approach_s]`，slot 1 再加 `[prev, rel]`。
现有的 `cond_mask` 扩展与 CFG-drop 都按 `cond_s.shape[1]` 取尺寸，**自动覆盖新 token，无需改 mask/CFG**。

### 3.4 采样数据流（自洽，从**预测**头而非 GT）
`sample_latent_ar` 逐步用**预测**的 `h_s` 重建同一批 token（顺序与训练完全一致，rel 用 `stack([h_0,h_1])`）。
采样端**不需要 xyz3**（推理不算目标），故只有 `compute_loss_latent_ar` 一个调用点改了返回元数。
`flow` 输出、Euler 积分、`denormalize_pose`、`infer_dataset.py`、`prepare_bench_data.py`、eval/eval3/eval4
**全部未动**。

### 3.5 新增 batch 字段（ê 单一来源）
`grasp_center (B,2,3)`、`approach_dir (B,2,3)`，均在**物体原始米坐标系**，均在 dataset 由**同一个 ê**导出——
`grasp_center` 必须来自 dataset（归一化后的 `gt_poses[:,:,:3]` 无法还原原始米），`approach_dir` 同理。

---

## 4. 损失（自监督 · 掩码 · 合法性）

目标全部由 GT 几何自监督，无外部标签、无 mesh、无 FK。每个逐手项都乘 `mask_f=hand_mask.float()`（可选再乘
`grace_task_mask[task_type_ids]`）并除以在场手数；相对项按双手都在场计数。**pad slot 贡献恒为 0**（满足
`test_latent_ar_mixed_batch` 的 pad-garbage 不变性）。因 rot6d 归一化直通（`pose_normalizer.py`）而合法，
且凡是接地在 xyz3 上的量都在原始米坐标系、帧一致。

**目标导出（dataset，原始物体系）**：对每只在场手，从 `centered = _center_hand_pose(raw, record)` 取
`t=centered[:3]`、`R=axis_angle_to_matrix(canonicalize_axis_angle(centered[3:6]))`、`ê=normalize(palm_forward_axis)`：
`approach_dir = R@ê`（单位）；`grasp_center = t + palm_offset·(R@ê)`。模型内由 `gt_poses[:,s,3:9]` 经
`rotation_6d_to_matrix` 还原的 R 与此一致（归一化只是重正交），帧保证一致。

| 损失 | 想法 | 目标 | 计算 | 权重 |
|---|---|---|---|---|
| `loss_contact` | WHERE | 尺度感知高斯 `q=softmax(−‖xyz3−gc‖²/2σ²)`，`σ=clamp(scale·L,min,max)` | 逐手 soft-CE(`contact_logits`, `q.detach()`) | 0.5 |
| `loss_anchor` | HOW(主) | `argmax_k(approach_dir·anchor_k)`（fp32） | 逐手 CE | 0.5 |
| `loss_approach` | HOW(精+图+oracle) | `approach_dir`（单位） | 逐手 `1−(a_hat·approach_dir).sum()`（**不除范数**）| 0.25 |
| `loss_rel_rot` | RELATE(瓶颈2) | `R0ᵀR1` | 双手都在场时测地角，acos 前 `float()`+clamp | 0.5 |

> **审计必修项（已并入代码）**：① `loss_approach` **绝不除以 `‖approach_dir‖`**——pad slot 的
> `approach_dir=[0,0,0]` 会让 `0/0=NaN`、再 `NaN×mask(0)=NaN` 静默毒化总损失、每步被跳过、悄悄学不到东西；
> 改用点积形式 `1−(a_hat·approach_dir)`（二者对在场手都是单位向量，pad 处点积=0→损失=1→有限→被掩码归零）。
> sanity 里专门把 pad slot 置 `[0,0,0]` 触发该路径。② `grace_task_mask[task_type_ids]` 仅在 `task_type_ids
> is not None` 时应用；仅当 `task_types` 非空才**要求** `task_type_ids`（镜像 `rot_geo_needs_task_ids`）——
> 默认 `task_types:[]` 必须能在 `task_type_ids=None` 下运行。③ 接触目标 `q` 必须 `.detach()`（xyz3 带编码器
> 梯度，否则"目标"会随编码器滑动而退化）；`forward()` 必须新增 `grasp_center/approach_dir` 形参并透传
> （否则 batch 1 就 TypeError）。④ `self.use_grace` 对**所有**架构都定义（legacy 置 False 并对误配 raise）。

---

## 5. 落地实现（file-by-file，已提交）

| 文件 | 改动 |
|---|---|
| `models/pointnet2.py` | `forward(return_centers=False)`：暴露 `xyz3`；默认路径逐位不变。 |
| `models/dexvlg.py` | 模块级 `_fibonacci_sphere`；`use_grace` 对所有架构定义（误配 raise）；latent_ar 内构造 GRACE 头 + `grace_anchors` buffer + 权重/σ/task_mask（`use_grace+joint_hand_denoise` 明确 `NotImplementedError`）；`_grace_heads/_grace_rel/_grace_losses`；`encode_condition(return_centers)`；`compute_loss_latent_ar` 计 4 损失并入 total、透传 `grace_cond`；`_sequential_flow_terms` 追加 GRACE token（固定顺序）；`sample_latent_ar` 从预测头重建 token；`forward` 新增两形参并透传。 |
| `data/dataset.py` | `__init__(grace_targets=…)` → `_grace_on/_palm_axis/_palm_offset`；`_grace_targets(record,sides)` 算 `grasp_center/approach_dir`（原始系）；`_getitem_multi_task` 追加；`_collate_multi_task` 零填充 `(B,2,3)`（`grasp_center in batch[0]` 守卫，legacy/collate 测试不变）。 |
| `train.py` | `_TRAIN_LOSS_COMPONENT_KEYS` += 4 键（日志/TB 自动带上）；`extra_inputs` 在 batch 含字段时带上 `grasp_center/approach_dir`；`dataset_kwargs` += `grace_targets`。 |
| `configs/v3_reason.yaml` | 新配置：`grace.enabled:true`、退休 `affordance`、`joint_hand_denoise` 缺省(false)、走 sequential、从零训练。 |
| `test_sanity.py` | 新增 `test_grace_latent_ar`：4 损失有限 + backward 梯度有限 + **pad zero-norm approach NaN** + 采样契约；旧 10 测试保持全绿。 |

**默认关闭 = 逐位一致**：无 `grace` 块 → `use_grace=False`、无参无损、`grace_cond=None`、dataset `_grace_on=False`；
`v3_multitask.yaml` / `v3_smoke.yaml` / legacy / 10 个旧 sanity 测试不受影响。

---

## 6. 怎么跑 + 验收 + 消融

**从零训练**（tmux，空闲卡）：
```bash
CUDA_VISIBLE_DEVICES=<idle> /mnt/conda/jiaxuan/miniconda3/envs/dexvlg/bin/python \
  train.py --config configs/v3_reason.yaml
```
推理→转换→bench：与既有链**完全一致**（`infer_dataset.py` / `prepare_bench_data.py` 未动），照旧四通道跑。

**先验证（按序）**：
1. `python test_sanity.py` → 11 个（旧 10 + GRACE）全绿：证明默认逐位一致 + GRACE 前向/掩码/backward/采样。
2. 真实一个 batch 过 dataset（`grace_targets.enabled`）：断言每个在场手 `grasp_center` 落在该样本 `xyz` 包围盒内、
   `‖approach_dir‖≈1`——抓那唯一会"静默但致命"的帧/尺度 bug。
3. 画 ~10 个 lgbidex 样本的 WHERE 热力图（`xyz3` 上）；若 blob 偏离手部区域，只调 `data.grace_targets`
   的 `palm_forward_axis/palm_offset`（**只动目标导出，永不动输出**）。
4. `tools/make_gt_pred.py` 过一遍 convert→eval（用 `v3_reason.yaml`）复核 GT 基线（left/lgbidex/bidex 60/60）——
   GRACE 与模型无关不改它，这步只是给 config 路径背书。

**消融表（每行动一个具名指标）**：

| # | 配置 | 主要移动 |
|---|---|---|
| 0 | GRACE off | 对照——与 baseline 逐位一致 |
| 1 | 完整 GRACE | lgbidex/left rot 98°→↓、几何合格率 2.97%→↑、rel_r 1.97→↓ |
| 2 | 仅 HOW（`w_contact=w_rel_rot=0`） | 把旋转增益归因到接近机制 |
| 3 | 仅 WHERE（`w_anchor=w_approach=0`） | 平移 + structure_acc + 小幅 rot（有监督替代旧池） |
| 4 | **M=1**（单锚） | rot 打回 ~98°——旗舰因果证明 |
| 5 | M∈{16,32,64} | rot 升降 + 报告锚 top-1 acc（覆盖 vs 可学） |
| 6 | supervise-but-don't-condition（`condition:false`） | 若 rot 几乎不动 → 因果杠杆是 token **条件**而非表征 |
| 7 | RELATE off（`w_rel_rot=0`） | 双手 rel_rot + right_release_failure% |
| 8 | 旧 affordance on vs off | ≈0——"它什么都没做"的动机图 |
| 9 | **Oracle**：推理喂 GT `approach_dir` token | 完美 2-DOF 约束下的 rot 上界 → 残余 roll 的 headroom 图 |
| 10 | + `joint_hand_denoise:true` | 进一步压 rel_r（与 A1 组合；**需先补 joint 路径的 GRACE 接线 + 测试**）|

> 注：GRACE 目前只接线并测试了 **sequential** flow 路径；`joint_hand_denoise+use_grace` 被**显式
> `NotImplementedError`**（避免上线未验证的分支）。旗舰结果 0–9 全走 sequential；行 10 作为后续小改。

---

## 7. 诚实的天花板（写给 reviewer，不是 bug）

- 接近方向只钉 2/3 自由度，残余 roll 是 flow 的活——行 9 oracle 精确量化它还剩多少。
- 接触场是 64 个较粗中心上的**可抓区域**（尺度感知 σ，非指尖级接触）；ê 是模板、看图微调，ê 错只是平移
  确定性目标（削弱、绝不腐蚀，且永不触及输出）。
- 这些都明说、且各有一行消融——正是这份**可证伪性**让故事撑得起论文。

---

## 8. 明确不做 / 与既有诊断的关系

- ✗ 改输出位姿参数化 / 锚系残差 flow（约定风险高；本方案用"仅条件化"拿到同样的分类-精修收益）。
- ✗ 读 `contact_area` 作条件（字段全 null）——改用自监督腕部几何。
- ✗ 打开 `joint_hand_denoise + GRACE`（未测；已 `NotImplementedError` 拦截）。
- ✓ 与既有杠杆正交可叠加：`rot_geodesic`（磨 roll）、CFG（瓶颈 3，`drop_prob:0.1` + 推理只对 lgbidex 扫
  `guidance_scale`）、`flow_loss_task_scales`。
- 本方案即诊断文档 **A2+A3** 的一次性安全实现，把"reasoning"从装饰变成可测量、可消融、对症的量。
