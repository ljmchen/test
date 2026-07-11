#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_dgtr_diversity.py  —  多样性指标（复用 GAYS / DGTR 官方实现，左右手分别）

必须在 dexgys 环境运行:
    source /mnt/conda/jiaxuan/miniconda3/etc/profile.d/conda.sh && conda activate dexgys

本脚本直接 import 两套参考实现（不重写数学）:
  1) DGTR:  /home/jiaxuan/isee2025/DGTR/tools/evaluate.py :: DiversityEvalator
            -> var_translation / var_joint_angle / entropy_joints
  2) GAYS:  /home/jiaxuan/isee2025/Grasp-as-You-Say/scripts/diversity/calc_diversity.py
            :: DiversityEvalator
            -> 更完整（rotation 欧拉度、平移 cm、std、real_var 循环版）
两者都会输出。

多样性纯 torch 张量统计, 不用 csdf / CUDA, 因此可在 CPU 上跑（对 RTX 5090 + torch1.10
的 CUDA hang 免疫）。

输入: 由 gays_convert_bimanual.py 产出的 results_left.json / results_right.json
      （每条含 obj_id / predictions=N×28）。

用法:
  python eval_dgtr_diversity.py -l results_left.json -r results_right.json -o div.json
"""
import argparse
import copy
import json
import os
import sys

# 参考仓库路径
GAYS_ROOT = "/home/jiaxuan/isee2025/Grasp-as-You-Say"
DGTR_ROOT = "/home/jiaxuan/isee2025/DGTR"

# GAYS 版 DiversityEvalator（含 rotation / std / real_var / cm）
sys.path.insert(0, os.path.join(GAYS_ROOT, "scripts", "diversity"))
from calc_diversity import DiversityEvalator as GAYSDiversity  # noqa: E402

# DGTR 版 DiversityEvalator（var_translation / var_joint_angle / entropy_joints）
# 直接从源文件加载它的类（不跑它的 __main__ / 多进程 eval）。
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "dgtr_evaluate", os.path.join(DGTR_ROOT, "tools", "evaluate.py"))
_dgtr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dgtr)
DGTRDiversity = _dgtr.DiversityEvalator


def _prep_for_gays(results):
    """GAYS 版 squeeze(1) 期望 predictions 为 (N,1,28)。这里补一个中间维。"""
    out = []
    for r in results:
        r2 = copy.deepcopy(r)
        preds = r2["predictions"]
        # predictions: list[N][28] -> list[N][1][28]
        r2["predictions"] = [[p] for p in preds]
        out.append(r2)
    return out


def _clamp_joints_to_limits(results):
    """把每条预测的 22 个关节(28D 的 [6:28]) clamp 到 DGTR joint_limits。

    原因：DGTR `_calc_entropy` 按各关节物理限位分 100 bin 做直方图；若某关节的一组取值
    **全部落在限位外**，直方图求和=0 -> 概率 0/0 = nan -> Categorical 崩溃。DGTR 原数据在
    限位内故不触发；本项目模型会输出少量超限关节。GAYS 的 calc_diversity 本身也先对关节
    clamp 到限位再统计，故此处与 GAYS 处理一致（只夹关节，不动平移/旋转）。"""
    import torch
    JL = torch.tensor(DGTRDiversity.joint_limits, dtype=torch.float)  # (22,2)
    lo, hi = JL[:, 0], JL[:, 1]
    for r in results:
        P = torch.tensor(r["predictions"], dtype=torch.float)  # (N,28)
        P[:, 6:28] = torch.max(torch.min(P[:, 6:28], hi), lo)
        r["predictions"] = P.tolist()
    return results


def _run_dgtr(results):
    """DGTR 版直接吃 (N,28)；它按 res['obj_id']/pose_id/scale_id 生成 key，
    这里注入 metric['details'] 骨架，并保证 get_sample_key 可用。"""
    ev = DGTRDiversity.__new__(DGTRDiversity)
    ev.results = _clamp_joints_to_limits(copy.deepcopy(results))
    # 补齐 get_sample_key 需要的字段 & metric 骨架
    ev.metric = {"details": {}}
    for r in ev.results:
        r.setdefault("obj_id", r.get("obj_code", "unk"))
        # get_sample_key = f'{obj_id}_pose{pose_id}_scale{scale_id}'。本项目每组唯一键是
        # (task_type, obj_id, action_id, pose_id, scale_id)（见 convert_to_gays.py），故保留
        # 真实 pose_id、把 scale_id 置为 "action_id_真实scale_id" 组合使每组 key 唯一 ——
        # bidex 同 obj/pose 有多尺度，只用 action_id 会撞 key（不改任何多样性数学，
        # 只影响 per-object 明细分组）。
        r.setdefault("pose_id", r.get("guidance", "0"))
        r["scale_id"] = (f"{r.get('action_id', r.get('guidance', '0'))}"
                         f"_{r.get('scale_id', '0')}")
        key = DGTRDiversity.get_sample_key(r)
        ev.metric["details"][key] = {"mean": {}}
    return ev.calc_diversity()


def run_side(name, results):
    if not results:
        return {"side": name, "n": 0, "note": "empty"}
    gays = GAYSDiversity(_prep_for_gays(results)).calc_diversity()
    dgtr = _run_dgtr(results)
    return {
        "side": name,
        "n_entries": len(results),
        "n_grasps": sum(len(r["predictions"]) for r in results),
        "gays_diversity": gays["diversity"],   # overall + mean_object (rotation/std/real_var)
        "dgtr_diversity": dgtr["diversity"],   # overall + mean_object (var_trans/var_joints/entropy)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--left", default=None)
    ap.add_argument("-r", "--right", default=None)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    out = {}
    for name, path in [("left", args.left), ("right", args.right)]:
        if path is None:
            continue
        with open(path) as f:
            res = json.load(f)
        print(f"[{name}] {len(res)} entries from {path}")
        out[name] = run_side(name, res)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    # 打印样例数值
    for name in ("left", "right"):
        if name not in out or out[name].get("n", 1) == 0:
            continue
        d_dgtr = out[name]["dgtr_diversity"]["overall"]
        d_gays = out[name]["gays_diversity"]["overall"]
        print(f"\n=== {name.upper()} (n_grasps={out[name]['n_grasps']}) ===")
        print(f"  DGTR overall: var_trans={d_dgtr['mean_var_trans']:.5f} "
              f"var_joints={d_dgtr['mean_var_joints']:.5f} "
              f"entropy={d_dgtr['mean_entropy_joints']:.4f}")
        print(f"  GAYS overall: var_trans(cm)={d_gays['mean_var_trans']:.4f} "
              f"var_joints(deg)={d_gays['mean_var_joints']:.4f} "
              f"var_rot(deg)={d_gays['mean_var_rotation']:.4f} "
              f"std_joints(deg)={d_gays['mean_std_joints']:.4f}")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
