#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_to_gays.py  —  exp-pami/test 专用：把 predictions_*.json 转成 GAYS/DGTR 评测输入。

本项目 (DexVLG v3 多任务) 的预测 JSON 每条记录字段（已核对 predictions_v2e29_s5_test.json）:
  task_type   : left | right | lgbidex | bidex     —— 四通道；本转换器全部保留
  obj_id      : 友好名 (e.g. bottle_s102 / 040_large_marker / sem_TissueBox_*)
  obj_id_old  : OakInk 码 (e.g. A16012)。仅 left/lgbidex 有；right(YCB)/bidex(sem_*) 无
                —— 可选字段，只影响 Q1/FID 的 mesh 解析，有则带上
  obj_pose    : 7D [xyz, quat wxyz]  (world 系)
  action_id   : 动作码 (e.g. 0004)。bidex 无 -> 组键用 task_type 兜底
  pose_id     : 物体摆放 id；scale_id：物体尺度 id（bidex 同 obj/pose 有多尺度）
  pred_left / pred_right       : 28D 预测 [平移(3)+轴角(3)+关节(22)]  (world 系)
                （单手记录只带自己那侧的 pred_*，另一键缺失）
  dex_grasp_left / dex_grasp_right : 28D 真值 (world 系)

关键适配点（与 dgbench-pami/tools 通用转换器的差别）:
  1) 本项目 pred/GT 是 **world 系**（pred_pose_frame="world"，实测 pred 平移≈obj_pose 平移）。
     GAYS/DGTR 评测器期望 **物体规范系**（物体在原点）。故每条按其自身 obj_pose 变换:
         t' = R_oᵀ (t_h − t_o) ;  R' = R_oᵀ R_h ;  关节不变
     （组内 obj_pose 有微小抖动 —— 逐条用各自 obj_pose 才正确。）
  2) 每条记录是**一个抓取**；GAYS/DGTR 多样性/FID 需要"每 (obj,动作,摆放) 组多条候选"。
     故按 (task_type, obj_id, action_id, pose_id, scale_id) 分组，组内所有 pred 汇成
     predictions=[N×28]。bidex 缺 action_id 时用 f"tt_{task_type}" 兜底；scale_id 入组键
     防 bidex 同 obj/pose 多尺度塌成一组（OakInk 三类 scale_id 恒定，分组不变）。
  3) GAYS mesh 用 OakInk 码 -> 取 **obj_id_old**（缺失时输出 obj_id 回退友好名，仅供分组；
     Q1/FID 侧按 obj_id_old 有无过滤）。

覆盖范围: 多样性(diversity)吃全部四类 task_type；Q1/FID 只能吃 left/lgbidex
（right/bidex 无 OakInk coacd mesh，由 eval_gays_q1/fid 显式过滤并打印跳过数）。

  >>> 只做刚体系变换，不翻关节符号：本项目 dex 已是 robot0/GYS 约定（见项目 CLAUDE.md /
      prepare_bench_data.py），与 GAYS/DGTR 关节约定一致，直接可用。

⚠️ Q1/FID 的 mesh 对齐（仅影响 Q1/FID，不影响多样性）:
   GAYS 加载 OakInk mesh 时会 bbox 居中(vertices-=bbox_center) 且用其自身尺度；本项目
   obj_scale=0.1。若在兼容 GPU 上跑 Q1/FID，需确认 obj_pose 定义的规范系与 GAYS bbox 居中
   /缩放一致（可能还需减 bbox_center 或按 obj_scale 缩放）。多样性只用位姿统计，与 mesh 无关，
   不受此影响。输出里保留 obj_scale 供后续对齐。

输出（写到 --out_dir）:
  results_left.json / results_right.json，每条:
    { "obj_id": <OakInk码，无则友好名>, "obj_id_old": <OakInk码|None>,
      "obj_name": <友好名>, "task_type": <left|right|lgbidex|bidex>,
      "guidance": <action_id|tt_兜底>, "action_id","pose_id","scale_id","obj_scale",
      "predictions": [[28]...N 物体系预测],
      "dex_grasps":  [[28]...M 物体系真值] }

用法:
  python convert_to_gays.py -i ../outputs/predictions_e0115_test.json -o results [--limit N]
"""
import argparse
import json
import os
from collections import Counter, OrderedDict

import numpy as np
from scipy.spatial.transform import Rotation


def quat_wxyz_to_R(q):
    q = np.asarray(q, float)
    return Rotation.from_quat(np.roll(q, -1)).as_matrix()  # wxyz -> xyzw


def world_to_object_28(pose28, obj_pose, obj_scale=None):
    """world 系 28D [t(3), axisangle(3), joints(22)] -> 物体规范系 28D。

    t' = R_oᵀ (t_h − t_o) / obj_scale ; R' = R_oᵀ R_h ; 关节不变。
    除以 obj_scale：本项目世界物体 = canonical(bbox 居中) mesh 经 obj_scale 缩放摆放，
    故平移除以 obj_scale 才落到 GAYS 加载 mesh 的 canonical 单位（旋转/关节与缩放无关）。
    obj_scale=None 时不除（保持世界米制的物体系）。"""
    pose28 = np.asarray(pose28, float)
    t_h, aa, joints = pose28[:3], pose28[3:6], pose28[6:]
    t_o = np.asarray(obj_pose[:3], float)
    R_o = quat_wxyz_to_R(obj_pose[3:7])
    R_h = Rotation.from_rotvec(aa).as_matrix()
    t_rel = R_o.T @ (t_h - t_o)
    if obj_scale:
        t_rel = t_rel / float(obj_scale)
    aa_rel = Rotation.from_matrix(R_o.T @ R_h).as_rotvec()
    return np.concatenate([t_rel, aa_rel, joints]).tolist()


def build(records, side, div_obj_scale=True):
    """按 (task_type, obj_id, action_id, pose_id, scale_id) 分组，返回 GAYS 格式 list。

    守卫只看 pred_{side} 与 obj_id（obj_id_old 变为可选，有则带上供 Q1/FID）；
    单手记录只带自己那侧 pred_*，另一侧构建时按 no_pred 跳过属正常。
    bidex 缺 action_id -> 组键用 f"tt_{task_type}" 兜底，避免整个 task_type 塌成一组。

    Returns:
        entries: GAYS 格式 list。
        kept_by_tt / skip_by_tt: 每 task_type 的保留/跳过条数 Counter。
    """
    pred_key, gt_key = f"pred_{side}", f"dex_grasp_{side}"
    groups = OrderedDict()
    kept_by_tt, skip_by_tt = Counter(), Counter()
    for r in records:
        tt = str(r.get("task_type", "unknown"))
        if pred_key not in r or r.get("obj_id") is None:
            skip_by_tt[tt] += 1
            continue
        kept_by_tt[tt] += 1
        oid_oak = r.get("obj_id_old")           # OakInk 码；right/bidex 无 -> None
        action = r.get("action_id")
        if action is None:
            action = f"tt_{tt}"
        key = (tt, r.get("obj_id"), action, r.get("pose_id"), r.get("scale_id"))
        g = groups.setdefault(key, {
            "obj_id": oid_oak if oid_oak is not None else r.get("obj_id"),
            "obj_id_old": oid_oak,              # 可选：Q1/FID mesh 解析用，无则 None
            "obj_name": r.get("obj_id"),
            "task_type": tt,
            "guidance": str(action),
            "action_id": str(action),
            "pose_id": r.get("pose_id"),
            "scale_id": r.get("scale_id"),
            "obj_scale": r.get("obj_scale"),
            "predictions": [],
            "dex_grasps": [],
        })
        op = r["obj_pose"]
        sc = r.get("obj_scale") if div_obj_scale else None
        g["predictions"].append(world_to_object_28(r[pred_key], op, sc))
        if gt_key in r and r[gt_key] is not None:
            g["dex_grasps"].append(world_to_object_28(r[gt_key], op, sc))
    return list(groups.values()), kept_by_tt, skip_by_tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only first N records (debug)")
    ap.add_argument("--div-obj-scale", action="store_true",
                    help="把平移除以 obj_scale。默认【不除】——实测本项目物体系平移|t|≈0.167 已与 "
                         "GAYS 的 dex_grasp|t|≈0.149 量级吻合；除以 0.1 会 10× 偏大。仅当确认本项目 "
                         "grasp 平移是 scaled-mesh 单位时才开。")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    for side in ("left", "right"):
        entries, kept_by_tt, skip_by_tt = build(data, side, div_obj_scale=args.div_obj_scale)
        n_pred = sum(len(e["predictions"]) for e in entries)
        n_gt = sum(len(e["dex_grasps"]) for e in entries)
        path = os.path.join(args.out_dir, f"results_{side}.json")
        with open(path, "w") as f:
            json.dump(entries, f)
        print(f"[{side}] {len(entries)} groups | {n_pred} preds | {n_gt} gts "
              f"| skip {sum(skip_by_tt.values())} -> {path}")
        print(f"   kept per task_type: {dict(sorted(kept_by_tt.items()))}")
        print(f"   skip per task_type: {dict(sorted(skip_by_tt.items()))} "
              f"(缺 pred_{side} 或 obj_id；单手记录只计入自己那侧属正常)")
        if entries:
            e = entries[0]
            print(f"   e0: obj_id={e['obj_id']} obj_id_old={e['obj_id_old']} "
                  f"task_type={e['task_type']} name={e['obj_name']} "
                  f"action={e['action_id']} pose={e['pose_id']} scale={e['scale_id']} "
                  f"n_pred={len(e['predictions'])} n_gt={len(e['dex_grasps'])}")


if __name__ == "__main__":
    main()
