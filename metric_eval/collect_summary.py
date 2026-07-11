#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""collect_summary.py — 把 diversity/q1/fid 三个结果 JSON 汇总成一条记录，追加到累积记录文件。

单独可用（不重跑评测）：只要 results/ 下有对应的 *_<tag>.json，就能生成/更新汇总。
缺哪个就在表里标 "—"。由 run_all_metrics.sh 自动调用，也可手动跑：

  python collect_summary.py --tag e0115 --input <预测.json> \
    --div results/diversity_e0115.json --q1 results/q1_e0115.json --fid results/fid_e0115.json \
    --out results/metrics_record.md
"""
import argparse
import json
import os
from datetime import datetime


def load(p):
    if not p or not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def g(d, *keys, default="—"):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def fmt(x, nd=5):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--input", default="")
    ap.add_argument("--div", default=None)
    ap.add_argument("--q1", default=None)
    ap.add_argument("--fid", default=None)
    ap.add_argument("--out", required=True, help="累积记录文件（追加）")
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()

    div, q1, fid = load(args.div), load(args.q1), load(args.fid)
    stamp = args.stamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    npred = g(div, "left", "n_grasps") if div else (g(fid, "left", "n_pred") if fid else "—")

    rows = []
    if div:
        for lab, key in [("DGTR var_trans (m²)", "mean_var_trans"),
                         ("DGTR var_joints (rad²)", "mean_var_joints"),
                         ("DGTR entropy_joints", "mean_entropy_joints")]:
            rows.append((lab, g(div, "left", "dgtr_diversity", "overall", key),
                         g(div, "right", "dgtr_diversity", "overall", key), 5))
        for lab, key in [("GAYS var_rot (deg²)", "mean_var_rotation"),
                         ("GAYS std_joints (deg)", "mean_std_joints"),
                         ("GAYS var_trans (cm²)", "mean_var_trans")]:
            rows.append((lab, g(div, "left", "gays_diversity", "overall", key),
                         g(div, "right", "gays_diversity", "overall", key), 4))
    if q1:
        for lab, key in [("Q1 (越大越好)", "q1"), ("pen (m)", "pen"), ("valid_q1", "valid_q1")]:
            rows.append((lab, g(q1, "left", "mean_overall", key, "mean"),
                         g(q1, "right", "mean_overall", key, "mean"), 5))
    if fid:
        rows.append(("FID (越小越好)", g(fid, "left", "fid"), g(fid, "right", "fid"), 4))

    lines = [f"\n## {args.tag}  ({stamp})"]
    if args.input:
        lines.append(f"- 输入: `{args.input}`")
    lines.append(f"- 样本(每手): {npred}")
    have = [n for n, d in [("diversity", div), ("q1", q1), ("fid", fid)] if d]
    miss = [n for n, d in [("diversity", div), ("q1", q1), ("fid", fid)] if not d]
    lines.append(f"- 已含: {', '.join(have) or '无'}" + (f" ｜ 缺: {', '.join(miss)}" if miss else ""))
    lines.append("- 覆盖范围: diversity=全部四类 task_type(left/right/lgbidex/bidex)；"
                 "Q1/FID=仅 left/lgbidex（right/bidex 无 OakInk mesh，已显式跳过）")
    lines.append("")
    lines.append("| 指标 | 左手 | 右手 |")
    lines.append("|---|---|---|")
    for lab, l, r, nd in rows:
        lines.append(f"| {lab} | {fmt(l, nd)} | {fmt(r, nd)} |")
    block = "\n".join(lines) + "\n"

    new_file = not os.path.exists(args.out)
    with open(args.out, "a") as f:
        if new_file:
            f.write("# exp-pami/test 指标评测累积记录\n\n"
                    "> 每次 `run_all_metrics.sh` / `collect_summary.py` 追加一条。\n"
                    "> 指标定义与坐标/尺度约定见同目录 `信息记录.md`。\n")
        f.write(block)
    print(block)
    print(f"[collect] 追加到 {args.out}")


if __name__ == "__main__":
    main()
