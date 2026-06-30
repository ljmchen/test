#!/usr/bin/env python3
"""Statistics of object-centered grasp poses, for choosing normalization factors.

For each hand reports, on the centered 28-d pose
``[translation(3), axis_angle(3), joints(22)]``:
  - translation 1% / 99% percentiles per axis (a robust [min,max] for minmax11);
  - mean and std of all 28 dims (for a meastd-style standardizer).

Usage:
  python tools/compute_pose_stats.py \\
    --split-path /mnt/afs/L202500241/asserts/lgbidex/pose_data/train_v4.json \\
    --mesh-root  /mnt/afs/L202500241/asserts/lgbidex/oakink_obj/processed_data \\
    --max-records 100000 --out tools/pose_stats_v4.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import DexGraspDataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-path", required=True)
    ap.add_argument("--mesh-root", required=True)
    ap.add_argument("--max-records", type=int, default=100000)
    ap.add_argument("--out", default="tools/pose_stats.json")
    args = ap.parse_args()

    ds = DexGraspDataset(
        data_path=args.split_path, mesh_root=args.mesh_root, n_points=4096,
        joint_dim=22, augment=False, point_cloud_source="mesh_sample",
        point_cloud_color_mode="fill", center_on_object=True,
        obj_pose_quaternion_order="wxyz", normalization=None,
    )
    n = min(args.max_records, len(ds.data))
    # evenly strided sample for broad object coverage
    step = max(1, len(ds.data) // n)
    idxs = list(range(0, len(ds.data), step))[:n]
    print(f"records={len(ds.data)} sampling={len(idxs)} (stride {step})", flush=True)

    acc = {"left": [], "right": []}
    for c, i in enumerate(idxs):
        rec = ds.data[i]
        for side in ("left", "right"):
            raw = torch.tensor(rec[f"dex_grasp_{side}"], dtype=torch.float32)
            centered = ds._center_hand_pose(raw, rec)  # 28-d, translation centered
            acc[side].append(centered.numpy())
        if (c + 1) % 5000 == 0:
            print(f"  {c + 1}/{len(idxs)}", flush=True)

    out = {}
    for side in ("left", "right"):
        arr = np.asarray(acc[side])  # [n, 28]
        p1 = np.percentile(arr, 1, axis=0)
        p99 = np.percentile(arr, 99, axis=0)
        mean = arr.mean(0)
        std = arr.std(0)
        out[side] = {
            "translation_p1": p1[:3].tolist(),
            "translation_p99": p99[:3].tolist(),
            "mean_28": mean.tolist(),
            "std_28": std.tolist(),
        }
        rnd = lambda a: np.round(a, 4).tolist()  # noqa: E731
        print(f"\n===== {side} =====")
        print("centered translation 1%/99% (x,y,z):")
        for ax, lo, hi in zip("xyz", p1[:3], p99[:3]):
            print(f"  {ax}: [{lo:.4f}, {hi:.4f}]")
        print("28-d mean:", rnd(mean))
        print("28-d std :", rnd(std))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
