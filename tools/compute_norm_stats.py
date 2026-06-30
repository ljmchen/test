#!/usr/bin/env python3
"""Compute per-hand mean/std normalization factors (for meastd11) from a split.

Processes records through the dataset's centering + 6D conversion (with
normalization OFF) so the statistics match the actual training targets, then
reports per-component mean/std for translation(3) and joints(22) of each hand.
Rotation (6D) is not meastd-normalized, so it is omitted.

Usage:
  python tools/compute_norm_stats.py \\
    --split-path /mnt/afs/L202500241/asserts/lgbidex/pose_data/train_v3_sim3.json \\
    --mesh-root  /mnt/afs/L202500241/asserts/lgbidex/oakink_obj/processed_data \\
    --max-records 4000 --out tools/meastd_factors.json
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
    ap.add_argument("--max-records", type=int, default=4000)
    ap.add_argument("--out", default="tools/meastd_factors.json")
    args = ap.parse_args()

    ds = DexGraspDataset(
        data_path=args.split_path, mesh_root=args.mesh_root, n_points=4096,
        joint_dim=22, augment=False, point_cloud_source="mesh_sample",
        point_cloud_color_mode="fill", center_on_object=True,
        obj_pose_quaternion_order="wxyz", normalization=None,
    )
    n = min(args.max_records, len(ds.data))
    print(f"records={len(ds.data)} using={n}", flush=True)

    acc = {"left": [], "right": []}
    for i in range(n):
        rec = ds.data[i]
        for side in ("left", "right"):
            pose31 = ds._build_hand_pose(rec, side)  # centered, 6D, NOT normalized
            trans = pose31[:3].numpy()
            joints = pose31[9:].numpy()
            acc[side].append(np.concatenate([trans, joints]))  # 3 + 22 = 25
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    out = {}
    for side in ("left", "right"):
        arr = np.asarray(acc[side])  # [n, 25]
        mean = arr.mean(0)
        std = arr.std(0)
        out[side] = {
            "translation_mean": mean[:3].tolist(),
            "translation_std": std[:3].tolist(),
            "joint_mean": mean[3:].tolist(),
            "joint_std": std[3:].tolist(),
        }
        print(f"\n=== {side} ===")
        print("trans mean:", np.round(mean[:3], 4).tolist())
        print("trans std :", np.round(std[:3], 4).tolist())
        print("joint std :", np.round(std[3:], 4).tolist())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
