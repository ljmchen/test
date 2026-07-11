#!/usr/bin/env python3
"""Split a held-out validation set from train_fuse_v2 by obj_id (object level).

Motivation (审查发现 J): v3_multitask 此前 val_data == test_data，ckpt 选型用了
评测集。这里从 train_fuse_v2（3k+ 物体，与 test 零重叠）按 obj_id 随机划出 ~5%
物体作为 held-out val，train/val 物体零交集，供之后启动的训练做无偏选型。

Method: 收集全部 obj_id 并**排序**（消除 set 迭代顺序不确定性），用固定 seed 的
``random.Random(seed).sample`` 抽 ``round(N * val_frac)`` 个物体为 val；随后单遍
分派记录。同一物体的所有 task_type 记录进同一侧。

Outputs (written next to --train-path by default):
  train_fuse_v2_split.json        剩余 ~95% 物体的全部记录（训练用）
  val_fuse_v2_split.json          val 物体的全部记录（验证用）
  val_fuse_v2_split_objects.json  val 物体清单 + 两侧 per-task_type 统计

Usage:
  python tools/split_val_from_train.py \
      --train-path /home/jiaxuan/data/pose_data/train_fuse_v2.json \
      --val-frac 0.05 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-path", default="/home/jiaxuan/data/pose_data/train_fuse_v2.json")
    ap.add_argument("--out-train", default=None,
                    help="default: <train_dir>/train_fuse_v2_split.json")
    ap.add_argument("--out-val", default=None,
                    help="default: <train_dir>/val_fuse_v2_split.json")
    ap.add_argument("--out-objects", default=None,
                    help="default: <train_dir>/val_fuse_v2_split_objects.json")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def tt_object_stats(records: list[dict]) -> tuple[Counter, dict[str, int]]:
    """Per-task_type record counts and per-task_type unique-object counts."""
    rec_cnt: Counter = Counter()
    objs_by_tt: dict[str, set] = defaultdict(set)
    for r in records:
        tt = str(r.get("task_type", "unknown"))
        rec_cnt[tt] += 1
        objs_by_tt[tt].add(str(r["obj_id"]))
    return rec_cnt, {tt: len(s) for tt, s in sorted(objs_by_tt.items())}


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_path)
    out_dir = train_path.parent
    out_train = Path(args.out_train) if args.out_train else out_dir / "train_fuse_v2_split.json"
    out_val = Path(args.out_val) if args.out_val else out_dir / "val_fuse_v2_split.json"
    out_objects = (
        Path(args.out_objects) if args.out_objects else out_dir / "val_fuse_v2_split_objects.json"
    )

    print(f"loading {train_path} ...", flush=True)
    with train_path.open("r") as f:
        data = json.load(f)
    print(f"total records = {len(data)}", flush=True)

    all_objs = sorted({str(r["obj_id"]) for r in data})
    n_val = max(1, round(len(all_objs) * args.val_frac))
    rng = random.Random(args.seed)
    val_objs = set(rng.sample(all_objs, n_val))
    print(
        f"objects total = {len(all_objs)}; val objects = {n_val} "
        f"({args.val_frac:.1%}, seed={args.seed})",
        flush=True,
    )

    train_recs: list[dict] = []
    val_recs: list[dict] = []
    for r in data:
        (val_recs if str(r["obj_id"]) in val_objs else train_recs).append(r)

    if len(train_recs) + len(val_recs) != len(data):
        raise RuntimeError(
            f"split leak: train {len(train_recs)} + val {len(val_recs)} != total {len(data)}"
        )
    train_obj_set = {str(r["obj_id"]) for r in train_recs}
    val_obj_set = {str(r["obj_id"]) for r in val_recs}
    inter = train_obj_set & val_obj_set
    if inter:
        raise RuntimeError(f"object overlap between train/val: {sorted(inter)[:5]} ...")

    stats = {}
    for name, recs, obj_set in (
        ("train", train_recs, train_obj_set),
        ("val", val_recs, val_obj_set),
    ):
        rec_cnt, obj_cnt = tt_object_stats(recs)
        stats[name] = {
            "records": len(recs),
            "objects": len(obj_set),
            "records_per_task_type": dict(sorted(rec_cnt.items())),
            "objects_per_task_type": obj_cnt,
        }
        print(
            f"[{name}] records={len(recs)} objects={len(obj_set)} "
            f"per_task_type={stats[name]['records_per_task_type']} "
            f"objects_per_task_type={obj_cnt}",
            flush=True,
        )

    print(f"writing {out_train} ...", flush=True)
    with out_train.open("w") as f:
        json.dump(train_recs, f)
    print(f"writing {out_val} ...", flush=True)
    with out_val.open("w") as f:
        json.dump(val_recs, f)

    manifest = {
        "source": str(train_path),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "split_rule": "sorted unique obj_id -> random.Random(seed).sample; object-level, no overlap",
        "objects_total": len(all_objs),
        "objects_val": n_val,
        "outputs": {"train": str(out_train), "val": str(out_val)},
        "stats": stats,
        "val_objects": sorted(val_objs),
    }
    with out_objects.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"writing {out_objects} ... done", flush=True)
    print(
        f"OK: train {len(train_recs)} + val {len(val_recs)} = {len(data)}; "
        f"object overlap = 0",
        flush=True,
    )


if __name__ == "__main__":
    main()
