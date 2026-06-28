#!/usr/bin/env python3
"""Two-in-one: predictions JSON -> pregrasp/squeeze -> dgbench-pami eval3 graspdata.

Combines the dexvlm pipeline's two steps into a single script for the test
project:

  1. ``get_pre_squ``  : from each record's ``pred_left`` / ``pred_right`` (28-d
     ``[trans(3), axis_angle(3), joints(22)]``) compute the pregrasp and squeeze
     poses, producing ``dex_{grasp,pregrasp,squeeze}_{left,right}``;
  2. ``prepare_eval3``: convert each 28-d GYS pose to the bench's 29-d qpos
     (axis-angle -> wxyz quaternion, palm offset, Shadow sign fixes) and save one
     ``.npy`` per grasp under ``<output_root>/<exp_name>_<hand_name>/graspdata``.

The pose math is copied verbatim from the dexvlm references
(``get_pre_squ_dexvlm.py`` and ``prepare_lgbidex_eval3.py``) so the produced
graspdata is byte-compatible with the existing ``dexvlm_bench_all.sh`` flow.

Usage:
  python prepare_bench_data.py \\
    --input outputs/predictions_test.json \\
    --mesh-root /mnt/afs/L202500241/Dataset/MeshProcess/assets/object/oakink_obj/processed_data \\
    --output-root /mnt/afs/L202500241/temp/dgbench-pami/output \\
    --exp-name slaitest_RUNID_bestXX --hand-name shadow_left --numworker 16
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np

SINGLE_HAND_DIM = 28
MOVE_LOCAL_Y = np.array([0.0, 0.0, 0.0], dtype=np.float64)
PALM_OFFSET = np.array([0.0, 0.0, 0.034], dtype=np.float32)

POSE_KEYS = (
    "dex_pregrasp_left", "dex_grasp_left", "dex_squeeze_left",
    "dex_pregrasp_right", "dex_grasp_right", "dex_squeeze_right",
)
METADATA_KEYS = (
    "obj_id", "obj_key", "oakink_obj_id", "dexgys_obj_id", "cate_id", "pose_id",
    "scale_id", "action_id", "guidance", "guidence", "contact_area", "grasp_id",
    "index_in_obj", "candidate_idx", "seq_id",
)


# ───────────────────────── pre / squeeze (get_pre_squ) ─────────────────────


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(axis_angle)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis_angle / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def flatten_pose(value, name: str = "pose") -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    while pose.ndim > 1 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.shape != (SINGLE_HAND_DIM,):
        raise ValueError(f"{name} must be 28D, got shape {pose.shape}.")
    return pose


def compute_pregrasp(grasp_pose: np.ndarray) -> np.ndarray:
    grasp_pose = flatten_pose(grasp_pose, "grasp_pose")
    pre = grasp_pose.copy()
    rot_mat = axis_angle_to_matrix(grasp_pose[3:6])
    pre[:3] = grasp_pose[:3] + rot_mat @ MOVE_LOCAL_Y
    pre[7:10] -= 0.1
    pre[11:14] -= 0.1
    pre[15:18] -= 0.1
    pre[20:23] -= 0.1
    pre[25:28] += 0.1
    return pre


def compute_squeeze(grasp_pose: np.ndarray, pre_pose: np.ndarray) -> np.ndarray:
    grasp_pose = flatten_pose(grasp_pose, "grasp_pose")
    pre_pose = flatten_pose(pre_pose, "pre_pose")
    squ = grasp_pose.copy()
    squ[:3] = 2.0 * grasp_pose[:3] - pre_pose[:3]
    squ[6:] = 2.0 * grasp_pose[6:] - pre_pose[6:]
    return squ


def add_pre_squeeze(record: dict, idx: int) -> dict:
    if "pred_left" not in record or "pred_right" not in record:
        raise KeyError(f"Record {idx} must contain 'pred_left' and 'pred_right'.")
    grasp_left = flatten_pose(record["pred_left"], f"record[{idx}].pred_left")
    grasp_right = flatten_pose(record["pred_right"], f"record[{idx}].pred_right")
    pre_left = compute_pregrasp(grasp_left)
    pre_right = compute_pregrasp(grasp_right)
    record["dex_grasp_left"] = grasp_left.tolist()
    record["dex_grasp_right"] = grasp_right.tolist()
    record["dex_pregrasp_left"] = pre_left.tolist()
    record["dex_pregrasp_right"] = pre_right.tolist()
    record["dex_squeeze_left"] = compute_squeeze(grasp_left, pre_left).tolist()
    record["dex_squeeze_right"] = compute_squeeze(grasp_right, pre_right).tolist()
    record.pop("pred_left", None)
    record.pop("pred_right", None)
    return record


# ──────────────────────── bench qpos (prepare_eval3) ───────────────────────


def axis_angle_to_quaternion_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = axis_angle / angle
    half = 0.5 * angle
    return np.concatenate(
        [np.array([np.cos(half)], dtype=np.float32), axis * np.sin(half)]
    ).astype(np.float32)


def quaternion_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(q, dtype=np.float64)
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def gys_pose_to_bench_qpos(value: Iterable[float], name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float32)
    if pose.ndim != 1:
        pose = np.squeeze(pose)
    if pose.shape != (28,):
        raise ValueError(f"{name} must be 28D GYS pose, got {pose.shape}.")
    pose = pose.copy()
    # Shadow-hand joint sign fixes (verbatim from prepare_lgbidex_eval3.py).
    pose[-1] = -pose[-1]
    pose[-2] = -pose[-2]
    pose[6] = -pose[6]
    pose[10] = -pose[10]
    quaternion = axis_angle_to_quaternion_wxyz(pose[3:6])
    rotation = quaternion_wxyz_to_matrix(quaternion)
    translation = pose[:3].astype(np.float64) - rotation @ PALM_OFFSET.astype(np.float64)
    qpos = np.concatenate([translation.astype(np.float32), quaternion, pose[6:]], axis=0)
    if qpos.shape != (29,) or not np.all(np.isfinite(qpos)):
        raise ValueError(f"{name} produced invalid qpos {qpos.shape}.")
    return qpos.astype(np.float32)


def resolve_object_path(mesh_root: str | Path, object_id: str) -> Path:
    object_path = Path(mesh_root) / object_id
    required = [
        object_path / "info" / "simplified.json",
        object_path / "mesh" / "simplified.obj",
        object_path / "urdf" / "meshes",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing object assets for {object_id}: {', '.join(missing)}")
    return object_path.resolve()


def sample_key(record: dict) -> tuple:
    return (
        str(record.get("obj_id", "")), str(record.get("scale_id", "")),
        str(record.get("pose_id", "")), str(record.get("action_id", "")),
        str(record.get("grasp_id", "")),
    )


def format_id(value, width: int = 3) -> str:
    try:
        return f"{int(value):0{width}d}"
    except (TypeError, ValueError):
        return str(value)


def relative_sample_dir(record: dict) -> Path:
    obj_id = str(record["obj_id"])
    pose_id = format_id(record.get("pose_id", 0))
    action_id = str(record.get("action_id", ""))
    grasp_id = format_id(record.get("grasp_id", 0))
    scale_id = record.get("scale_id")
    if scale_id is None:
        leaf = f"pose{pose_id}_act{action_id}_grasp{grasp_id}"
    else:
        leaf = f"scale{format_id(scale_id)}_pose{pose_id}_act{action_id}_grasp{grasp_id}"
    return Path(obj_id) / leaf


def make_eval3_payload(record: dict, variant_idx: int, mesh_root: str, json_path: str) -> dict:
    object_path = resolve_object_path(mesh_root, str(record["obj_id"]))
    payload = {
        "obj_path": str(object_path),
        "obj_pose": np.asarray(record["obj_pose"], dtype=np.float32),
        "obj_scale": float(record.get("obj_scale", 0.1)),
        "left_pregrasp_qpos": gys_pose_to_bench_qpos(record["dex_pregrasp_left"], "dex_pregrasp_left"),
        "left_grasp_qpos": gys_pose_to_bench_qpos(record["dex_grasp_left"], "dex_grasp_left"),
        "left_squeeze_qpos": gys_pose_to_bench_qpos(record["dex_squeeze_left"], "dex_squeeze_left"),
        "right_pregrasp_qpos": gys_pose_to_bench_qpos(record["dex_pregrasp_right"], "dex_pregrasp_right"),
        "right_grasp_qpos": gys_pose_to_bench_qpos(record["dex_grasp_right"], "dex_grasp_right"),
        "right_squeeze_qpos": gys_pose_to_bench_qpos(record["dex_squeeze_right"], "dex_squeeze_right"),
        "raw_input_path": json_path,
        "candidate_idx": int(record.get("candidate_idx", variant_idx)),
    }
    for key in METADATA_KEYS:
        if key in record and key not in payload:
            payload[key] = record[key]
    payload["sample_key"] = "__".join(sample_key(record))
    payload["variant_idx"] = int(variant_idx)
    return payload


def save_task(task, *, mesh_root: str, json_path: str) -> None:
    record, variant_idx, save_path = task
    payload = make_eval3_payload(record, variant_idx, mesh_root, json_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="predictions JSON -> pre/squeeze -> dgbench eval3 graspdata (two-in-one)."
    )
    parser.add_argument("-i", "--input", required=True, help="predictions JSON from infer_dataset.py")
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--output-root", required=True, help="dgbench-pami output root.")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--hand-name", default="shadow_left")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--save-json", default=None,
                        help="Optional path to also dump the intermediate *_4bench.json.")
    parser.add_argument("--num-workers", "--numworker", "--num_worker",
                        dest="num_workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.input).open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Input JSON root must be a list of records.")

    # Stage 1: compute pre / squeeze poses.
    for idx, record in enumerate(records):
        add_pre_squeeze(record, idx)

    if args.save_json:
        with Path(args.save_json).open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[info] wrote intermediate 4bench json -> {args.save_json}")

    # Stage 2: validate, group, convert to bench qpos and save .npy.
    for idx, record in enumerate(records):
        missing = [k for k in ("obj_id", "obj_pose", *POSE_KEYS) if k not in record]
        if missing:
            raise KeyError(f"Record {idx} missing required keys: {missing}")

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[sample_key(record)].append(record)
    sample_keys = sorted(grouped)
    if args.max_samples > 0:
        sample_keys = sample_keys[: args.max_samples]

    exp_dir = Path(args.output_root) / f"{args.exp_name}_{args.hand_name}"
    grasp_dir = exp_dir / "graspdata"
    if args.clean and exp_dir.exists():
        shutil.rmtree(exp_dir)
    grasp_dir.mkdir(parents=True, exist_ok=True)

    save_tasks = []
    for key in sample_keys:
        sample_records = sorted(grouped[key], key=lambda r: int(r.get("candidate_idx", 0)))
        for variant_idx, record in enumerate(sample_records):
            save_path = grasp_dir / relative_sample_dir(record) / f"{variant_idx}.npy"
            save_tasks.append((record, variant_idx, save_path))

    worker = partial(save_task, mesh_root=args.mesh_root, json_path=str(args.input))
    if args.num_workers <= 1 or len(save_tasks) <= 1:
        for task in save_tasks:
            worker(task)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            for _ in ex.map(worker, save_tasks):
                pass

    print(f"Loaded records: {len(records)}")
    print(f"Prepared samples: {len(sample_keys)}")
    print(f"Saved eval3 records: {len(save_tasks)}")
    print(f"Output graspdata: {grasp_dir}")
    print(f"Use save_dir: {exp_dir}")


if __name__ == "__main__":
    main()
