#!/usr/bin/env python3
"""Export evenly sampled prediction visualizations as colored OBJ files.

Each exported OBJ contains the object mesh plus the model-emitted predicted
hand meshes. The latent-ar model may emit one or both hands; this script
visualizes exactly what was emitted instead of fabricating missing hands.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DGBENCH = "/home/jiaxuan/slai/dgbench-pami"
MESH_ROOT = "/home/jiaxuan/data/oakink_obj/processed_data"

OBJECT_COLOR = (0.62, 0.62, 0.64)
PRED_LEFT_COLOR = (0.18, 0.42, 0.95)
PRED_RIGHT_COLOR = (0.95, 0.36, 0.16)


def aa_to_quat_wxyz(axis_angle: list[float] | np.ndarray) -> np.ndarray:
    aa = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = aa / angle
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


def dex28_to_qpos29(pose28: list[float] | np.ndarray) -> np.ndarray:
    pose = np.asarray(pose28, dtype=np.float64)
    if pose.shape != (28,):
        raise ValueError(f"Expected 28-D pose, got shape {pose.shape}")
    quat = aa_to_quat_wxyz(pose[3:6])
    return np.concatenate([pose[:3], quat, pose[6:]]).astype(np.float32)


def sanitize_name(text: object, max_len: int = 64) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    value = value.strip("_") or "item"
    return value[:max_len]


def choose_even_indices(records: list[dict], per_task: int) -> list[int]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_task[str(record.get("task_type", ""))].append(i)

    selected: list[int] = []
    for task_type in ["left", "lgbidex", "right", "bidex"]:
        indices = by_task.get(task_type, [])
        if not indices:
            continue
        targets = np.linspace(0, len(indices) - 1, min(per_task, len(indices)))
        used_obj_ids: set[str] = set()
        used_indices: set[int] = set()
        for target in targets:
            center = int(round(float(target)))
            order = sorted(range(len(indices)), key=lambda j: (abs(j - center), j))
            chosen = None
            for j in order:
                idx = indices[j]
                obj_id = str(records[idx].get("obj_id", ""))
                if idx not in used_indices and obj_id not in used_obj_ids:
                    chosen = idx
                    break
            if chosen is None:
                for j in order:
                    idx = indices[j]
                    if idx not in used_indices:
                        chosen = idx
                        break
            if chosen is None:
                continue
            selected.append(chosen)
            used_indices.add(chosen)
            used_obj_ids.add(str(records[chosen].get("obj_id", "")))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-task", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(DGBENCH, "src"))
    os.chdir(DGBENCH)
    from task.vis_obj3 import _build_hand_mesh, _build_object_mesh, _write_colored_obj
    from util.hand_util import RobotKinematics

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{input_path} must contain a JSON list")

    selected = choose_even_indices(records, args.per_task)
    left_fk = RobotKinematics("assets/hand/shadow/left_hand.xml")
    right_fk = RobotKinematics("assets/hand/shadow/right_hand.xml")

    manifest = []
    for rank, idx in enumerate(selected):
        record = records[idx]
        obj_path = os.path.join(MESH_ROOT, str(record["obj_id"]))
        obj_pose = np.asarray(record["obj_pose"], dtype=np.float32)
        obj_scale = float(record.get("obj_scale", 0.1))

        obj_verts, obj_faces = _build_object_mesh(obj_path, obj_scale, obj_pose)
        groups = [
            {
                "name": "object",
                "verts": obj_verts,
                "faces": obj_faces,
                "color": OBJECT_COLOR,
            }
        ]

        emitted = []
        if record.get("pred_left") is not None:
            qpos = dex28_to_qpos29(record["pred_left"])
            verts, faces = _build_hand_mesh(left_fk, qpos)
            groups.append(
                {
                    "name": "pred_left",
                    "verts": verts,
                    "faces": faces,
                    "color": PRED_LEFT_COLOR,
                }
            )
            emitted.append("left")
        if record.get("pred_right") is not None:
            qpos = dex28_to_qpos29(record["pred_right"])
            verts, faces = _build_hand_mesh(right_fk, qpos)
            groups.append(
                {
                    "name": "pred_right",
                    "verts": verts,
                    "faces": faces,
                    "color": PRED_RIGHT_COLOR,
                }
            )
            emitted.append("right")
        if not emitted:
            continue

        task_type = str(record.get("task_type", "task"))
        obj_name = sanitize_name(record.get("obj_id", "obj"), max_len=56)
        stem = f"{rank:02d}_{task_type}_idx{idx:06d}_{obj_name}"
        out_path = out_dir / f"{stem}.obj"
        _write_colored_obj(str(out_path), groups)

        manifest.append(
            {
                "rank": rank,
                "prediction_index": idx,
                "task_type": task_type,
                "obj_id": record.get("obj_id"),
                "pose_id": record.get("pose_id"),
                "scale_id": record.get("scale_id"),
                "candidate_idx": record.get("candidate_idx"),
                "pred_hands": emitted,
                "guidance": record.get("guidance", ""),
                "obj_file": str(out_path),
            }
        )

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"wrote {len(manifest)} OBJ visualizations -> {out_dir}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
