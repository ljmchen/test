#!/usr/bin/env python3
"""Render GT + predicted grasps (4 hands) + object into a single colored OBJ.

Picks one record from a predictions JSON (which carries both the dataset GT
`dex_grasp_{left,right}` and the model `pred_{left,right}`), converts each 28-D
dex pose to a 29-D bench qpos (axis-angle -> wxyz quat, joints pass through),
and reuses dgbench-pami's vis_obj3 FK helpers to write ONE .obj containing:
object + GT-left + GT-right + pred-left + pred-right, each a distinct color.

Run with the DGBench env (needs mujoco/trimesh + dgbench src on the path):
  MUJOCO_GL=egl /mnt/conda/jiaxuan/miniconda3/envs/DGBench/bin/python \
    tools/export_gt_pred_obj.py --index 0
"""
import argparse
import json
import os
import sys

import numpy as np

DGBENCH = "/home/jiaxuan/slai/dgbench-pami"
MESH_ROOT = "/home/jiaxuan/data/oakink_obj/processed_data"

# colors (rgb 0..1): GT cool tones, pred warm tones
OBJECT_COLOR = (0.60, 0.60, 0.62)
GT_LEFT_COLOR = (0.15, 0.65, 0.35)    # green
GT_RIGHT_COLOR = (0.15, 0.45, 0.85)   # blue
PRED_LEFT_COLOR = (0.90, 0.18, 0.20)  # red
PRED_RIGHT_COLOR = (0.95, 0.62, 0.15) # orange


def aa_to_quat_wxyz(aa):
    aa = np.asarray(aa, dtype=np.float64)
    ang = float(np.linalg.norm(aa))
    if ang < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = aa / ang
    half = 0.5 * ang
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


def dex28_to_qpos29(pose28):
    """[trans(3), axis_angle(3), joints(22)] -> [trans(3), quat wxyz(4), joints(22)].

    Same convention as prepare_bench_data.gys_pose_to_bench_qpos (fixed version):
    only aa->quat; translation + 22 joints pass through unchanged.
    """
    p = np.asarray(pose28, dtype=np.float64)
    assert p.shape == (28,), p.shape
    q = aa_to_quat_wxyz(p[3:6])
    return np.concatenate([p[:3], q, p[6:]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/jiaxuan/exp-pami/test/outputs/predictions_e0020_test.json")
    ap.add_argument("--index", type=int, default=0, help="which record in the predictions JSON")
    ap.add_argument("--out-dir", default=f"{DGBENCH}/output/dexvlg_e0020_compare")
    args = ap.parse_args()

    # dgbench src on path + cwd for relative hand xml paths
    sys.path.insert(0, os.path.join(DGBENCH, "src"))
    os.chdir(DGBENCH)
    from task.vis_obj3 import _build_hand_mesh, _build_object_mesh, _write_colored_obj
    from util.hand_util import RobotKinematics

    with open(args.input, "r", encoding="utf-8") as f:
        recs = json.load(f)
    rec = recs[args.index]

    obj_path = os.path.join(MESH_ROOT, str(rec["obj_id"]))
    obj_pose = np.asarray(rec["obj_pose"], dtype=np.float32)
    obj_scale = float(rec.get("obj_scale", 0.1))

    left_fk = RobotKinematics("assets/hand/shadow/left_hand.xml")
    right_fk = RobotKinematics("assets/hand/shadow/right_hand.xml")

    obj_verts, obj_faces = _build_object_mesh(obj_path, obj_scale, obj_pose)
    groups = [{"name": "object", "verts": obj_verts, "faces": obj_faces, "color": OBJECT_COLOR}]

    specs = [
        ("gt_left", left_fk, rec["dex_grasp_left"], GT_LEFT_COLOR),
        ("gt_right", right_fk, rec["dex_grasp_right"], GT_RIGHT_COLOR),
        ("pred_left", left_fk, rec["pred_left"], PRED_LEFT_COLOR),
        ("pred_right", right_fk, rec["pred_right"], PRED_RIGHT_COLOR),
    ]
    for name, fk, pose28, color in specs:
        qpos = dex28_to_qpos29(pose28)
        verts, faces = _build_hand_mesh(fk, qpos)
        groups.append({"name": name, "verts": verts, "faces": faces, "color": color})

    os.makedirs(args.out_dir, exist_ok=True)
    leaf = (f"{rec['obj_id']}_pose{rec.get('pose_id')}_act{rec.get('action_id')}"
            f"_grasp{rec.get('grasp_id')}_cand{rec.get('candidate_idx', 0)}")
    out = os.path.join(args.out_dir, f"{leaf}_gt_vs_pred.obj")
    _write_colored_obj(out, groups)

    print(f"sample: obj_id={rec['obj_id']} pose_id={rec.get('pose_id')} "
          f"act={rec.get('action_id')} grasp={rec.get('grasp_id')}")
    print(f"guidance: {rec.get('guidance','')}")
    print(f"wrote {out}")
    print("groups: object(gray) | gt_left(green) gt_right(blue) | pred_left(red) pred_right(orange)")


if __name__ == "__main__":
    main()
