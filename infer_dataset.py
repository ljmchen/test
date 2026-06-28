#!/usr/bin/env python3
"""Dataset inference for DexVLG: run the model over a whole split and dump JSON.

Mirrors the dexvlm pipeline's ``project/training/inference.py`` so the output
feeds the same downstream pre/squeeze + bench conversion. For each record the
model samples a bimanual grasp, which is denormalized, converted to the 28-d
raw format ``[translation(3), axis_angle(3), joints(22)]`` per hand, and (when
the dataset was object-centered) restored to the world frame.

Each output record keeps the original split-record fields and adds:
  - ``pred_left`` / ``pred_right``: 28-d world-frame grasp poses (axis-angle)
  - ``pred_pose_frame`` and ``_dexvlg_inference`` metadata

Usage:
  python infer_dataset.py \\
    --config configs/default.yaml \\
    --checkpoint outputs/checkpoints/best_eXXXX_sX.XXXX.pt \\
    --split val \\
    --output outputs/predictions_test.json \\
    --num-steps 10 --batch-size 256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from data.dataset import DexGraspDataset
from data.pose_normalizer import build_hand_normalizers
from models.dexvlg import DexVLG
from utils.rotation import rotation_6d_to_matrix


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices ``(..., 3, 3)`` to axis-angle ``(..., 3)``."""
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    # axis from the skew-symmetric part; stable for small angles via sin scaling.
    rx = matrix[..., 2, 1] - matrix[..., 1, 2]
    ry = matrix[..., 0, 2] - matrix[..., 2, 0]
    rz = matrix[..., 1, 0] - matrix[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)
    sin_angle = torch.sin(angle).unsqueeze(-1)
    axis = axis / (2.0 * sin_angle).clamp_min(1e-8)
    axis = torch.nn.functional.normalize(axis, dim=-1)
    return axis * angle.unsqueeze(-1)


def hand_pose_to_raw28(
    translation: torch.Tensor, rotation_6d: torch.Tensor, joints: torch.Tensor
) -> torch.Tensor:
    """Assemble a 28-d ``[trans(3), axis_angle(3), joints(22)]`` pose."""
    axis_angle = matrix_to_axis_angle(rotation_6d_to_matrix(rotation_6d))
    return torch.cat([translation, axis_angle, joints], dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DexVLG dataset inference -> predictions JSON")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Use data.{split}_data from the config.")
    parser.add_argument("--split-path", type=str, default=None,
                        help="Override the split JSON path directly.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Euler ODE steps (default: config inference.num_steps or 10).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    num_steps = args.num_steps
    if num_steps is None:
        num_steps = int(cfg.get("inference", {}).get("num_steps", 10))

    data_cfg = cfg["data"]
    split_path = args.split_path or data_cfg.get(f"{args.split}_data") or data_cfg.get("val_data")
    if not split_path or not Path(split_path).exists():
        raise FileNotFoundError(f"Split JSON not found: {split_path}")

    joint_dim = cfg["model"].get("joint_dim", 22)
    normalization = data_cfg.get("normalization", None)
    dataset = DexGraspDataset(
        data_path=split_path,
        mesh_root=data_cfg["mesh_root"],
        n_points=data_cfg.get("n_points", 4096),
        joint_dim=joint_dim,
        augment=False,
        point_cloud_source=data_cfg.get("point_cloud_source", "auto"),
        point_cloud_color_mode=data_cfg.get("point_cloud_color_mode", "real"),
        point_cloud_color_fill=data_cfg.get("point_cloud_color_fill", 0.4),
        point_cloud_file_points=data_cfg.get("point_cloud_file_points", None),
        center_on_object=data_cfg.get("center_on_object", True),
        obj_pose_quaternion_order=data_cfg.get("obj_pose_quaternion_order", "wxyz"),
        normalization=normalization,
    )
    center_on_object = bool(data_cfg.get("center_on_object", True))
    normalizers = build_hand_normalizers(normalization, joint_dim=joint_dim)

    model = DexVLG(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n = len(dataset)
    print(f"[info] split={args.split} path={split_path} records={n}")
    print(f"[info] device={device_name} batch_size={args.batch_size} num_steps={num_steps} "
          f"normalized={normalizers is not None} center_on_object={center_on_object}")

    results: list[dict] = []
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        items = [dataset.get_inference_item(i) for i in range(start, end)]
        xyz = torch.stack([it["xyz"] for it in items]).to(device)
        rgb = torch.stack([it["rgb"] for it in items]).to(device)
        texts = [it["text"] for it in items]

        pred = model.sample(xyz, rgb, texts, num_steps=num_steps)

        for hand in ("left", "right"):
            pose = torch.cat(
                [
                    pred[f"{hand}_translation"].float(),
                    pred[f"{hand}_rotation_6d"].float(),
                    pred[f"{hand}_joints"].float(),
                ],
                dim=-1,
            )
            if normalizers is not None:
                pose = normalizers[hand].denormalize_pose(pose)
            pred[f"{hand}_pose31"] = pose

        left28 = hand_pose_to_raw28(
            pred["left_pose31"][:, :3], pred["left_pose31"][:, 3:9], pred["left_pose31"][:, 9:]
        ).cpu()
        right28 = hand_pose_to_raw28(
            pred["right_pose31"][:, :3], pred["right_pose31"][:, 3:9], pred["right_pose31"][:, 9:]
        ).cpu()

        for i, it in enumerate(items):
            left = left28[i].clone()
            right = right28[i].clone()
            if center_on_object:
                offset = (it["object_translation"] + it["object_center_offset"]).to(left.dtype)
                left[:3] = left[:3] + offset
                right[:3] = right[:3] + offset

            out = dict(it["record"])
            out["pred_left"] = left.tolist()
            out["pred_right"] = right.tolist()
            out["pred_pose_frame"] = "world"
            out["_dexvlg_inference"] = {
                "prediction_fields": ["pred_left", "pred_right"],
                "pred_pose_format": "translation_axis_angle_joints_28",
                "pred_pose_frame": "world",
                "center_on_object_input": center_on_object,
                "normalized": normalizers is not None,
                "num_steps": int(num_steps),
                "checkpoint": str(Path(args.checkpoint).resolve()),
            }
            results.append(out)

        print(f"  [{end}/{n}] done", end="\r", flush=True)
    print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[info] saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
