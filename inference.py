"""Inference demo for DexVLG.

Loads a single object mesh, generates bimanual grasps conditioned on a
language instruction, and optionally visualizes the result.

Usage:
    python inference.py --config configs/default.yaml \
                        --checkpoint checkpoints/best.pt \
                        --mesh_path /path/to/mesh.obj \
                        --instruction "Grasp the handle of the mug" \
                        [--num_steps 50] [--visualize]
"""

import argparse

import numpy as np
import torch
import yaml

from data.dataset import sample_point_cloud_from_mesh
from models.dexvlg import DexVLG
from utils.rotation import rotation_6d_to_quaternion
from utils.visualization import create_hand_skeleton, visualize_grasp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DexVLG Inference Demo")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--n_points", type=int, default=10000)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    print("Loading model...")
    model = DexVLG(cfg["model"]).cuda()
    ckpt = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded from epoch {ckpt.get('epoch', '?')}")

    print(f"Sampling {args.n_points} points from {args.mesh_path}...")
    xyz_np, rgb_np = sample_point_cloud_from_mesh(
        args.mesh_path, args.n_points, args.scale,
    )

    xyz = torch.from_numpy(xyz_np).unsqueeze(0).cuda()
    rgb = torch.from_numpy(rgb_np).unsqueeze(0).cuda()

    print(f"Generating grasp with instruction: \"{args.instruction}\"")
    print(f"Using {args.num_steps} ODE integration steps...")

    with torch.no_grad():
        result = model.sample(xyz, rgb, [args.instruction], num_steps=args.num_steps)

    print("\n=== Predicted Grasp Poses ===")
    for hand in ["left", "right"]:
        trans = result[f"{hand}_translation"][0].cpu().numpy()
        rot6d = result[f"{hand}_rotation_6d"][0].cpu()
        joints = result[f"{hand}_joints"][0].cpu().numpy()
        quat = rotation_6d_to_quaternion(rot6d.unsqueeze(0)).squeeze(0).numpy()

        print(f"\n{hand.upper()} HAND:")
        print(f"  Translation: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]")
        print(f"  Quaternion (wxyz): [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
        print(f"  Joint angles ({len(joints)} DOF): [{', '.join(f'{j:.3f}' for j in joints[:6])}...]")

    if args.visualize or args.output:
        left_skel = create_hand_skeleton(
            result["left_translation"][0].cpu().numpy(),
            result["left_rotation_6d"][0].cpu().numpy(),
            result["left_joints"][0].cpu().numpy(),
        )
        right_skel = create_hand_skeleton(
            result["right_translation"][0].cpu().numpy(),
            result["right_rotation_6d"][0].cpu().numpy(),
            result["right_joints"][0].cpu().numpy(),
        )
        visualize_grasp(xyz_np, rgb_np, left_skel, right_skel, save_path=args.output)


if __name__ == "__main__":
    main()
