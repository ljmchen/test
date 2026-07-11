"""Inference demo for DexVLG.

Loads a single object mesh, generates language-conditioned grasps, and
optionally visualizes the result.

Under ``model.architecture: latent_ar`` the model decides how many hands and
which sides to emit, so only the emitted hands are printed/visualized; the
single-object characteristic length ``L_obj`` (bbox diagonal x scale) needed by
the ``relquantile11`` denormalizer is computed here. Under ``legacy_bimanual``
(default) a fixed left+right pair is produced (unchanged behavior).

Usage:
    python inference.py --config configs/v3_multitask.yaml \
                        --checkpoint checkpoints/best.pt \
                        --mesh_path /path/to/mesh.obj \
                        --instruction "Grasp the handle of the mug" \
                        [--num_steps 10] [--visualize]
"""

import argparse
from pathlib import Path

import torch
import yaml

from data.dataset import sample_point_cloud_from_mesh
from data.lgbidex_io import load_mesh_vertices
from data.pose_normalizer import build_hand_normalizers
from models.dexvlg import DexVLG
from utils.rotation import rotation_6d_to_quaternion
from utils.visualization import create_hand_skeleton, visualize_grasp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DexVLG Inference Demo")
    parser.add_argument("--config", type=str, default="configs/v3_multitask.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use-ema", action="store_true",
                        help="Load ema_state_dict (the weights the validation "
                             "score was computed with when EMA was on) instead "
                             "of the raw model_state_dict.")
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--n_points", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def _print_hand(name: str, trans, quat, joints) -> None:
    print(f"\n{name.upper()} HAND:")
    print(f"  Translation: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]")
    print(f"  Quaternion (wxyz): [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
    print(f"  Joint angles ({len(joints)} DOF): [{', '.join(f'{j:.3f}' for j in joints[:6])}...]")


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    print("Loading model...")
    model = DexVLG(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    use_ema = args.use_ema and "ema_state_dict" in ckpt
    sd = ckpt["ema_state_dict"] if use_ema else ckpt["model_state_dict"]
    if "ema_state_dict" in ckpt and not args.use_ema:
        print("[WARNING] 该 ckpt 验证分数来自 EMA 权重，当前加载 raw "
              "model_state_dict；如需与验证分数对齐请加 --use-ema。")
    elif args.use_ema and "ema_state_dict" not in ckpt:
        print("[WARNING] --use-ema 已指定但 ckpt 无 ema_state_dict，回退加载 raw 权重。")
    model.load_state_dict(sd)
    model.eval()
    print(f"Model loaded from epoch {ckpt.get('epoch', '?')} "
          f"(weights={'ema' if use_ema else 'raw'})")

    n_points = args.n_points or cfg["data"].get("n_points", 4096)
    print(f"Sampling {n_points} points from {args.mesh_path}...")
    xyz_np, rgb_np = sample_point_cloud_from_mesh(
        args.mesh_path, n_points, args.scale,
    )
    # Single-object characteristic length L_obj = mesh bbox diagonal x scale
    # (same definition as DexGraspDataset._object_scale_length); relquantile11
    # denormalization requires it.
    vertices = load_mesh_vertices(Path(args.mesh_path))
    bbox_diag = float((vertices.max(dim=0).values - vertices.min(dim=0).values).norm())
    l_obj = bbox_diag * float(args.scale)

    xyz = torch.from_numpy(xyz_np).unsqueeze(0).to(device)
    rgb = torch.from_numpy(rgb_np).unsqueeze(0).to(device)

    normalizers = build_hand_normalizers(
        cfg["data"].get("normalization", None), joint_dim=cfg["model"].get("joint_dim", 22)
    )

    print(f"Generating grasp with instruction: \"{args.instruction}\"")
    print(f"Using {args.num_steps} ODE integration steps...")

    architecture = str(cfg["model"].get("architecture", "legacy_bimanual"))
    if architecture == "latent_ar":
        presence_threshold = float(cfg.get("inference", {}).get("presence_threshold", 0.5))
        with torch.no_grad():
            pred = model.sample_latent_ar(
                xyz, rgb, [args.instruction],
                num_steps=args.num_steps, presence_threshold=presence_threshold,
            )
        poses = pred["poses"][0].float().cpu()
        mask = pred["hand_mask"][0].cpu()
        sides = pred["hand_sides"][0].cpu()

        print("\n=== Predicted Grasp Poses ===")
        skeletons: dict[str, dict | None] = {"left": None, "right": None}
        emitted = 0
        for s in range(poses.shape[0]):
            if not bool(mask[s]):
                continue
            name = "left" if int(sides[s]) == 0 else "right"
            pose = poses[s]
            if normalizers is not None:
                pose = normalizers[name].denormalize_pose(pose, L_obj=l_obj)
            trans = pose[:3].numpy()
            rot6d = pose[3:9]
            joints = pose[9:].numpy()
            quat = rotation_6d_to_quaternion(rot6d.unsqueeze(0)).squeeze(0).numpy()
            _print_hand(name, trans, quat, joints)
            skeletons[name] = create_hand_skeleton(trans, rot6d.numpy(), joints)
            emitted += 1
        if emitted == 0:
            print("  (model emitted no hands)")

        if args.visualize or args.output:
            visualize_grasp(
                xyz_np, rgb_np, skeletons["left"], skeletons["right"], save_path=args.output
            )
        return

    with torch.no_grad():
        result = model.sample(xyz, rgb, [args.instruction], num_steps=args.num_steps)

    # Map normalized model outputs back to real units when normalization is on.
    if normalizers is not None:
        for hand in ["left", "right"]:
            pose = torch.cat(
                [
                    result[f"{hand}_translation"].float(),
                    result[f"{hand}_rotation_6d"].float(),
                    result[f"{hand}_joints"].float(),
                ],
                dim=-1,
            )
            pose = normalizers[hand].denormalize_pose(pose)
            result[f"{hand}_translation"] = pose[:, :3]
            result[f"{hand}_rotation_6d"] = pose[:, 3:9]
            result[f"{hand}_joints"] = pose[:, 9:]

    print("\n=== Predicted Grasp Poses ===")
    for hand in ["left", "right"]:
        trans = result[f"{hand}_translation"][0].cpu().numpy()
        rot6d = result[f"{hand}_rotation_6d"][0].cpu()
        joints = result[f"{hand}_joints"][0].cpu().numpy()
        quat = rotation_6d_to_quaternion(rot6d.unsqueeze(0)).squeeze(0).numpy()
        _print_hand(hand, trans, quat, joints)

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
