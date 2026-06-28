"""Testing and evaluation script for DexVLG.

Loads a trained checkpoint, runs inference on a test set, and computes
evaluation metrics for both left and right hand grasp predictions.

Usage:
    python test.py --config configs/default.yaml --checkpoint checkpoints/best.pt \
                   [--num_steps 50] [--output_dir results/]
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.amp import autocast
import yaml
from tqdm import tqdm

from data.dataset import DexGraspDataset, collate_fn
from models.dexvlg import DexVLG
from utils.rotation import rotation_6d_to_matrix, quaternion_to_matrix
from utils.misc import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DexVLG model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def geodesic_distance(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Compute geodesic distance between rotation matrices.

    Args:
        r1, r2: Rotation matrices, shape (..., 3, 3).

    Returns:
        Geodesic distances in radians, shape (...).
    """
    r_diff = torch.matmul(r1.transpose(-1, -2), r2)
    trace = r_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    return torch.acos(cos_angle)


def compute_metrics(
    pred_poses: dict[str, torch.Tensor],
    gt_poses: torch.Tensor,
    trans_dim: int = 3,
    rot_dim: int = 6,
) -> dict[str, float]:
    """Compute evaluation metrics for predicted grasp poses.

    Args:
        pred_poses: Predicted poses from model.sample().
        gt_poses: Ground truth poses, shape (B, 2, pose_dim).
        trans_dim: Translation dimension.
        rot_dim: Rotation dimension (6D).

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {}

    for i, hand in enumerate(["left", "right"]):
        gt = gt_poses[:, i]
        gt_trans = gt[:, :trans_dim]
        gt_rot6d = gt[:, trans_dim : trans_dim + rot_dim]
        gt_joints = gt[:, trans_dim + rot_dim :]

        pred_trans = pred_poses[f"{hand}_translation"]
        pred_rot6d = pred_poses[f"{hand}_rotation_6d"]
        pred_joints = pred_poses[f"{hand}_joints"]

        trans_err = torch.norm(pred_trans - gt_trans, dim=-1).mean().item()
        metrics[f"{hand}/trans_err_cm"] = trans_err * 100

        pred_rot_mat = rotation_6d_to_matrix(pred_rot6d)
        gt_rot_mat = rotation_6d_to_matrix(gt_rot6d)
        rot_err = geodesic_distance(pred_rot_mat, gt_rot_mat)
        metrics[f"{hand}/rot_err_deg"] = torch.rad2deg(rot_err).mean().item()

        joint_err = torch.abs(pred_joints - gt_joints).mean().item()
        metrics[f"{hand}/joint_err_rad"] = joint_err
        metrics[f"{hand}/joint_err_deg"] = np.degrees(joint_err)

        trans_success = (torch.norm(pred_trans - gt_trans, dim=-1) < 0.02).float()
        metrics[f"{hand}/trans_success_2cm"] = trans_success.mean().item()

        rot_success = (rot_err < np.radians(15)).float()
        metrics[f"{hand}/rot_success_15deg"] = rot_success.mean().item()

    return metrics


@torch.no_grad()
def evaluate(
    model: DexVLG,
    dataloader: torch.utils.data.DataLoader,
    num_steps: int,
    save_predictions: bool = False,
) -> tuple[dict[str, float], list[dict]]:
    """Run evaluation on a dataset.

    Returns:
        all_metrics: Aggregated metrics.
        predictions: List of per-sample predictions (if save_predictions).
    """
    model.eval()
    device = next(model.parameters()).device

    all_metrics: dict[str, list[float]] = {}
    predictions = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        xyz = batch["xyz"].to(device)
        rgb = batch["rgb"].to(device)
        gt_poses = batch["gt_poses"].to(device)
        texts = batch["texts"]

        with autocast("cuda", enabled=True):
            pred_poses = model.sample(xyz, rgb, texts, num_steps=num_steps)

        batch_metrics = compute_metrics(pred_poses, gt_poses)

        for k, v in batch_metrics.items():
            all_metrics.setdefault(k, []).append(v)

        if save_predictions:
            B = xyz.shape[0]
            for b in range(B):
                pred = {}
                for key, val in pred_poses.items():
                    pred[key] = val[b].cpu().numpy().tolist()
                pred["obj_id"] = batch["obj_ids"][b]
                pred["cate_id"] = batch["cate_ids"][b]
                pred["text"] = texts[b]
                predictions.append(pred)

    aggregated = {}
    for k, v in all_metrics.items():
        aggregated[k] = sum(v) / len(v)

    return aggregated, predictions


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["training"].get("seed", 42))

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model = DexVLG(cfg["model"]).cuda()
    ckpt = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    data_cfg = cfg["data"]
    test_path = data_cfg.get("test_data", data_cfg.get("val_data"))

    test_dataset = DexGraspDataset(
        data_path=test_path,
        mesh_root=data_cfg["mesh_root"],
        n_points=data_cfg.get("n_points", 10000),
        joint_dim=cfg["model"].get("joint_dim", 22),
        augment=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Evaluating on {len(test_dataset)} samples with {args.num_steps} ODE steps...")
    metrics, predictions = evaluate(
        model, test_loader, args.num_steps, args.save_predictions,
    )

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    print("=" * 60)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if predictions:
        with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved {len(predictions)} predictions to {args.output_dir}/predictions.json")


if __name__ == "__main__":
    main()
