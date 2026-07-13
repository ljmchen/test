"""Testing and evaluation script for DexVLG.

Loads a trained checkpoint, runs inference on a test set, and computes
evaluation metrics for both left and right hand grasp predictions.

Usage:
    python test.py --config configs/v3_multitask.yaml --checkpoint checkpoints/best.pt \
                   [--num_steps 50] [--output_dir results/] [--use-ema] [--val-max-batches 0]
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from torch.amp import autocast
import yaml
from tqdm import tqdm

from data.dataset import DexGraspDataset, collate_fn
from data.pose_normalizer import build_hand_normalizers
from models.dexvlg import DexVLG
from train import (
    build_group_candidates_multitask,
    format_multitask_val_table,
    validate_multitask,
)
from utils.rotation import rotation_6d_to_matrix
from utils.misc import set_seed


def denormalize_predictions(
    pred_poses: dict, gt_poses: torch.Tensor, normalizers: dict,
    trans_dim: int = 3, rot_dim: int = 6,
) -> tuple[dict, torch.Tensor]:
    """Map normalized model outputs + gt back to real units (meters/radians)."""
    for i, hand in enumerate(["left", "right"]):
        pose = torch.cat(
            [
                pred_poses[f"{hand}_translation"].float(),
                pred_poses[f"{hand}_rotation_6d"].float(),
                pred_poses[f"{hand}_joints"].float(),
            ],
            dim=-1,
        )
        pose = normalizers[hand].denormalize_pose(pose)
        pred_poses[f"{hand}_translation"] = pose[:, :trans_dim]
        pred_poses[f"{hand}_rotation_6d"] = pose[:, trans_dim : trans_dim + rot_dim]
        pred_poses[f"{hand}_joints"] = pose[:, trans_dim + rot_dim :]
        gt_poses[:, i] = normalizers[hand].denormalize_pose(gt_poses[:, i].float())
    return pred_poses, gt_poses


def _sanitize_json(obj):
    """Recursively replace NaN/Inf with None so the output is valid JSON.

    validate_multitask returns float('nan') for task/hand combos without data;
    json.dump would otherwise emit a bare NaN that jq/json.loads reject.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DexVLG model")
    parser.add_argument("--config", type=str, default="configs/v3_multitask.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use-ema", action="store_true",
                        help="Load ema_state_dict (the weights the validation "
                             "score was computed with when EMA was on) instead "
                             "of the raw model_state_dict.")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="ODE steps (default: config inference.num_steps or 50).")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val-max-batches", type=int, default=-1,
                        help="latent_ar: cap validated batches (>=0 overrides "
                             "training.val_max_batches; -1 keeps the config value).")
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
    normalizers: dict | None = None,
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

        with autocast(device.type, enabled=device.type == "cuda"):
            pred_poses = model.sample(xyz, rgb, texts, num_steps=num_steps)

        if normalizers is not None:
            pred_poses, gt_poses = denormalize_predictions(pred_poses, gt_poses, normalizers)

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


def build_test_dataset(cfg: dict, multi_task: bool) -> DexGraspDataset:
    """Construct the test-split ``DexGraspDataset`` from the config."""
    data_cfg = cfg["data"]
    test_path = data_cfg.get("test_data", data_cfg.get("val_data"))
    return DexGraspDataset(
        data_path=test_path,
        mesh_root=data_cfg["mesh_root"],
        n_points=data_cfg.get("n_points", 4096),
        joint_dim=cfg["model"].get("joint_dim", 22),
        augment=False,
        point_cloud_source=data_cfg.get("point_cloud_source", "auto"),
        point_cloud_color_mode=data_cfg.get("point_cloud_color_mode", "real"),
        point_cloud_color_fill=data_cfg.get("point_cloud_color_fill", 0.4),
        point_cloud_file_points=data_cfg.get("point_cloud_file_points", None),
        center_on_object=data_cfg.get("center_on_object", True),
        obj_pose_quaternion_order=data_cfg.get("obj_pose_quaternion_order", "wxyz"),
        normalization=data_cfg.get("normalization", None),
        multi_task=multi_task,
    )


def evaluate_latent_ar(
    model: DexVLG,
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Offline per-task-type multi-task evaluation for the latent_ar model.

    Reuses ``train.validate_multitask``: a free rollout for decision metrics and
    a forced-decision rollout for pose errors (denormalized with per-sample
    ``L_obj``), bucketed by task type. Returns the ``validate_multitask`` metrics
    dict (``score``/``per_task``/``macro``/``bimanual``/``trans``/``rot``/``joint``).
    """
    dataset = build_test_dataset(cfg, multi_task=True)
    # Deterministic shuffle: the fuse splits are stored as contiguous
    # task_type blocks, so a val_max_batches cap over sequential order would
    # silently evaluate a single scenario.
    order_gen = torch.Generator()
    order_gen.manual_seed(int(cfg["training"].get("seed", 42)))
    fixed_order = torch.randperm(len(dataset), generator=order_gen).tolist()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=fixed_order,
        num_workers=cfg["training"].get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    normalizers = build_hand_normalizers(
        cfg["data"].get("normalization", None), joint_dim=cfg["model"].get("joint_dim", 22)
    )
    group_candidates = None
    if bool(cfg["training"].get("val_group_min", False)):
        group_candidates = build_group_candidates_multitask(dataset, normalizers, device)

    max_batches = int(cfg["training"].get("val_max_batches", 0))
    n_total = len(dataset)
    n_evaluated_est = n_total if max_batches <= 0 else min(n_total, max_batches * args.batch_size)
    print(f"评测约 {n_evaluated_est}/{n_total} 条（val_max_batches={max_batches}, "
          f"batch={args.batch_size}）(latent_ar, per-task-type)")
    if n_evaluated_est < n_total:
        print(f"[WARNING] 评测被截断：仅评约 {n_evaluated_est}/{n_total} 条 "
              f"(val_max_batches={max_batches})。传 --val-max-batches 0 可评全量。")

    metrics = validate_multitask(
        model, loader, cfg, rank=0, distributed=False,
        normalizers=normalizers, group_candidates=group_candidates,
    )
    # Prefer the real count accumulated inside the validation loop over the
    # max_batches*batch_size estimate (exact for partial last batches too).
    n_evaluated = int(metrics.get("n_samples_evaluated", n_evaluated_est))
    truncated = n_evaluated < n_total
    if truncated and n_evaluated != n_evaluated_est:
        print(f"[WARNING] 实际仅评 {n_evaluated}/{n_total} 条。")
    metrics["n_evaluated"] = n_evaluated
    metrics["n_total"] = int(n_total)
    metrics["truncated"] = bool(truncated)
    metrics["val_max_batches"] = int(max_batches)
    return metrics


def evaluate_legacy(
    model: DexVLG,
    cfg: dict,
    args: argparse.Namespace,
    num_steps: int,
) -> tuple[dict, list]:
    """Legacy bimanual evaluation (fixed left+right), unchanged behavior."""
    data_cfg = cfg["data"]
    dataset = build_test_dataset(cfg, multi_task=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    normalizers = build_hand_normalizers(
        data_cfg.get("normalization", None), joint_dim=cfg["model"].get("joint_dim", 22)
    )
    print(f"Evaluating on {len(dataset)} samples with {num_steps} ODE steps...")
    return evaluate(model, loader, num_steps, args.save_predictions, normalizers)


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["training"].get("seed", 42))

    os.makedirs(args.output_dir, exist_ok=True)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    num_steps = args.num_steps
    if num_steps is None:
        num_steps = int(cfg.get("inference", {}).get("num_steps", 50))
    cfg.setdefault("inference", {})["num_steps"] = num_steps
    if args.val_max_batches >= 0:
        cfg["training"]["val_max_batches"] = args.val_max_batches

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
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(weights={'ema' if use_ema else 'raw'})")

    architecture = str(cfg["model"].get("architecture", "legacy_bimanual"))
    predictions: list = []
    if architecture == "latent_ar":
        metrics = evaluate_latent_ar(model, cfg, args, device)
        print("\n" + "=" * 60)
        print(f"Evaluation Results (latent_ar)  score={metrics['score']:.4f}")
        print("=" * 60)
        print("macro:", {k: round(v, 4) for k, v in metrics["macro"].items()})
        print("bimanual:", {k: round(v, 4) for k, v in metrics["bimanual"].items()})
        for line in format_multitask_val_table(metrics["per_task"]):
            print(line)
        print("=" * 60)
    else:
        metrics, predictions = evaluate_legacy(model, cfg, args, num_steps)
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}")
        print("=" * 60)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(_sanitize_json(metrics), f, indent=2, allow_nan=False)

    if predictions:
        with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved {len(predictions)} predictions to {args.output_dir}/predictions.json")


if __name__ == "__main__":
    main()
