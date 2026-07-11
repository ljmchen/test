#!/usr/bin/env python3
"""Dataset inference for DexVLG: run the model over a whole split and dump JSON.

Mirrors the dexvlm pipeline's ``project/training/inference.py`` so the output
feeds the same downstream pre/squeeze + bench conversion. For each record the
model samples a grasp, which is denormalized, converted to the 28-d raw format
``[translation(3), axis_angle(3), joints(22)]`` per hand, and (when the dataset
was object-centered) restored to the world frame.

Two architectures are supported (``model.architecture``):
  - ``legacy_bimanual`` (default): a fixed left+right pair is always emitted;
  - ``latent_ar``: the model decides how many hands and which sides to emit, so
    only the emitted hands get a ``pred_{side}`` key (missing hands are absent).

Each output record keeps the original split-record fields and adds:
  - ``pred_left`` and/or ``pred_right``: 28-d world-frame grasp poses (axis-angle)
  - ``pred_pose_frame`` and ``_dexvlg_inference`` metadata

``--gt-baseline`` bypasses the model and copies each record's own
``dex_grasp_{left,right}`` (already world-frame 28-d) into ``pred_{side}`` for the
hands that exist, to sanity-check the downstream bench conversion chain.

Usage:
  python infer_dataset.py \\
    --config configs/v3_multitask.yaml \\
    --checkpoint outputs_v3/checkpoints/best_eXXXX_sX.XXXX.pt \\
    --split test \\
    --output outputs_v3/predictions_test.json \\
    --num-steps 10 --batch-size 256 --task-type-filter lgbidex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from data.dataset import DexGraspDataset, hands_present
from data.pose_normalizer import build_hand_normalizers
from models.dexvlg import DexVLG
from utils.rotation import rotation_6d_to_matrix

HAND_SIDES = ("left", "right")


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


def _record_value(record: dict, key: str) -> str:
    """Fetch a dedup-key value, tolerating the guidance/guidence spelling."""
    if key in ("guidance", "guidence"):
        return str(record.get("guidance", record.get("guidence", "")))
    return str(record.get(key, ""))


def _record_key(record: dict, keys: list[str]) -> tuple:
    return tuple(_record_value(record, k) for k in keys)


def _inference_meta(
    *,
    prediction_fields: list[str],
    hand_sides: list[str],
    task_type: str,
    l_obj: float,
    center_on_object: bool,
    normalized: bool,
    normalization_mode: str | None,
    architecture: str,
    num_steps: int,
    cand_idx: int,
    checkpoint: str | None,
    gt_baseline: bool,
) -> dict:
    """Assemble the ``_dexvlg_inference`` metadata block for one output record."""
    return {
        "prediction_fields": list(prediction_fields),
        "pred_num_hands": len(prediction_fields),
        "pred_hand_sides": list(hand_sides),
        "pred_pose_format": "translation_axis_angle_joints_28",
        "pred_pose_frame": "world",
        "task_type": task_type,
        "L_obj": float(l_obj),
        "center_on_object_input": bool(center_on_object),
        "normalized": bool(normalized),
        "normalization_mode": normalization_mode,
        "architecture": architecture,
        "num_steps": int(num_steps),
        "candidate_idx": int(cand_idx),
        "gt_baseline": bool(gt_baseline),
        "checkpoint": checkpoint,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DexVLG dataset inference -> predictions JSON")
    parser.add_argument("--config", type=str, default="configs/v3_multitask.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint (required unless --gt-baseline).")
    parser.add_argument("--use-ema", action="store_true",
                        help="Load ema_state_dict (the weights the validation "
                             "score was computed with when EMA was on) instead "
                             "of the raw model_state_dict.")
    parser.add_argument("--lang-padding", type=str, default=None,
                        choices=["longest", "max_length"],
                        help="Override model.lang_padding in the config: 'max_length' "
                             "fixes the tokenizer pad length for batch-invariant "
                             "inference; 'longest' pads per batch (legacy). Default: "
                             "keep the config value.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Use data.{split}_data from the config.")
    parser.add_argument("--split-path", type=str, default=None,
                        help="Override the split JSON path directly.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Euler ODE steps (default: config inference.num_steps or 10).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--gt-baseline", action="store_true",
        help="Bypass the model: copy each record's own dex_grasp_{left,right} "
             "(world-frame 28-d) into pred_{side} for the hands that exist, to "
             "sanity-check the downstream bench conversion chain.",
    )
    parser.add_argument(
        "--task-type-filter", type=str, default="",
        help="Comma-separated task_types to keep (e.g. 'lgbidex' or 'left,right'). "
             "Empty = all task types.",
    )
    parser.add_argument(
        "--presence-threshold", type=float, default=None,
        help="latent_ar sigmoid threshold for emitting hand slot 1 "
             "(default: config inference.presence_threshold or 0.5).",
    )
    parser.add_argument(
        "--dedup-by", type=str, default="",
        help="Comma-separated record keys to dedup inputs by (e.g. obj_id,pose_id,guidance). "
             "Empty = no dedup (one inference per record). 'guidance' also matches 'guidence'.",
    )
    parser.add_argument(
        "--samples-per-combo", type=int, default=1,
        help="Number of grasps to sample per (deduped) combination. Each becomes a "
             "separate output record with its own candidate_idx (0..N-1).",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not args.gt_baseline and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --gt-baseline is set.")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.lang_padding is not None:
        # CLI -> cfg passthrough; the model reads model.lang_padding when
        # building the tokenizer call (batch-invariance fix, P0-1).
        cfg["model"]["lang_padding"] = args.lang_padding

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    num_steps = args.num_steps
    if num_steps is None:
        num_steps = int(cfg.get("inference", {}).get("num_steps", 10))
    presence_threshold = args.presence_threshold
    if presence_threshold is None:
        presence_threshold = float(cfg.get("inference", {}).get("presence_threshold", 0.5))

    data_cfg = cfg["data"]
    split_path = args.split_path or data_cfg.get(f"{args.split}_data") or data_cfg.get("val_data")
    if not split_path or not Path(split_path).exists():
        raise FileNotFoundError(f"Split JSON not found: {split_path}")

    joint_dim = cfg["model"].get("joint_dim", 22)
    normalization = data_cfg.get("normalization", None)
    normalization_mode = None
    if isinstance(normalization, dict) and normalization.get("enabled", False):
        normalization_mode = str(normalization.get("mode", "minmax11"))
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

    architecture = str(cfg["model"].get("architecture", "legacy_bimanual"))
    is_latent = architecture == "latent_ar"

    model = None
    checkpoint_str: str | None = None
    if not args.gt_baseline:
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
        print(f"[info] loaded checkpoint weights={'ema' if use_ema else 'raw'}")
        checkpoint_str = str(Path(args.checkpoint).resolve())

    n = len(dataset)
    filter_set = {t.strip() for t in args.task_type_filter.split(",") if t.strip()} or None

    # Optional dedup: keep the first record per (dedup-by) combination.
    dedup_keys = [k.strip() for k in args.dedup_by.split(",") if k.strip()]
    if dedup_keys:
        seen: dict[tuple, int] = {}
        for i in range(n):
            key = _record_key(dataset.data[i], dedup_keys)
            if key not in seen:
                seen[key] = i
        unique_indices = list(seen.values())
    else:
        unique_indices = list(range(n))

    if filter_set is not None:
        unique_indices = [
            i for i in unique_indices
            if str(dataset.data[i].get("task_type", "")) in filter_set
        ]

    samples_per_combo = max(1, int(args.samples_per_combo))
    # Work list: (dataset_idx, candidate_idx). Flow matching samples from random
    # noise, so the same input replicated N times yields N distinct grasps.
    work = [(idx, c) for idx in unique_indices for c in range(samples_per_combo)]

    print(f"[info] split={args.split} path={split_path} records={n}")
    print(f"[info] task_type_filter={sorted(filter_set) if filter_set else 'all'} "
          f"dedup_by={dedup_keys or 'none'} unique_combos={len(unique_indices)} "
          f"samples_per_combo={samples_per_combo} -> {len(work)} inferences")
    print(f"[info] device={device_name} batch_size={args.batch_size} num_steps={num_steps} "
          f"architecture={architecture} gt_baseline={args.gt_baseline} "
          f"normalized={normalizers is not None} center_on_object={center_on_object}")

    total = len(work)
    results: list[dict] = []
    for start in range(0, total, args.batch_size):
        chunk = work[start : start + args.batch_size]
        cand_ids = [c for _, c in chunk]

        if args.gt_baseline:
            for i, (idx, _) in enumerate(chunk):
                record = dataset.data[idx]
                sides = hands_present(record)
                if not sides:
                    raise ValueError(
                        f"Record idx={idx} obj_id={record.get('obj_id')!r} has no "
                        "dex_grasp_left/right; cannot build a GT baseline."
                    )
                l_obj = float(dataset._object_scale_length(record))
                out = dict(record)
                for side in sides:
                    out[f"pred_{side}"] = list(record[f"dex_grasp_{side}"])
                out["pred_pose_frame"] = "world"
                out["candidate_idx"] = int(cand_ids[i])
                out["_dexvlg_inference"] = _inference_meta(
                    prediction_fields=[f"pred_{s}" for s in sides],
                    hand_sides=sides,
                    task_type=str(record.get("task_type", "")),
                    l_obj=l_obj,
                    center_on_object=center_on_object,
                    normalized=False,
                    normalization_mode=normalization_mode,
                    architecture=architecture,
                    num_steps=0,
                    cand_idx=int(cand_ids[i]),
                    checkpoint=None,
                    gt_baseline=True,
                )
                results.append(out)
            print(f"  [{min(start + args.batch_size, total)}/{total}] done", end="\r", flush=True)
            continue

        items = [dataset.get_inference_item(idx) for idx, _ in chunk]
        xyz = torch.stack([it["xyz"] for it in items]).to(device)
        rgb = torch.stack([it["rgb"] for it in items]).to(device)
        texts = [it["text"] for it in items]

        if is_latent:
            pred = model.sample_latent_ar(
                xyz, rgb, texts,
                num_steps=num_steps, presence_threshold=presence_threshold,
            )
            poses = pred["poses"].float()
            hmask = pred["hand_mask"].bool()
            hsides = pred["hand_sides"].long()
            l_obj_t = torch.tensor(
                [it["L_obj"] for it in items], dtype=torch.float32, device=poses.device
            )

            denorm = poses.clone()
            if normalizers is not None:
                for side_id, name in ((0, "left"), (1, "right")):
                    for s in range(poses.shape[1]):
                        sel = hmask[:, s] & (hsides[:, s] == side_id)
                        if bool(sel.any()):
                            denorm[sel, s] = normalizers[name].denormalize_pose(
                                poses[sel, s], L_obj=l_obj_t[sel]
                            )
            raw28 = hand_pose_to_raw28(
                denorm[..., :3], denorm[..., 3:9], denorm[..., 9:]
            ).cpu()

            for i, it in enumerate(items):
                out = dict(it["record"])
                emitted: list[str] = []
                if center_on_object:
                    offset = (it["object_translation"] + it["object_center_offset"]).float()
                for s in range(poses.shape[1]):
                    if not bool(hmask[i, s]):
                        continue
                    name = "left" if int(hsides[i, s]) == 0 else "right"
                    pose = raw28[i, s].clone()
                    if center_on_object:
                        pose[:3] = pose[:3] + offset
                    out[f"pred_{name}"] = pose.tolist()
                    emitted.append(name)
                out["pred_pose_frame"] = "world"
                out["candidate_idx"] = int(cand_ids[i])
                out["_dexvlg_inference"] = _inference_meta(
                    prediction_fields=[f"pred_{s}" for s in emitted],
                    hand_sides=emitted,
                    task_type=str(it["task_type"]),
                    l_obj=float(it["L_obj"]),
                    center_on_object=center_on_object,
                    normalized=normalizers is not None,
                    normalization_mode=normalization_mode,
                    architecture=architecture,
                    num_steps=int(num_steps),
                    cand_idx=int(cand_ids[i]),
                    checkpoint=checkpoint_str,
                    gt_baseline=False,
                )
                results.append(out)
            print(f"  [{min(start + args.batch_size, total)}/{total}] done", end="\r", flush=True)
            continue

        # legacy_bimanual: a fixed left+right pair is always emitted.
        pred = model.sample(xyz, rgb, texts, num_steps=num_steps)
        for hand in HAND_SIDES:
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
            # candidate index so downstream (prepare_bench_data.py) groups the N
            # samples of one combination as N variants of the same sample.
            out["candidate_idx"] = int(cand_ids[i])
            out["_dexvlg_inference"] = _inference_meta(
                prediction_fields=["pred_left", "pred_right"],
                hand_sides=list(HAND_SIDES),
                task_type=str(it["task_type"]),
                l_obj=float(it["L_obj"]),
                center_on_object=center_on_object,
                normalized=normalizers is not None,
                normalization_mode=normalization_mode,
                architecture=architecture,
                num_steps=int(num_steps),
                cand_idx=int(cand_ids[i]),
                checkpoint=checkpoint_str,
                gt_baseline=False,
            )
            results.append(out)

        print(f"  [{min(start + args.batch_size, total)}/{total}] done", end="\r", flush=True)
    print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[info] saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
