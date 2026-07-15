"""Training script for DexVLG bimanual dexterous grasp generation.

Trains the full model end-to-end: PC encoder, projector, fusion transformer,
and flow-matching head, using conditional flow matching loss.

Usage:
    python train.py --config configs/v3_multitask.yaml [--resume PATH]
    (argparse 默认 configs/v3_smoke.yaml —— smoke 规模，裸跑安全)
"""

import argparse
import copy
import logging
import math
import os
import re
import shutil
import subprocess

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

from data.dataset import (
    create_dataloader,
    DexGraspDataset,
    collate_fn,
    group_id_of,
    hands_present,
)
from data.pose_normalizer import build_hand_normalizers
from data.samplers import GroupedEpochSampler
from models.dexvlg import DexVLG
from utils.metrics import (
    BIMANUAL,
    MultiTaskValAccumulator,
    TASK_TYPES,
    align_hands_by_side,
    compose_score,
    pose_errors_physical,
    relative_pose_error,
    structure_metrics,
    task_type_index,
)
from utils.rotation import rotation_6d_to_matrix
from utils.misc import set_seed, count_parameters, AverageMeter

_TRAIN_LOSS_COMPONENT_KEYS = (
    "loss_flow",
    "loss_rot_geo",
    "loss_presence",
    "loss_side",
    "loss_probe_task",
    "loss_probe_lobj",
    "loss_contact",
    "loss_anchor",
    "loss_approach",
    "loss_rel_rot",
)


def geodesic_angle(pred_rot6d: torch.Tensor, gt_rot6d: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (radians) between two 6D rotations, shape (...,)."""
    r_pred = rotation_6d_to_matrix(pred_rot6d)
    r_gt = rotation_6d_to_matrix(gt_rot6d)
    r_rel = torch.matmul(r_pred.transpose(-1, -2), r_gt)
    trace = r_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.acos(cos_angle)


class _EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        src = model.module if hasattr(model, "module") else model
        for s_param, m_param in zip(self.shadow.parameters(), src.parameters()):
            s_param.lerp_(m_param.data, 1.0 - self.decay)
        for s_buf, m_buf in zip(self.shadow.buffers(), src.buffers()):
            s_buf.copy_(m_buf)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow.load_state_dict(state_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DexVLG model")
    parser.add_argument("--config", type=str, default="configs/v3_smoke.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--init-weights", type=str, default=None,
        help="Warm-start: load model weights from a checkpoint's "
             "model_state_dict with strict=False (new params such as the CFG "
             "null_cond stay at init). Optimizer/scheduler/epoch/global_step "
             "are NOT restored. Mutually exclusive with --resume.",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_amp_settings(train_cfg: dict) -> tuple[bool, torch.dtype, bool]:
    """Resolve AMP from config -> (autocast_enabled, dtype, use_grad_scaler).

    ``precision: fp16 | bf16 | fp32`` (preferred); falls back to the legacy
    ``fp16`` bool. **bf16 is recommended on H100 / with ModernBERT**: fp16 can
    overflow these activations to inf, which poisons BatchNorm running stats and
    yields permanent NaNs (GradScaler only gates the optimizer step, not the
    forward-updated buffers). The grad scaler is only used for fp16.
    """
    precision = train_cfg.get("precision")
    if precision is None:
        precision = "fp16" if train_cfg.get("fp16", True) else "fp32"
    precision = str(precision).lower()
    if precision in ("bf16", "bfloat16"):
        return True, torch.bfloat16, False
    if precision in ("fp16", "float16", "half"):
        return True, torch.float16, True
    return False, torch.float32, False


def setup_logging(log_dir: str, rank: int) -> logging.Logger:
    """Create a logger that mirrors console output to ``{log_dir}/train.log``.

    Only rank 0 writes the file; other ranks log to the console only.
    """
    logger = logging.getLogger("dexvlg")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_distributed() -> tuple[int, int]:
    """Initialize distributed training when launched via torchrun.

    Only enters distributed mode when the full torchrun env (RANK, WORLD_SIZE
    and LOCAL_RANK) is present and WORLD_SIZE > 1. A stray RANK left in the
    shell otherwise falls back to single-process so it can't trigger a broken
    distributed init (KeyError: 'LOCAL_RANK').
    """
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if all(k in os.environ for k in required) and int(os.environ["WORLD_SIZE"]) > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank
    return 0, 0


def train_one_epoch(
    vc: dict,
    dataloader: torch.utils.data.DataLoader,
    epoch: int,
    global_step: int,
) -> tuple[float, int]:
    """Train for one epoch; return average loss and updated global step.

    Validation + best-K checkpointing are driven per-epoch by the caller (see
    ``main``), not inside this loop.
    """
    model = vc["model"]
    cfg = vc["cfg"]
    rank = vc["rank"]
    writer = vc["writer"]
    logger = vc["logger"]
    optimizer = vc["optimizer"]
    scheduler = vc["scheduler"]
    scaler = vc["scaler"]

    model.train()
    ema = vc.get("ema")
    loss_meter = AverageMeter()
    amp_enabled, amp_dtype, _ = get_amp_settings(cfg["training"])
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    log_interval = cfg["training"].get("log_interval", 50)
    multi_task = bool(cfg["data"].get("multi_task", False))
    architecture = str(cfg["model"].get("architecture", "legacy_bimanual"))
    nonfinite_skips = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=rank != 0)
    for batch in pbar:
        xyz = batch["xyz"].cuda(non_blocking=True)
        rgb = batch["rgb"].cuda(non_blocking=True)
        gt_poses = batch["gt_poses"].cuda(non_blocking=True)
        texts = batch["texts"]

        extra_inputs: dict = {}
        if multi_task:
            hand_mask = batch["hand_mask"].cuda(non_blocking=True)
            hand_side_ids = batch["hand_side_ids"].cuda(non_blocking=True)
            l_obj = batch["L_obj"].cuda(non_blocking=True)
            if architecture == "latent_ar":
                extra_inputs = {
                    "hand_mask": hand_mask,
                    "hand_side_ids": hand_side_ids,
                    "task_type_ids": task_type_index(
                        batch["task_types"], device=xyz.device
                    ),
                    "log_l_obj": l_obj.log(),
                }
                # GRACE self-supervised targets (present only when
                # data.grace_targets.enabled); required by the GRACE losses.
                if "grasp_center" in batch:
                    extra_inputs["grasp_center"] = batch["grasp_center"].cuda(
                        non_blocking=True
                    )
                    extra_inputs["approach_dir"] = batch["approach_dir"].cuda(
                        non_blocking=True
                    )

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(xyz, rgb, texts, gt_poses=gt_poses, **extra_inputs)
            loss = outputs["loss"]

        # Skip the step on a non-finite loss instead of stepping into NaN weights.
        # The skip decision must be synchronized across ranks: if one rank skips
        # backward while others enter the gradient all-reduce, the DDP buckets
        # pair across iterations (silent gradient corruption, then hang/crash).
        nonfinite_flag = (~torch.isfinite(loss.detach())).to(
            device=loss.device, dtype=torch.float32
        )
        if dist.is_initialized():
            dist.all_reduce(nonfinite_flag, op=dist.ReduceOp.MAX)
        if bool(nonfinite_flag.item()):
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            nonfinite_skips += 1
            if rank == 0 and nonfinite_skips <= 5:
                logger.warning(
                    f"[epoch {epoch} step {global_step}] non-finite loss; step skipped "
                    f"(total skipped this epoch: {nonfinite_skips})"
                )
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        scheduler.step()

        loss_val = loss.item()
        loss_meter.update(loss_val, xyz.shape[0])
        global_step += 1

        if rank == 0:
            pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{loss_meter.avg:.4f}")
            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                components = {
                    key: outputs[key].item()
                    for key in _TRAIN_LOSS_COMPONENT_KEYS
                    if key in outputs
                }
                if writer:
                    writer.add_scalar("train/loss", loss_val, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    for key, value in components.items():
                        writer.add_scalar(f"train/{key}", value, global_step)
                parts = "".join(
                    f" {key.removeprefix('loss_')}={value:.4f}"
                    for key, value in components.items()
                )
                logger.info(
                    f"[epoch {epoch} step {global_step}] "
                    f"train_loss={loss_val:.4f} avg={loss_meter.avg:.4f} lr={lr:.2e}"
                    f"{parts}"
                )

    return loss_meter.avg, global_step


def run_validation(vc: dict, epoch: int, global_step: int) -> None:
    """Run sampling-based validation, log it, and update best-K checkpoints.

    Collective: every rank must call this together. Only rank 0 logs/saves.
    Dispatches to the multi-task latent_ar validator when ``data.multi_task``
    is set; the legacy bimanual path is unchanged otherwise.
    """
    model = vc["model"]
    cfg = vc["cfg"]
    ema = vc.get("ema")
    val_model = ema.shadow if ema is not None else model
    multi_task = bool(cfg["data"].get("multi_task", False))
    # Build (once, cached) the group->GT-candidates map for group-internal matching.
    group_candidates = None
    if bool(cfg["training"].get("val_group_min", False)) and vc["val_loader"] is not None:
        if vc.get("group_candidates") is None:
            if multi_task:
                vc["group_candidates"] = build_group_candidates_multitask(
                    vc["val_loader"].dataset, vc.get("normalizers"), torch.device("cuda")
                )
            else:
                vc["group_candidates"] = build_group_candidates(
                    vc["val_loader"].dataset, vc.get("normalizers"), torch.device("cuda")
                )
            if vc["rank"] == 0:
                vc["logger"].info(
                    f"Built val group candidates: {len(vc['group_candidates'])} groups "
                    "(group-internal best-match validation ON)"
                )
        group_candidates = vc["group_candidates"]
    if multi_task:
        val_metrics = validate_multitask(
            val_model, vc["val_loader"], cfg, vc["rank"], vc["distributed"],
            normalizers=vc.get("normalizers"), group_candidates=group_candidates,
        )
    else:
        val_metrics = validate(
            val_model, vc["val_loader"], cfg, vc["rank"], vc["distributed"],
            normalizers=vc.get("normalizers"), group_candidates=group_candidates,
        )
    model.train()  # restore training mode on every rank before continuing

    if vc["rank"] != 0:
        return

    writer, logger = vc["writer"], vc["logger"]
    if writer:
        writer.add_scalar("val/trans_err_m", val_metrics["trans"], global_step)
        writer.add_scalar("val/rot_err_rad", val_metrics["rot"], global_step)
        writer.add_scalar("val/joint_err_rad", val_metrics["joint"], global_step)
        writer.add_scalar("val/score", val_metrics["score"], global_step)
        if "per_task" in val_metrics:
            for tt, m in val_metrics["per_task"].items():
                for key in (
                    "trans", "rot", "joint",
                    "count_acc", "side_acc", "structure_acc",
                ):
                    if math.isfinite(m[key]):
                        writer.add_scalar(f"val/{tt}/{key}", m[key], global_step)
            for key, value in val_metrics["bimanual"].items():
                if math.isfinite(value):
                    writer.add_scalar(f"val/bimanual/{key}", value, global_step)

    if "per_task" in val_metrics:
        for line in format_multitask_val_table(val_metrics["per_task"]):
            logger.info(line)

    score = val_metrics["score"]
    vc["best_registry"], kept = update_best_checkpoints(
        vc["best_registry"], score, vc["keep_best_k"], vc["ckpt_dir"], epoch,
        lambda path: save_checkpoint(
            model, vc["optimizer"], vc["scheduler"], vc["scaler"],
            epoch, global_step, score, path, val_metrics, ema=vc.get("ema"),
        ),
    )
    msg = (
        f"[epoch {epoch} step {global_step}] VAL "
        f"trans={val_metrics['trans']:.4f} rot={val_metrics['rot']:.4f} "
        f"joint={val_metrics['joint']:.4f} score={score:.4f}"
    )
    if kept:
        msg += f"  -> kept top-{vc['keep_best_k']} (best={vc['best_registry'][0][0]:.4f})"
    logger.info(msg)


def pose_component_errors(
    pred: torch.Tensor,
    target: torch.Tensor,
    trans_dim: int,
    rot_dim: int,
    mse_trans: bool,
    mse_rot: bool,
    mse_joint: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-element translation/rotation/joint errors between ``pred`` and ``target``.

    Broadcasts: shapes ``(..., 31)`` -> three ``(...,)`` tensors. Used both for
    one-to-one (B vs B) and group matching (one pred vs K candidates).
    """
    p_t, p_r, p_j = pred[..., :trans_dim], pred[..., trans_dim:trans_dim + rot_dim], pred[..., trans_dim + rot_dim:]
    g_t, g_r, g_j = target[..., :trans_dim], target[..., trans_dim:trans_dim + rot_dim], target[..., trans_dim + rot_dim:]
    trans = ((p_t - g_t) ** 2).mean(-1) if mse_trans else torch.norm(p_t - g_t, dim=-1)
    rot = ((p_r - g_r) ** 2).mean(-1) if mse_rot else geodesic_angle(p_r, g_r)
    joint = ((p_j - g_j) ** 2).mean(-1) if mse_joint else torch.abs(p_j - g_j).mean(-1)
    return trans, rot, joint


@torch.no_grad()
def build_group_candidates(dataset, normalizers: dict | None, device) -> dict:
    """Collect all GT bimanual grasps of each (obj_id,pose_id,guidance) group.

    Returns ``{group_id: {"left": (K,31), "right": (K,31)}}`` in real
    (denormalized) units on ``device``, for group-internal best-match validation.
    """
    from collections import defaultdict

    left: dict[str, list] = defaultdict(list)
    right: dict[str, list] = defaultdict(list)
    for record in dataset.data:
        gid = group_id_of(record)
        left[gid].append(dataset._build_hand_pose(record, "left"))
        right[gid].append(dataset._build_hand_pose(record, "right"))
    out: dict[str, dict] = {}
    for gid in left:
        cand_left = torch.stack(left[gid]).float()
        cand_right = torch.stack(right[gid]).float()
        if normalizers is not None:
            cand_left = normalizers["left"].denormalize_pose(cand_left)
            cand_right = normalizers["right"].denormalize_pose(cand_right)
        out[gid] = {"left": cand_left.to(device), "right": cand_right.to(device)}
    return out


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cfg: dict,
    rank: int,
    distributed: bool = False,
    normalizers: dict | None = None,
    group_candidates: dict | None = None,
) -> dict[str, float]:
    """Validate by running the full N-step ODE sampling, then scoring poses.

    Unlike training (a single-timestep flow-matching loss), this generates the
    grasp poses via ``model.sample`` (``inference.num_steps`` Euler steps) and
    reports separate errors for translation, rotation and hand joints, averaged
    over both hands and (when distributed) reduced across ranks:

        - translation: mean Euclidean distance (meters)
        - rotation:    mean geodesic angle (radians)
        - joint:       mean absolute joint-angle error (radians)

    Each component can instead be reported as MSE via
    ``training.val_loss_mse.{translation,rotation,joint}``. Metrics are computed
    in real (denormalized) units when normalization is enabled.

    Returns a dict with ``trans``, ``rot``, ``joint`` and a combined ``score``
    (the unweighted sum, lower is better) used for checkpoint selection.
    """
    model.eval()
    net = model.module if hasattr(model, "module") else model
    amp_enabled, amp_dtype, _ = get_amp_settings(cfg["training"])
    num_steps = cfg.get("inference", {}).get("num_steps", 10)
    # optional cap on validation batches (0 = whole val set) for cheap, frequent eval
    max_batches = int(cfg["training"].get("val_max_batches", 0))
    trans_dim, rot_dim = net.trans_dim, net.rot_dim
    # per-component metric switch: True -> F.mse_loss-style MSE, False -> physical
    # metric (translation L2 distance / rotation geodesic angle / joint L1).
    mse_cfg = dict(cfg["training"].get("val_loss_mse", {}) or {})
    mse_trans = bool(mse_cfg.get("translation", False))
    mse_rot = bool(mse_cfg.get("rotation", False))
    mse_joint = bool(mse_cfg.get("joint", False))

    # accumulate (sum, count) so a distributed all-reduce stays exact
    sums = torch.zeros(3, device="cuda")
    count = torch.zeros(1, device="cuda")

    for batch_idx, batch in enumerate(
        tqdm(dataloader, desc="Validation", disable=rank != 0)
    ):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        xyz = batch["xyz"].cuda(non_blocking=True)
        rgb = batch["rgb"].cuda(non_blocking=True)
        gt_poses = batch["gt_poses"].cuda(non_blocking=True)
        texts = batch["texts"]
        bsz = xyz.shape[0]

        with autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            pred = net.sample(xyz, rgb, texts, num_steps=num_steps)

        # denormalized predictions per hand -> real units (B, 31)
        pred_real = {}
        for hand in ("left", "right"):
            pp = torch.cat(
                [
                    pred[f"{hand}_translation"].float(),
                    pred[f"{hand}_rotation_6d"].float(),
                    pred[f"{hand}_joints"].float(),
                ],
                dim=-1,
            )
            if normalizers is not None:
                pp = normalizers[hand].denormalize_pose(pp)
            pred_real[hand] = pp

        if group_candidates is not None:
            group_ids = batch["group_ids"]
            dev = pred_real["left"].device
            pose_dim_v = pred_real["left"].shape[-1]
            max_k = max(
                (group_candidates[g]["left"].shape[0] for g in group_ids if g in group_candidates),
                default=0,
            )
            if max_k > 0:
                left_cands = pred_real["left"].new_zeros(bsz, max_k, pose_dim_v)
                right_cands = pred_real["left"].new_zeros(bsz, max_k, pose_dim_v)
                cand_mask = torch.zeros(bsz, max_k, dtype=torch.bool, device=dev)
                valid = torch.zeros(bsz, dtype=torch.bool, device=dev)
                for b in range(bsz):
                    cand = group_candidates.get(group_ids[b])
                    if cand is None:
                        continue
                    k = cand["left"].shape[0]
                    left_cands[b, :k] = cand["left"]
                    right_cands[b, :k] = cand["right"]
                    cand_mask[b, :k] = True
                    valid[b] = True
                v = valid
                tl, rl, jl = pose_component_errors(
                    pred_real["left"][v].unsqueeze(1), left_cands[v],
                    trans_dim, rot_dim, mse_trans, mse_rot, mse_joint,
                )
                tr, rr, jr = pose_component_errors(
                    pred_real["right"][v].unsqueeze(1), right_cands[v],
                    trans_dim, rot_dim, mse_trans, mse_rot, mse_joint,
                )
                combined = (tl + rl + jl) + (tr + rr + jr)
                combined.masked_fill_(~cand_mask[v], float("inf"))
                best = combined.argmin(dim=-1, keepdim=True)
                sums[0] += tl.gather(1, best).sum() + tr.gather(1, best).sum()
                sums[1] += rl.gather(1, best).sum() + rr.gather(1, best).sum()
                sums[2] += jl.gather(1, best).sum() + jr.gather(1, best).sum()
                count += 2 * v.sum()
        else:
            # one-to-one: each prediction is scored against its own record's GT.
            for i, hand in enumerate(("left", "right")):
                gt = gt_poses[:, i].float()
                if normalizers is not None:
                    gt = normalizers[hand].denormalize_pose(gt)
                trans, rot, joint = pose_component_errors(
                    pred_real[hand], gt, trans_dim, rot_dim, mse_trans, mse_rot, mse_joint
                )
                sums[0] += trans.sum()
                sums[1] += rot.sum()
                sums[2] += joint.sum()
                count += bsz

    if distributed:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    n = count.clamp_min(1.0)
    trans = (sums[0] / n).item()
    rot = (sums[1] / n).item()
    joint = (sums[2] / n).item()
    return {"trans": trans, "rot": rot, "joint": joint, "score": trans + rot + joint}


def denormalize_hand_slots(
    poses: torch.Tensor,
    sides: torch.Tensor,
    mask: torch.Tensor,
    l_obj: torch.Tensor,
    normalizers: dict | None,
) -> torch.Tensor:
    """Denormalize per-slot hand poses, picking the normalizer by hand side.

    Args:
        poses: Normalized poses, shape (B, 2, pose_dim); pad slots are zero.
        sides: Hand side ids per slot (0=left, 1=right, -1=pad), shape (B, 2).
        mask: Slot validity, shape (B, 2) bool.
        l_obj: Per-sample object characteristic length, shape (B,).
        normalizers: ``{"left", "right"}`` pose normalizers, or None when
            normalization is disabled (poses are already physical).

    Returns:
        Poses in physical units, shape (B, 2, pose_dim); pad slots stay zero.
    """
    out = poses.float().clone()
    if normalizers is None:
        return out
    mask = mask.bool()
    for side_id, name in ((0, "left"), (1, "right")):
        for s in range(poses.shape[1]):
            sel = mask[:, s] & (sides[:, s] == side_id)
            if bool(sel.any()):
                out[sel, s] = normalizers[name].denormalize_pose(
                    poses[sel, s].float(), L_obj=l_obj[sel]
                )
    return out


@torch.no_grad()
def build_group_candidates_multitask(dataset, normalizers: dict | None, device) -> dict:
    """Collect each group's GT grasps bucketed by hand-set signature.

    A signature is the tuple of sorted hand side ids present in a record
    (``(0,)`` left-only, ``(1,)`` right-only, ``(0, 1)`` bimanual), so
    predictions are only matched against candidates with the same hand set.

    Args:
        dataset: Multi-task ``DexGraspDataset`` (val split).
        normalizers: Per-hand pose normalizers or None.
        device: Device for the candidate tensors.

    Returns:
        ``{group_id: {signature: (K, H, pose_dim) tensor}}`` in physical
        units, with the H hands in ascending side-id (canonical) order.
    """
    from collections import defaultdict

    buckets: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
    for idx, record in enumerate(dataset.data):
        sides = hands_present(record)
        if not sides:
            raise ValueError(
                f"Val record idx={idx} obj_id={record.get('obj_id')!r} has no "
                "dex_grasp_left/right; cannot build group candidates."
            )
        l_obj = dataset._object_scale_length(record)
        poses = []
        for side in sides:
            pose = dataset._build_hand_pose(record, side, l_obj)
            if normalizers is not None:
                pose = normalizers[side].denormalize_pose(pose, L_obj=l_obj)
            poses.append(pose)
        sig = tuple(0 if side == "left" else 1 for side in sides)
        buckets[group_id_of(record)][sig].append(torch.stack(poses))
    return {
        gid: {
            sig: torch.stack(cands).float().to(device)
            for sig, cands in sig_buckets.items()
        }
        for gid, sig_buckets in buckets.items()
    }


def _accumulate_group_min(
    acc: MultiTaskValAccumulator,
    group_candidates: dict,
    group_ids: list[str],
    tt_ids: torch.Tensor,
    pred_real: torch.Tensor,
    pred_mask: torch.Tensor,
    pred_sides: torch.Tensor,
    trans_dim: int,
    rot_dim: int,
) -> None:
    """Accumulate group-internal best-match pose errors per sample.

    Candidates are looked up by (group_id, hand-set signature); the candidate
    minimizing the summed per-hand (trans + rot + joint) error is scored.
    Predictions whose signature has no candidate are counted as structure
    misses and excluded from the pose means.
    """
    for b in range(pred_real.shape[0]):
        slots = pred_mask[b].nonzero(as_tuple=True)[0]
        order = torch.argsort(pred_sides[b, slots])
        slots = slots[order]
        sig = tuple(int(s) for s in pred_sides[b, slots].tolist())
        sig_buckets = group_candidates.get(group_ids[b])
        cands = None if sig_buckets is None else sig_buckets.get(sig)
        if cands is None or len(sig) == 0:
            acc.add_structure_miss(int(tt_ids[b]))
            continue
        pred_h = pred_real[b, slots]
        trans, rot, joint = pose_errors_physical(
            pred_h.unsqueeze(0), cands, trans_dim, rot_dim
        )
        best = (trans + rot + joint).sum(dim=-1).argmin()
        acc.add_pose_sample(int(tt_ids[b]), trans[best], rot[best], joint[best])


@torch.no_grad()
def validate_multitask(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cfg: dict,
    rank: int,
    distributed: bool = False,
    normalizers: dict | None = None,
    group_candidates: dict | None = None,
) -> dict:
    """Multi-task validation for the latent_ar architecture.

    Two rollouts per batch keep decision and pose quality unpolluted:

      - **decision metrics** (hand_count/side/structure accuracy) come from a
        free rollout (no forcing), optionally capped to the first
        ``training.val_decision_max_batches`` batches (0 = all);
      - **pose metrics** (translation m / rotation rad / joint rad) come from a
        forced-decision rollout (``force_sides``/``force_mask`` = GT), each
        slot denormalized with its own hand-side normalizer and per-sample
        ``L_obj``, then side-aligned against GT. With ``group_candidates``
        the errors are the argmin over same-signature GT candidates of the
        sample's group.
      - **bimanual relative errors** (rel_trans/rel_rot) compare the two-hand
        relative pose against the sample's own GT record.

    All statistics accumulate into one fixed-layout tensor and are reduced
    with a single all_reduce (never a dict reduction).

    Returns:
        Dict with ``score`` (compose_score; lower is better), ``per_task``,
        ``macro``, ``bimanual``, macro ``trans``/``rot``/``joint``, and
        ``n_samples_evaluated`` (actual samples seen, summed across ranks).
    """
    model.eval()
    net = model.module if hasattr(model, "module") else model
    if net.architecture != "latent_ar":
        raise ValueError(
            "validate_multitask requires model.architecture 'latent_ar'; "
            f"got '{net.architecture}'."
        )
    device = next(net.parameters()).device
    amp_enabled, amp_dtype, _ = get_amp_settings(cfg["training"])
    amp_enabled = amp_enabled and device.type == "cuda"
    infer_cfg = cfg.get("inference", {}) or {}
    num_steps = int(infer_cfg.get("num_steps", 10))
    presence_threshold = float(infer_cfg.get("presence_threshold", 0.5))
    max_batches = int(cfg["training"].get("val_max_batches", 0))
    decision_max_batches = int(cfg["training"].get("val_decision_max_batches", 0))
    trans_dim, rot_dim = net.trans_dim, net.rot_dim

    acc = MultiTaskValAccumulator(device=device)
    n_samples_evaluated = 0

    for batch_idx, batch in enumerate(
        tqdm(dataloader, desc="Validation", disable=rank != 0)
    ):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        xyz = batch["xyz"].to(device, non_blocking=True)
        n_samples_evaluated += int(xyz.shape[0])
        rgb = batch["rgb"].to(device, non_blocking=True)
        gt_poses = batch["gt_poses"].to(device, non_blocking=True)
        hand_mask = batch["hand_mask"].to(device, non_blocking=True)
        hand_side_ids = batch["hand_side_ids"].to(device, non_blocking=True)
        l_obj = batch["L_obj"].to(device, non_blocking=True)
        texts = batch["texts"]
        tt_ids = task_type_index(batch["task_types"], device=device)

        run_decision = decision_max_batches <= 0 or batch_idx < decision_max_batches
        with autocast(device.type, enabled=amp_enabled, dtype=amp_dtype):
            if run_decision:
                free = net.sample_latent_ar(
                    xyz, rgb, texts,
                    num_steps=num_steps, presence_threshold=presence_threshold,
                )
            forced = net.sample_latent_ar(
                xyz, rgb, texts,
                num_steps=num_steps, presence_threshold=presence_threshold,
                force_sides=hand_side_ids, force_mask=hand_mask,
            )

        if run_decision:
            count_ok, side_ok, struct_ok = structure_metrics(
                free["hand_mask"], free["hand_sides"], hand_mask, hand_side_ids
            )
            acc.add_decision(tt_ids, count_ok, side_ok, struct_ok, hand_mask.sum(dim=1))

        pred_real = denormalize_hand_slots(
            forced["poses"], forced["hand_sides"], forced["hand_mask"], l_obj, normalizers
        )
        gt_real = denormalize_hand_slots(
            gt_poses, hand_side_ids, hand_mask, l_obj, normalizers
        )
        aligned_pred, matched, _ = align_hands_by_side(
            pred_real, forced["hand_mask"], forced["hand_sides"],
            gt_real, hand_mask, hand_side_ids,
        )

        if group_candidates is None:
            trans, rot, joint = pose_errors_physical(
                aligned_pred, gt_real, trans_dim, rot_dim
            )
            acc.add_pose_errors(tt_ids, trans, rot, joint, matched)
        else:
            _accumulate_group_min(
                acc, group_candidates, batch["group_ids"], tt_ids,
                pred_real, forced["hand_mask"], forced["hand_sides"],
                trans_dim, rot_dim,
            )

        pair_ok = matched.all(dim=1) & (hand_mask.sum(dim=1) == 2)
        if bool(pair_ok.any()):
            rel_trans, rel_rot = relative_pose_error(
                aligned_pred[pair_ok], gt_real[pair_ok], trans_dim, rot_dim
            )
            acc.add_relative(tt_ids[pair_ok], rel_trans, rel_rot)

    acc.all_reduce(distributed)
    if distributed and dist.is_available() and dist.is_initialized():
        n_samples_tensor = torch.tensor(
            float(n_samples_evaluated), dtype=torch.float64, device=device
        )
        dist.all_reduce(n_samples_tensor, op=dist.ReduceOp.SUM)
        n_samples_evaluated = int(n_samples_tensor.item())
    summary = acc.summary()
    per_task = summary["per_task"]

    per_tt_errors = {
        tt: {k: m[k] for k in ("trans", "rot", "joint")}
        for tt, m in per_task.items()
        if m["n_hand"] > 0
    }
    structure_acc = {
        tt: m["structure_acc"] for tt, m in per_task.items() if m["n_decision"] > 0
    }
    rel_errors = {
        tt: {"rel_trans": m["rel_trans"], "rel_rot": m["rel_rot"]}
        for tt, m in per_task.items()
        if tt in BIMANUAL and m["n_rel"] > 0
    }
    score = compose_score(
        per_tt_errors,
        structure_acc,
        rel_errors,
        cfg["training"].get("val_score_weights", None),
        reduce=str(cfg["training"].get("val_score_reduce", "macro")),
    )
    return {
        "score": score,
        "per_task": per_task,
        "macro": summary["macro"],
        "bimanual": summary["bimanual"],
        "trans": summary["macro"]["trans"],
        "rot": summary["macro"]["rot"],
        "joint": summary["macro"]["joint"],
        "n_samples_evaluated": int(n_samples_evaluated),
    }


def format_multitask_val_table(per_task: dict) -> list[str]:
    """Render the per-task-type validation table as log lines.

    Args:
        per_task: ``validate_multitask``'s per-task summary.

    Returns:
        List of aligned text lines (header + one row per task type).
    """

    def fmt(value: float, width: int = 7, prec: int = 4) -> str:
        if math.isfinite(value):
            return f"{value:{width}.{prec}f}"
        return " " * (width - 1) + "-"

    lines = [
        "  task_type  n_hand   trans     rot   joint | n_dec cnt_acc sid_acc str_acc"
        " | n_rel   rel_t   rel_r | miss"
    ]
    for tt in TASK_TYPES:
        m = per_task[tt]
        lines.append(
            f"  {tt:<9}  {m['n_hand']:>6} {fmt(m['trans'])} {fmt(m['rot'])}"
            f" {fmt(m['joint'])} | {m['n_decision']:>5} {fmt(m['count_acc'])}"
            f" {fmt(m['side_acc'])} {fmt(m['structure_acc'])} | {m['n_rel']:>5}"
            f" {fmt(m['rel_trans'])} {fmt(m['rel_rot'])} | {m['n_struct_miss']:>4}"
        )
    return lines


def scenario_composition(
    sampler: GroupedEpochSampler, records: list[dict]
) -> dict[str, int]:
    """Count per-task_type record draws in the sampler's current epoch shard.

    Re-iterating the sampler is deterministic for a fixed epoch, so this does
    not disturb the subsequent DataLoader iteration.

    Args:
        sampler: The epoch sampler after ``set_epoch``.
        records: The dataset's raw records.

    Returns:
        ``{task_type: count}`` sorted by task type.
    """
    counts: dict[str, int] = {}
    for idx in iter(sampler):
        tt = str(records[idx].get("task_type", ""))
        counts[tt] = counts.get(tt, 0) + 1
    return dict(sorted(counts.items()))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    val_loss: float,
    path: str,
    val_metrics: dict[str, float] | None = None,
    ema: _EMA | None = None,
) -> None:
    state = model.module if hasattr(model, "module") else model
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": state.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_loss": val_loss,
        "val_metrics": val_metrics or {},
        # Which weights produced the validation score: with EMA on, validation
        # runs the EMA shadow, so consumers should load ema_state_dict
        # (--use-ema) to match the score.
        "val_weights": "ema" if ema is not None else "raw",
    }
    if ema is not None:
        ckpt["ema_state_dict"] = ema.state_dict()
    # Atomic write: crash/kill mid-save must not leave a truncated checkpoint
    # at `path` (os.replace is atomic within the same directory/filesystem).
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


_BEST_CKPT_RE = re.compile(r"best_e\d+_s([0-9]+\.[0-9]+)\.pt$")


def scan_best_checkpoints(ckpt_dir: str) -> list[tuple[float, str]]:
    """Rebuild the best-checkpoint registry from existing files (for resume).

    Returns a list of ``(score, path)`` sorted ascending (best first).
    """
    registry: list[tuple[float, str]] = []
    if not os.path.isdir(ckpt_dir):
        return registry
    for name in os.listdir(ckpt_dir):
        match = _BEST_CKPT_RE.match(name)
        if match:
            registry.append((float(match.group(1)), os.path.join(ckpt_dir, name)))
    registry.sort(key=lambda item: item[0])
    return registry


def update_best_checkpoints(
    registry: list[tuple[float, str]],
    score: float,
    keep_best_k: int,
    ckpt_dir: str,
    epoch: int,
    save_fn,
) -> tuple[list[tuple[float, str]], bool]:
    """Keep only the ``keep_best_k`` lowest-score checkpoints on disk.

    Saves the current checkpoint when it ranks within the top-k, evicting and
    deleting the worst retained one if the directory is full. ``save_fn(path)``
    performs the actual ``torch.save``. Returns the updated registry and whether
    the current checkpoint was kept.
    """
    if keep_best_k <= 0:
        return registry, False
    if len(registry) >= keep_best_k and score >= registry[-1][0]:
        return registry, False  # not good enough to enter the top-k

    path = os.path.join(ckpt_dir, f"best_e{epoch:04d}_s{score:.4f}.pt")
    save_fn(path)
    registry.append((score, path))
    registry.sort(key=lambda item: item[0])
    while len(registry) > keep_best_k:
        _, drop_path = registry.pop()  # worst (highest score)
        if os.path.exists(drop_path):
            os.remove(drop_path)
    return registry, True


def maybe_launch_bench_subset(
    cfg: dict,
    output_dir: str,
    ckpt_dir: str,
    epoch: int,
    numbered_ckpt: str | None,
    logger: logging.Logger,
) -> None:
    """Fire-and-forget async bench-subset hook (``training.bench_eval``).

    Rank-0 only, called at epoch end after checkpoints are on disk. Every
    ``interval`` epochs it detaches ``scripts/bench_subset.sh`` (new session,
    no wait/join) to measure the real-simulation success rate on a bench
    subset. ZERO-impact contract with training: never blocks, never touches
    the training GPUs (the script picks an idle card and excludes the
    inherited ``CUDA_VISIBLE_DEVICES``), and any launch failure is swallowed
    into a warning.

    Args:
        cfg: Full config; reads ``training.bench_eval`` (missing = disabled).
        output_dir: Training run directory (holds config.yaml backup).
        ckpt_dir: Checkpoint directory (for last.pt).
        epoch: 0-based epoch just finished; fires when (epoch+1) % interval == 0.
        numbered_ckpt: Path of the epoch_XXXX.pt saved this epoch, or None.
            When None, last.pt is copied to a stable ``bench_snap_e{N}.pt``
            first (last.pt is overwritten every epoch, so the async reader
            must never point at it directly).
        logger: Training logger.
    """
    be_cfg = dict(cfg["training"].get("bench_eval", {}) or {})
    if not bool(be_cfg.get("enabled", False)):
        return
    interval = int(be_cfg.get("interval", 20))
    if interval <= 0 or (epoch + 1) % interval != 0:
        return
    try:
        bench_dir = os.path.join(output_dir, "bench_subset")
        os.makedirs(bench_dir, exist_ok=True)
        if numbered_ckpt is not None:
            ckpt_path = numbered_ckpt
        else:
            ckpt_path = os.path.join(output_dir, f"bench_snap_e{epoch}.pt")
            shutil.copy2(os.path.join(ckpt_dir, "last.pt"), ckpt_path)
        config_path = os.path.join(output_dir, "config.yaml")  # startup backup
        exp_tag = f"{os.path.basename(os.path.normpath(output_dir))}_e{epoch}_sub"
        test_root = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(test_root, "scripts", "bench_subset.sh")
        gpu = be_cfg.get("gpu", "")
        cmd = [
            "bash", script, ckpt_path, config_path, exp_tag,
            str(be_cfg.get("channels", "lgbidex")),
            str(be_cfg.get("max_num", 6000)),
            "" if gpu is None else str(gpu),
            str(be_cfg.get("n_worker", 48)),
        ]
        launch_log = os.path.join(bench_dir, f"launch_e{epoch}.log")
        with open(launch_log, "ab") as log_f:
            subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=test_root,
            )
        logger.info(
            f"[epoch {epoch}] bench_eval hook launched (async, detached): "
            f"exp={exp_tag} ckpt={ckpt_path} log={launch_log}"
        )
    except Exception as exc:  # zero-impact principle: never break training
        logger.warning(
            f"[epoch {epoch}] bench_eval hook launch FAILED "
            f"(training unaffected): {exc!r}"
        )


def main():
    args = parse_args()
    if args.init_weights and args.resume:
        raise ValueError(
            "--init-weights and --resume are mutually exclusive: --resume "
            "restores the full training state, --init-weights only warm-starts "
            "model weights (fresh optimizer/scheduler/epoch)."
        )
    cfg = load_config(args.config)
    rank, local_rank = setup_distributed()
    distributed = dist.is_initialized()

    train_cfg = cfg["training"]
    # Rank-offset so per-step RNG draws (flow t/noise, hand-cond noise, dropout,
    # CFG drop) differ across ranks; the samplers run their own base_seed-driven
    # generators so sharding stays rank-consistent, and DDP broadcasts rank-0
    # weights at wrap time so init still matches.
    set_seed(int(train_cfg.get("seed", 42)) + rank)

    multi_task = bool(cfg["data"].get("multi_task", False))
    architecture = str(cfg["model"].get("architecture", "legacy_bimanual"))
    # training.validate_enabled=false: skip ALL training-time validation and
    # best-K selection (val's group-min matching proved blind to bench
    # regressions); the only quality signal is the async bench_eval hook.
    validate_enabled = bool(train_cfg.get("validate_enabled", True))
    if multi_task and architecture != "latent_ar":
        raise ValueError(
            "data.multi_task: true requires model.architecture: latent_ar; "
            f"got '{architecture}'. The legacy bimanual path only supports "
            "multi_task: false."
        )

    output_dir = args.output_dir or cfg["paths"]["output_dir"]
    ckpt_dir = os.path.join(output_dir, cfg["paths"]["checkpoint_dir"])
    log_dir = os.path.join(output_dir, cfg["paths"]["log_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = setup_logging(log_dir, rank)

    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir)
        # back up the effective config alongside the run for reproducibility
        config_backup = os.path.join(output_dir, "config.yaml")
        with open(config_backup, "w") as cf:
            yaml.safe_dump(cfg, cf, sort_keys=False, allow_unicode=True)
        logger.info("=== DexVLG Training ===")
        logger.info(f"Config: {args.config}")
        logger.info(f"Config backup: {config_backup}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Text log: {os.path.join(log_dir, 'train.log')}")

    model = DexVLG(cfg["model"]).cuda()

    if rank == 0:
        total = count_parameters(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {total:,} total, {trainable:,} trainable")

    if args.init_weights:
        # Warm-start weights only (strict=False so config-added params such as
        # the CFG null_cond keep their fresh init). Loaded BEFORE the DDP wrap
        # so the construction-time rank-0 broadcast propagates the weights, and
        # before the EMA shadow deep-copy so an enabled EMA starts from the
        # loaded weights instead of random init. Optimizer/scheduler/epoch/
        # global_step deliberately start fresh.
        init_ckpt = torch.load(
            args.init_weights, map_location="cuda", weights_only=False
        )
        incompat = model.load_state_dict(
            init_ckpt["model_state_dict"], strict=False
        )
        if rank == 0:
            logger.info(
                f"Warm-start --init-weights from {args.init_weights} "
                f"(ckpt epoch={init_ckpt.get('epoch', '?')}, strict=False): "
                f"{len(incompat.missing_keys)} missing / "
                f"{len(incompat.unexpected_keys)} unexpected keys; "
                "optimizer/scheduler/epoch/global_step start fresh"
            )
            if incompat.missing_keys:
                logger.info(
                    "  missing (new params keep fresh init), sample: "
                    f"{incompat.missing_keys[:8]}"
                )
            if incompat.unexpected_keys:
                logger.warning(
                    "  unexpected (ignored), sample: "
                    f"{incompat.unexpected_keys[:8]}"
                )
        del init_ckpt

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    ema_cfg = train_cfg.get("ema", {}) or {}
    use_ema = bool(ema_cfg.get("enabled", False))
    ema = None
    if use_ema:
        ema_decay = float(ema_cfg.get("decay", 0.9999))
        base = model.module if hasattr(model, "module") else model
        ema = _EMA(base, decay=ema_decay)
        if rank == 0:
            logger.info(f"EMA enabled: decay={ema_decay}")

    data_cfg = cfg["data"]
    dataset_kwargs = dict(
        point_cloud_source=data_cfg.get("point_cloud_source", "auto"),
        point_cloud_color_mode=data_cfg.get("point_cloud_color_mode", "real"),
        point_cloud_color_fill=data_cfg.get("point_cloud_color_fill", 0.4),
        point_cloud_file_points=data_cfg.get("point_cloud_file_points", None),
        center_on_object=data_cfg.get("center_on_object", True),
        obj_pose_quaternion_order=data_cfg.get("obj_pose_quaternion_order", "wxyz"),
        normalization=data_cfg.get("normalization", None),
        multi_task=multi_task,
        grace_targets=data_cfg.get("grace_targets", None),
    )
    train_dataset = DexGraspDataset(
        data_path=data_cfg["train_data"],
        mesh_root=data_cfg["mesh_root"],
        n_points=data_cfg.get("n_points", 4096),
        joint_dim=cfg["model"].get("joint_dim", 22),
        augment=data_cfg.get("augment", False),
        **dataset_kwargs,
    )
    # Keep workers (and their warm point-cloud caches) alive across epochs and
    # prefetch batches so disk/mesh I/O overlaps with the GPU step. prefetch_factor
    # and persistent_workers are only valid when num_workers > 0.
    num_workers = train_cfg.get("num_workers", 4)
    loader_kwargs: dict = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = train_cfg.get("persistent_workers", True)
        loader_kwargs["prefetch_factor"] = train_cfg.get("prefetch_factor", 2)

    # Optional per-epoch grouped resampling: each epoch draw N grasps per
    # (obj_id, pose_id, guidance) combo instead of using every record. Keeps the
    # original full-dataset path when disabled.
    world_size = dist.get_world_size() if distributed else 1
    es_cfg = dict(train_cfg.get("epoch_sampling", {}) or {})
    use_epoch_sampling = bool(es_cfg.get("enabled", False))
    group_keys = es_cfg.get("group_keys", ["obj_id", "pose_id", "guidance"])
    seed = train_cfg.get("seed", 42)
    scenario_key = es_cfg.get("scenario_key")
    if rank == 0 and use_epoch_sampling:
        logger.info(
            f"Epoch sampling ON: {es_cfg.get('samples_per_group', 4)} per "
            f"{'+'.join(group_keys)} combo (train), "
            f"{es_cfg.get('val_samples_per_group', es_cfg.get('samples_per_group', 4))} (val)."
        )
        if scenario_key:
            logger.info(
                f"Scenario balancing ON: key={scenario_key} "
                f"shares={es_cfg.get('scenario_shares')} "
                f"epoch_size={es_cfg.get('epoch_size')}"
            )

    if use_epoch_sampling:
        train_sampler = GroupedEpochSampler(
            train_dataset.data, group_keys, int(es_cfg.get("samples_per_group", 4)),
            shuffle=True, num_replicas=world_size, rank=rank, base_seed=seed, drop_last=True,
            scenario_key=scenario_key,
            scenario_shares=es_cfg.get("scenario_shares"),
            epoch_size=es_cfg.get("epoch_size"),
        )
    else:
        train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        **loader_kwargs,
    )

    # validate disabled -> no val_loader at all: every downstream validation
    # call site (per-epoch + final) is guarded by `val_loader is not None`.
    val_loader = None
    if validate_enabled and os.path.exists(data_cfg.get("val_data", "")):
        val_dataset = DexGraspDataset(
            data_path=data_cfg["val_data"],
            mesh_root=data_cfg["mesh_root"],
            n_points=data_cfg.get("n_points", 4096),
            joint_dim=cfg["model"].get("joint_dim", 22),
            augment=False,
            **dataset_kwargs,
        )
        if use_epoch_sampling:
            # Fixed (epoch 0) grouped subset for a stable, comparable val metric.
            # multi_task shuffles the (still epoch-fixed, deterministic) order so
            # a val_max_batches cap sees a mix of task types, not one data-order
            # contiguous scenario.
            val_per_group = int(es_cfg.get("val_samples_per_group", es_cfg.get("samples_per_group", 4)))
            val_sampler = GroupedEpochSampler(
                val_dataset.data, group_keys, val_per_group, shuffle=multi_task,
                num_replicas=world_size, rank=rank, base_seed=seed, drop_last=False,
            )
        else:
            if multi_task:
                # Same rationale as above: the fuse splits are contiguous
                # task_type blocks, so a val_max_batches cap over sequential
                # order would validate a single scenario. Fixed permutation
                # (no set_epoch / epoch-0 shuffle) keeps the metric stable.
                if distributed:
                    val_sampler = DistributedSampler(val_dataset, shuffle=True, seed=int(seed))
                else:
                    val_gen = torch.Generator()
                    val_gen.manual_seed(int(seed))
                    val_sampler = torch.randperm(len(val_dataset), generator=val_gen).tolist()
            else:
                val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            **loader_kwargs,
        )

    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    total_steps = len(train_loader) * train_cfg["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, train_cfg.get("warmup_steps", 1000), total_steps,
    )
    _, _, use_scaler = get_amp_settings(train_cfg)
    scaler = GradScaler("cuda", enabled=use_scaler)

    start_epoch = 0
    global_step = 0
    keep_best_k = train_cfg.get("keep_best_k", 4)
    # Rolling latest checkpoint (last.pt) is saved every epoch; a numbered
    # snapshot (epoch_XXXX.pt) is kept every `save_interval` epochs (0 disables).
    save_interval = int(train_cfg.get("save_interval", 20))
    # registry of (score, path) for the best checkpoints kept on disk (rank 0).
    # Only inherited on --resume: a fresh run must not adopt a previous run's
    # best_*.pt (their scores are not comparable and inherited entries would be
    # evicted/DELETED by this run's top-K bookkeeping). With validate disabled
    # the registry stays empty: no best_*.pt is produced, and pre-existing ones
    # are never adopted (hence never evicted/deleted).
    best_registry = (
        scan_best_checkpoints(ckpt_dir)
        if (rank == 0 and args.resume and validate_enabled)
        else []
    )
    if rank == 0 and validate_enabled and not args.resume:
        stale_best = scan_best_checkpoints(ckpt_dir)
        if stale_best:
            logger.warning(
                "!" * 70 + "\n"
                f"ckpt_dir 已存在 {len(stale_best)} 个旧 run 的 best_*.pt（未 --resume，"
                "本次 fresh run 不继承其注册表）：旧 best 文件会与新 run 的 best 混在同一目录，"
                "同名冲突时可能被覆盖；建议先清理/移走旧 best，或改用 --resume。\n"
                + "\n".join(f"  {p} (score={s:.4f})" for s, p in stale_best) + "\n"
                + "!" * 70
            )
    best_score = best_registry[0][0] if best_registry else float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cuda", weights_only=False)
        base_model = model.module if hasattr(model, "module") else model
        base_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        if ema is not None:
            if "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            else:
                # The shadow was deep-copied from the pre-resume (random-init)
                # model; rebuild from the loaded weights or every validation
                # would score a near-random shadow.
                ema = _EMA(base_model, decay=ema.decay)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
            if validate_enabled:
                logger.info(
                    f"Existing top-{keep_best_k} best score: {best_score:.4f}"
                )

    val_interval = train_cfg.get("val_interval", 1)
    # per-hand pose denormalizers for validation metrics (None if disabled)
    normalizers = build_hand_normalizers(
        data_cfg.get("normalization", None), joint_dim=cfg["model"].get("joint_dim", 22)
    )
    # Shared context for per-epoch validation + best-K checkpointing.
    vc = {
        "model": model,
        "val_loader": val_loader,
        "cfg": cfg,
        "rank": rank,
        "distributed": distributed,
        "writer": writer,
        "logger": logger,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "ckpt_dir": ckpt_dir,
        "keep_best_k": keep_best_k,
        "best_registry": best_registry,
        "val_interval": val_interval,
        "normalizers": normalizers,
        "ema": ema,
    }

    if rank == 0:
        _amp_en, _amp_dt, _amp_sc = get_amp_settings(train_cfg)
        _prec = str(_amp_dt).replace("torch.", "") if _amp_en else "fp32"
        logger.info(f"Training for {train_cfg['epochs']} epochs, {total_steps} steps")
        logger.info(
            f"Batch size: {train_cfg['batch_size']}, LR: {train_cfg['lr']}, "
            f"precision: {_prec} (grad_scaler={_amp_sc})"
        )
        if not validate_enabled:
            logger.info("!" * 70)
            logger.info(
                "VALIDATE DISABLED (training.validate_enabled=false): 训练期不跑任何 "
                "validate/best-K 选型（不产生 best_*.pt）；checkpoint 仅 last.pt + "
                "save_interval 快照；训练期唯一质量信号 = bench 子集钩子 "
                "(training.bench_eval)"
            )
            logger.info("!" * 70)
        else:
            logger.info(
                f"Validation: every {val_interval} epoch(s) via "
                f"{cfg.get('inference', {}).get('num_steps', 10)}-step sampling"
                + (f", normalized={normalizers is not None}")
                + (" (val disabled: no val_data)" if val_loader is None else "")
            )
        _be_cfg = dict(train_cfg.get("bench_eval", {}) or {})
        if bool(_be_cfg.get("enabled", False)):
            _be_gpu = _be_cfg.get("gpu", "")
            _be_gpu = "" if _be_gpu is None else str(_be_gpu)
            logger.info(
                f"bench_eval hook ON (async, detached, 0 training impact): every "
                f"{int(_be_cfg.get('interval', 20))} epoch(s), "
                f"channels={_be_cfg.get('channels', 'lgbidex')}, "
                f"max_num={_be_cfg.get('max_num', 6000)}, "
                f"gpu='{_be_gpu}' (空=自动找空闲卡), "
                f"n_worker={_be_cfg.get('n_worker', 48)}"
            )
        logger.info("-" * 60)

    last_epoch = train_cfg["epochs"] - 1
    validated_last_epoch = False
    for epoch in range(start_epoch, train_cfg["epochs"]):
        # Re-seed the per-epoch sampler so each epoch draws a fresh subset
        # (DistributedSampler also needs this for correct cross-epoch shuffling).
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if rank == 0 and multi_task and isinstance(train_sampler, GroupedEpochSampler):
            logger.info(
                f"[epoch {epoch}] train scenario composition (rank-0 shard): "
                f"{scenario_composition(train_sampler, train_dataset.data)}"
            )

        train_loss, global_step = train_one_epoch(vc, train_loader, epoch, global_step)

        if rank == 0:
            logger.info(
                f"[epoch {epoch} done] train_loss={train_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # Per-epoch validation. (epoch + 1) % N avoids validating after epoch 0.
        validated_last_epoch = False
        if (
            val_loader is not None
            and val_interval > 0
            and (epoch + 1) % val_interval == 0
        ):
            run_validation(vc, epoch, global_step)
            validated_last_epoch = epoch == last_epoch

        # Rolling latest checkpoint every epoch so an interrupted run can always
        # resume from the newest state; plus a numbered snapshot every
        # `save_interval` epochs. Runs after validation so best_registry (hence
        # the recorded best score) is current.
        if rank == 0:
            cur_best = vc["best_registry"][0][0] if vc["best_registry"] else float("inf")
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, global_step, cur_best,
                os.path.join(ckpt_dir, "last.pt"), ema=ema,
            )
            numbered_ckpt = None
            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                snap_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, cur_best, snap_path, ema=ema,
                )
                logger.info(f"[epoch {epoch}] saved periodic checkpoint {os.path.basename(snap_path)}")
                numbered_ckpt = snap_path
            # Async bench-subset hook: launched only after this epoch's
            # checkpoints are on disk; fire-and-forget (never blocks training).
            maybe_launch_bench_subset(
                cfg, output_dir, ckpt_dir, epoch, numbered_ckpt, logger
            )

    # Final validation so the last state can enter top-K, unless already done.
    if val_loader is not None and not validated_last_epoch:
        run_validation(vc, last_epoch, global_step)
    best_registry = vc["best_registry"]
    best_score = best_registry[0][0] if best_registry else float("inf")

    if rank == 0:
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            train_cfg["epochs"] - 1, global_step, best_score,
            os.path.join(ckpt_dir, "last.pt"), ema=ema,
        )
        logger.info("Training complete.")
        if best_registry:
            logger.info(f"Best val score: {best_score:.4f}")
            logger.info(f"Kept top-{keep_best_k} checkpoints by val score:")
            for s, p in best_registry:
                logger.info(f"  {s:.4f}  {os.path.basename(p)}")
        else:
            logger.info("No validation ran; saved last.pt only.")
        if writer:
            writer.close()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
