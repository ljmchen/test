"""Training script for DexVLG bimanual dexterous grasp generation.

Trains the full model end-to-end: PC encoder, projector, fusion transformer,
and flow-matching head, using conditional flow matching loss.

Usage:
    python train.py --config configs/default.yaml [--resume PATH]
"""

import argparse
import logging
import math
import os
import re

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

from data.dataset import create_dataloader, DexGraspDataset, collate_fn
from data.pose_normalizer import build_hand_normalizers
from models.dexvlg import DexVLG
from utils.rotation import rotation_6d_to_matrix
from utils.misc import set_seed, count_parameters, AverageMeter


def geodesic_angle(pred_rot6d: torch.Tensor, gt_rot6d: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (radians) between two 6D rotations, shape (...,)."""
    r_pred = rotation_6d_to_matrix(pred_rot6d)
    r_gt = rotation_6d_to_matrix(gt_rot6d)
    r_rel = torch.matmul(r_pred.transpose(-1, -2), r_gt)
    trace = r_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.acos(cos_angle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DexVLG model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
    loss_meter = AverageMeter()
    use_fp16 = cfg["training"].get("fp16", True)
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    log_interval = cfg["training"].get("log_interval", 50)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=rank != 0)
    for batch in pbar:
        xyz = batch["xyz"].cuda(non_blocking=True)
        rgb = batch["rgb"].cuda(non_blocking=True)
        gt_poses = batch["gt_poses"].cuda(non_blocking=True)
        texts = batch["texts"]

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_fp16):
            outputs = model(xyz, rgb, texts, gt_poses=gt_poses)
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = loss.item()
        loss_meter.update(loss_val, xyz.shape[0])
        global_step += 1

        if rank == 0:
            pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{loss_meter.avg:.4f}")
            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                if writer:
                    writer.add_scalar("train/loss", loss_val, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                logger.info(
                    f"[epoch {epoch} step {global_step}] "
                    f"train_loss={loss_val:.4f} avg={loss_meter.avg:.4f} lr={lr:.2e}"
                )

    return loss_meter.avg, global_step


def run_validation(vc: dict, epoch: int, global_step: int) -> None:
    """Run sampling-based validation, log it, and update best-K checkpoints.

    Collective: every rank must call this together. Only rank 0 logs/saves.
    """
    model = vc["model"]
    val_metrics = validate(
        model, vc["val_loader"], vc["cfg"], vc["rank"], vc["distributed"],
        normalizers=vc.get("normalizers"),
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

    score = val_metrics["score"]
    vc["best_registry"], kept = update_best_checkpoints(
        vc["best_registry"], score, vc["keep_best_k"], vc["ckpt_dir"], epoch,
        lambda path: save_checkpoint(
            model, vc["optimizer"], vc["scheduler"], vc["scaler"],
            epoch, global_step, score, path, val_metrics,
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


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cfg: dict,
    rank: int,
    distributed: bool = False,
    normalizers: dict | None = None,
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
    use_fp16 = cfg["training"].get("fp16", True)
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

        with autocast("cuda", enabled=use_fp16):
            pred = net.sample(xyz, rgb, texts, num_steps=num_steps)

        for i, hand in enumerate(["left", "right"]):
            gt = gt_poses[:, i].float()
            pred_pose = torch.cat(
                [
                    pred[f"{hand}_translation"].float(),
                    pred[f"{hand}_rotation_6d"].float(),
                    pred[f"{hand}_joints"].float(),
                ],
                dim=-1,
            )
            # back to real units (meters / radians) so the reported errors and
            # the checkpoint-selection score are interpretable.
            if normalizers is not None:
                gt = normalizers[hand].denormalize_pose(gt)
                pred_pose = normalizers[hand].denormalize_pose(pred_pose)

            gt_trans = gt[:, :trans_dim]
            gt_rot6d = gt[:, trans_dim : trans_dim + rot_dim]
            gt_joints = gt[:, trans_dim + rot_dim :]
            pred_trans = pred_pose[:, :trans_dim]
            pred_rot6d = pred_pose[:, trans_dim : trans_dim + rot_dim]
            pred_joints = pred_pose[:, trans_dim + rot_dim :]

            # translation: MSE (mean squared) or L2 Euclidean distance
            if mse_trans:
                sums[0] += ((pred_trans - gt_trans) ** 2).mean(dim=-1).sum()
            else:
                sums[0] += torch.norm(pred_trans - gt_trans, dim=-1).sum()
            # rotation: MSE on the 6D representation, or geodesic angle (radians)
            if mse_rot:
                sums[1] += ((pred_rot6d - gt_rot6d) ** 2).mean(dim=-1).sum()
            else:
                sums[1] += geodesic_angle(pred_rot6d, gt_rot6d).sum()
            # joint: MSE (mean squared) or L1 mean absolute error (radians)
            if mse_joint:
                sums[2] += ((pred_joints - gt_joints) ** 2).mean(dim=-1).sum()
            else:
                sums[2] += torch.abs(pred_joints - gt_joints).mean(dim=-1).sum()
            count += bsz

    if distributed:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    n = count.clamp_min(1.0)
    trans = (sums[0] / n).item()
    rot = (sums[1] / n).item()
    joint = (sums[2] / n).item()
    return {"trans": trans, "rot": rot, "joint": joint, "score": trans + rot + joint}


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
) -> None:
    state = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": state.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_loss": val_loss,
            "val_metrics": val_metrics or {},
        },
        path,
    )


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


def main():
    args = parse_args()
    cfg = load_config(args.config)
    rank, local_rank = setup_distributed()
    distributed = dist.is_initialized()

    train_cfg = cfg["training"]
    set_seed(train_cfg.get("seed", 42))

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

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    data_cfg = cfg["data"]
    dataset_kwargs = dict(
        point_cloud_source=data_cfg.get("point_cloud_source", "auto"),
        point_cloud_color_mode=data_cfg.get("point_cloud_color_mode", "real"),
        point_cloud_color_fill=data_cfg.get("point_cloud_color_fill", 0.4),
        point_cloud_file_points=data_cfg.get("point_cloud_file_points", None),
        center_on_object=data_cfg.get("center_on_object", True),
        obj_pose_quaternion_order=data_cfg.get("obj_pose_quaternion_order", "wxyz"),
        normalization=data_cfg.get("normalization", None),
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
        loader_kwargs["prefetch_factor"] = train_cfg.get("prefetch_factor", 4)

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

    val_loader = None
    if os.path.exists(data_cfg.get("val_data", "")):
        val_dataset = DexGraspDataset(
            data_path=data_cfg["val_data"],
            mesh_root=data_cfg["mesh_root"],
            n_points=data_cfg.get("n_points", 4096),
            joint_dim=cfg["model"].get("joint_dim", 22),
            augment=False,
            **dataset_kwargs,
        )
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
    scaler = GradScaler("cuda", enabled=train_cfg.get("fp16", True))

    start_epoch = 0
    global_step = 0
    keep_best_k = train_cfg.get("keep_best_k", 4)
    # registry of (score, path) for the best checkpoints kept on disk (rank 0).
    best_registry = scan_best_checkpoints(ckpt_dir) if rank == 0 else []
    best_score = best_registry[0][0] if best_registry else float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cuda", weights_only=False)
        base_model = model.module if hasattr(model, "module") else model
        base_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
            logger.info(f"Existing top-{keep_best_k} best score: {best_score:.4f}")

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
    }

    if rank == 0:
        logger.info(f"Training for {train_cfg['epochs']} epochs, {total_steps} steps")
        logger.info(f"Batch size: {train_cfg['batch_size']}, LR: {train_cfg['lr']}")
        logger.info(
            f"Validation: every {val_interval} epoch(s) via "
            f"{cfg.get('inference', {}).get('num_steps', 10)}-step sampling"
            + (f", normalized={normalizers is not None}")
            + (" (val disabled: no val_data)" if val_loader is None else "")
        )
        logger.info("-" * 60)

    last_epoch = train_cfg["epochs"] - 1
    validated_last_epoch = False
    for epoch in range(start_epoch, train_cfg["epochs"]):
        if distributed:
            train_sampler.set_epoch(epoch)

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

    # Final validation so the last state can enter top-K, unless already done.
    if val_loader is not None and not validated_last_epoch:
        run_validation(vc, last_epoch, global_step)
    best_registry = vc["best_registry"]
    best_score = best_registry[0][0] if best_registry else float("inf")

    if rank == 0:
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            train_cfg["epochs"] - 1, global_step, best_score,
            os.path.join(ckpt_dir, "last.pt"),
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
