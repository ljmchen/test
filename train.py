"""Training script for DexVLG bimanual dexterous grasp generation.

Implements two-stage training following the DexVLG paradigm:
    Stage 1: Train the full model end-to-end (PC encoder, projector,
             fusion transformer, flow-matching head).
    Stage 2: Freeze the vision-language backbone and fine-tune only the
             flow-matching head, which resolves optimization conflicts
             between the VLM and denoising objectives.

Usage:
    python train.py --config configs/default.yaml [--stage 1|2] [--resume PATH]
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

from data.dataset import create_dataloader, DexGraspDataset, collate_fn
from models.dexvlg import DexVLG
from utils.misc import set_seed, count_parameters, AverageMeter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DexVLG model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def freeze_backbone(model: DexVLG) -> None:
    """Freeze vision-language backbone, keep flow-matching head trainable."""
    for name, param in model.named_parameters():
        if not any(
            key in name
            for key in ["flow_transformer", "hand_embed"]
        ):
            param.requires_grad = False


def setup_distributed() -> tuple[int, int]:
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank
    return 0, 0


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    cfg: dict,
    writer: SummaryWriter | None,
    rank: int,
    global_step: int,
) -> tuple[float, int]:
    """Train for one epoch, return average loss and updated global step."""
    model.train()
    loss_meter = AverageMeter()
    use_fp16 = cfg["training"].get("fp16", True)
    grad_clip = cfg["training"].get("_grad_clip", 1.0)
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
            if writer and global_step % log_interval == 0:
                writer.add_scalar("train/loss", loss_val, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

    return loss_meter.avg, global_step


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cfg: dict,
    rank: int,
) -> float:
    """Run validation, return average loss."""
    model.eval()
    loss_meter = AverageMeter()
    use_fp16 = cfg["training"].get("fp16", True)

    for batch in tqdm(dataloader, desc="Validation", disable=rank != 0):
        xyz = batch["xyz"].cuda(non_blocking=True)
        rgb = batch["rgb"].cuda(non_blocking=True)
        gt_poses = batch["gt_poses"].cuda(non_blocking=True)
        texts = batch["texts"]

        with autocast("cuda", enabled=use_fp16):
            outputs = model(xyz, rgb, texts, gt_poses=gt_poses)

        loss_meter.update(outputs["loss"].item(), xyz.shape[0])

    return loss_meter.avg


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    val_loss: float,
    path: str,
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
        },
        path,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    rank, local_rank = setup_distributed()
    distributed = dist.is_initialized()

    set_seed(cfg["training"].get("seed", 42))

    stage_cfg = cfg["training"][f"stage{args.stage}"]
    output_dir = args.output_dir or cfg["paths"]["output_dir"]
    ckpt_dir = os.path.join(output_dir, cfg["paths"]["checkpoint_dir"], f"stage{args.stage}")
    log_dir = os.path.join(output_dir, cfg["paths"]["log_dir"], f"stage{args.stage}")
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = None
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

    if rank == 0:
        print(f"=== DexVLG Training Stage {args.stage} ===")
        print(f"Config: {args.config}")
        print(f"Output: {output_dir}")

    model = DexVLG(cfg["model"]).cuda()

    if args.stage == 2:
        freeze_backbone(model)

    if rank == 0:
        total = count_parameters(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {total:,} total, {trainable:,} trainable")

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    data_cfg = cfg["data"]
    train_dataset = DexGraspDataset(
        data_path=data_cfg["train_data"],
        mesh_root=data_cfg["mesh_root"],
        n_points=data_cfg.get("n_points", 10000),
        joint_dim=cfg["model"].get("joint_dim", 22),
        augment=data_cfg.get("augment", True),
    )
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=stage_cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg["training"].get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if os.path.exists(data_cfg.get("val_data", "")):
        val_dataset = DexGraspDataset(
            data_path=data_cfg["val_data"],
            mesh_root=data_cfg["mesh_root"],
            n_points=data_cfg.get("n_points", 10000),
            joint_dim=cfg["model"].get("joint_dim", 22),
            augment=False,
        )
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=stage_cfg["batch_size"],
            shuffle=False,
            sampler=val_sampler,
            num_workers=cfg["training"].get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=stage_cfg["lr"],
        weight_decay=stage_cfg.get("weight_decay", 1e-4),
    )

    total_steps = len(train_loader) * stage_cfg["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, stage_cfg.get("warmup_steps", 1000), total_steps,
    )
    scaler = GradScaler("cuda", enabled=cfg["training"].get("fp16", True))

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cuda", weights_only=False)
        base_model = model.module if hasattr(model, "module") else model
        base_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("val_loss", float("inf"))
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}, step {global_step}")

    if rank == 0:
        print(f"Training for {stage_cfg['epochs']} epochs, {total_steps} steps")
        print(f"Batch size: {stage_cfg['batch_size']}, LR: {stage_cfg['lr']}")
        print("-" * 60)

    for epoch in range(start_epoch, stage_cfg["epochs"]):
        if distributed:
            train_sampler.set_epoch(epoch)

        cfg["training"]["_grad_clip"] = stage_cfg.get("grad_clip", 1.0)
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            epoch, cfg, writer, rank, global_step,
        )

        val_loss = float("inf")
        if val_loader and epoch % cfg["training"].get("val_interval", 1) == 0:
            val_loss = validate(model, val_loader, cfg, rank)
            if rank == 0 and writer:
                writer.add_scalar("val/loss", val_loss, global_step)

        if rank == 0:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}"
            )

            if epoch % cfg["training"].get("save_interval", 5) == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, val_loss,
                    os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"),
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, val_loss,
                    os.path.join(ckpt_dir, "best.pt"),
                )

    if rank == 0:
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            stage_cfg["epochs"] - 1, global_step, val_loss,
            os.path.join(ckpt_dir, "last.pt"),
        )
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
        if writer:
            writer.close()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
