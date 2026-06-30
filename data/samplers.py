"""Per-epoch grouped resampling for balanced training/validation.

``GroupedEpochSampler`` groups records by a combination of keys (e.g.
``obj_id + pose_id + guidance``) and, each epoch, randomly draws up to
``samples_per_group`` record indices from every group. This balances
combinations that have many grasp candidates against those with few, and
exposes the model to a fresh subset of each group's grasps every epoch.

DDP-aware: with ``num_replicas`` > 1 the per-epoch index list is built
identically on every rank (seeded by ``base_seed + epoch``) and then sharded
evenly, mirroring ``DistributedSampler`` semantics.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


def combo_value(record: dict, key: str) -> str:
    """Fetch a group-key value, tolerating the guidance/guidence spelling."""
    if key in ("guidance", "guidence"):
        return str(record.get("guidance", record.get("guidence", "")))
    return str(record.get(key, ""))


def combo_key(record: dict, keys: Sequence[str]) -> tuple:
    return tuple(combo_value(record, k) for k in keys)


class GroupedEpochSampler(Sampler[int]):
    """Sample up to ``samples_per_group`` indices per group, re-drawn each epoch."""

    def __init__(
        self,
        records: Sequence[dict],
        group_keys: Sequence[str],
        samples_per_group: int,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        base_seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if samples_per_group < 1:
            raise ValueError("samples_per_group must be >= 1.")
        if num_replicas < 1 or not (0 <= rank < num_replicas):
            raise ValueError(f"Invalid rank/num_replicas: {rank}/{num_replicas}.")
        self.samples_per_group = int(samples_per_group)
        self.shuffle = bool(shuffle)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.base_seed = int(base_seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        groups: dict[tuple, list[int]] = {}
        for idx, rec in enumerate(records):
            groups.setdefault(combo_key(rec, group_keys), []).append(idx)
        self.groups: list[list[int]] = list(groups.values())

        # Per-epoch index count is deterministic (does not depend on the draw).
        total = sum(min(self.samples_per_group, len(g)) for g in self.groups)
        if self.drop_last:
            self.num_samples = total // self.num_replicas
        else:
            self.num_samples = math.ceil(total / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.num_groups = len(self.groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + self.epoch)

        indices: list[int] = []
        for group in self.groups:
            m = min(self.samples_per_group, len(group))
            if len(group) <= m:
                picked = list(group)
            else:
                perm = torch.randperm(len(group), generator=generator).tolist()
                picked = [group[i] for i in perm[:m]]
            indices.extend(picked)

        if self.shuffle:
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in order]

        # Even sharding across ranks (pad or trim to total_size).
        if len(indices) < self.total_size:
            if indices:
                indices = (indices * (self.total_size // len(indices) + 1))[: self.total_size]
        else:
            indices = indices[: self.total_size]
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
