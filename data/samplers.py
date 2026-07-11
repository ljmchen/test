"""Per-epoch grouped resampling for balanced training/validation.

``GroupedEpochSampler`` groups records by a combination of keys (e.g.
``obj_id + pose_id + guidance``) and, each epoch, randomly draws up to
``samples_per_group`` record indices from every group. This balances
combinations that have many grasp candidates against those with few, and
exposes the model to a fresh subset of each group's grasps every epoch.

With ``scenario_key`` set (e.g. ``task_type``), the per-group draw runs
independently inside each scenario and the per-scenario pools are then
up-/down-sampled to fixed quotas ``round(epoch_size * share)`` so every epoch
has a stable scenario composition.

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
        scenario_key: str | None = None,
        scenario_shares: dict[str, float] | None = None,
        epoch_size: int | None = None,
    ) -> None:
        """
        Args:
            records: Split records (list order defines the sampled indices).
            group_keys: Record keys whose combined value defines a group.
            samples_per_group: Max indices drawn per group per epoch.
            shuffle: Shuffle the epoch index list.
            num_replicas: DDP world size.
            rank: DDP rank.
            base_seed: Epoch seed base (``base_seed + epoch``).
            drop_last: Round the per-rank count down (True) or up (False).
            scenario_key: Record key partitioning records into scenarios
                (e.g. ``task_type``); None keeps the original single-pool
                behavior.
            scenario_shares: Per-scenario share of the epoch, auto-normalized;
                None means equal shares. Only valid with ``scenario_key``.
            epoch_size: Total pre-shard epoch length; None uses the sum of the
                scenarios' deterministic per-group draw counts.
        """
        if samples_per_group < 1:
            raise ValueError("samples_per_group must be >= 1.")
        if num_replicas < 1 or not (0 <= rank < num_replicas):
            raise ValueError(f"Invalid rank/num_replicas: {rank}/{num_replicas}.")
        if scenario_key is None and (scenario_shares is not None or epoch_size is not None):
            raise ValueError("scenario_shares/epoch_size require scenario_key.")
        self.samples_per_group = int(samples_per_group)
        self.shuffle = bool(shuffle)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.base_seed = int(base_seed)
        self.drop_last = bool(drop_last)
        self.scenario_key = scenario_key
        self.epoch = 0

        if scenario_key is None:
            groups: dict[tuple, list[int]] = {}
            for idx, rec in enumerate(records):
                groups.setdefault(combo_key(rec, group_keys), []).append(idx)
            self.groups: list[list[int]] = list(groups.values())

            # Per-epoch index count is deterministic (does not depend on the draw).
            total = sum(min(self.samples_per_group, len(g)) for g in self.groups)
            self.num_groups = len(self.groups)
        else:
            scenario_groups: dict[str, dict[tuple, list[int]]] = {}
            for idx, rec in enumerate(records):
                scenario = combo_value(rec, scenario_key)
                scenario_groups.setdefault(scenario, {}).setdefault(
                    combo_key(rec, group_keys), []
                ).append(idx)
            self.scenario_groups: dict[str, list[list[int]]] = {
                s: list(g.values()) for s, g in sorted(scenario_groups.items())
            }
            base_quota = {
                s: sum(min(self.samples_per_group, len(g)) for g in groups_s)
                for s, groups_s in self.scenario_groups.items()
            }
            if not base_quota:
                raise ValueError("No records to sample from.")

            if scenario_shares is None:
                shares = {s: 1.0 for s in self.scenario_groups}
            else:
                unknown = set(scenario_shares) - set(self.scenario_groups)
                if unknown:
                    raise ValueError(
                        f"scenario_shares has scenarios absent from the data: {sorted(unknown)} "
                        f"(observed: {sorted(self.scenario_groups)})."
                    )
                missing = set(self.scenario_groups) - set(scenario_shares)
                if missing:
                    raise ValueError(
                        f"scenario_shares is missing observed scenarios: {sorted(missing)}."
                    )
                shares = {s: float(v) for s, v in scenario_shares.items()}
                if any(v <= 0 for v in shares.values()):
                    raise ValueError(f"scenario_shares must be positive: {shares}.")
            share_sum = sum(shares.values())
            shares = {s: v / share_sum for s, v in shares.items()}

            total = int(epoch_size) if epoch_size is not None else sum(base_quota.values())
            if total < 1:
                raise ValueError(f"epoch_size must be >= 1, got {total}.")
            scenario_names = list(self.scenario_groups)
            quota = {s: int(round(total * shares[s])) for s in scenario_names}
            quota[scenario_names[-1]] += total - sum(quota.values())
            if any(q < 1 for q in quota.values()):
                raise ValueError(f"Scenario quota underflow: {quota} (epoch_size={total}).")
            self.scenario_quota: dict[str, int] = quota
            self.num_groups = sum(len(g) for g in self.scenario_groups.values())

        if self.drop_last:
            self.num_samples = total // self.num_replicas
        else:
            self.num_samples = math.ceil(total / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _draw_groups(
        groups: list[list[int]], samples_per_group: int, generator: torch.Generator
    ) -> list[int]:
        indices: list[int] = []
        for group in groups:
            m = min(samples_per_group, len(group))
            if len(group) <= m:
                picked = list(group)
            else:
                perm = torch.randperm(len(group), generator=generator).tolist()
                picked = [group[i] for i in perm[:m]]
            indices.extend(picked)
        return indices

    @staticmethod
    def _resample(pool: list[int], quota: int, generator: torch.Generator) -> list[int]:
        if not pool:
            raise ValueError("Cannot resample an empty scenario pool.")
        if len(pool) == quota:
            return list(pool)
        if len(pool) > quota:
            perm = torch.randperm(len(pool), generator=generator).tolist()
            return [pool[i] for i in perm[:quota]]
        repeated = pool * (quota // len(pool))
        remainder = quota - len(repeated)
        perm = torch.randperm(len(pool), generator=generator).tolist()
        return repeated + [pool[i] for i in perm[:remainder]]

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + self.epoch)

        if self.scenario_key is None:
            indices = self._draw_groups(self.groups, self.samples_per_group, generator)
        else:
            indices = []
            for scenario, groups_s in self.scenario_groups.items():
                pool = self._draw_groups(groups_s, self.samples_per_group, generator)
                indices.extend(self._resample(pool, self.scenario_quota[scenario], generator))

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
