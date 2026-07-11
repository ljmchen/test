"""Multi-task validation metrics for variable-hand grasp generation.

Implements the metric layer of the multi-task (left / right / lgbidex / bidex)
validation protocol:

  - side-based prediction<->GT hand alignment (within a sample the emitted hand
    sides are distinct, so no Hungarian matching is needed);
  - structure metrics (hand count / side set / full structure correctness),
    kept separate from pose errors so decision mistakes never pollute them;
  - physical pose errors (translation L2, rotation geodesic, joint L1) and
    bimanual relative-pose errors (rel-translation L2, rel-rotation geodesic);
  - a fixed-layout accumulator whose state is a single flat tensor so DDP
    aggregation is exactly one ``all_reduce`` (never a dict reduction, which
    breaks when ranks see different task-type subsets);
  - the combined validation score used for checkpoint selection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.distributed as dist

from utils.rotation import rotation_6d_to_matrix

TASK_TYPES: tuple[str, ...] = ("left", "right", "lgbidex", "bidex")
BIMANUAL: tuple[str, ...] = ("lgbidex", "bidex")

_TASK_TYPE_TO_ID: dict[str, int] = {t: i for i, t in enumerate(TASK_TYPES)}

_SCORE_WEIGHT_KEYS: tuple[str, ...] = (
    "translation", "rotation", "joint", "structure", "relative",
)


def task_type_index(
    task_types: Sequence[str], device: torch.device | str | None = None
) -> torch.Tensor:
    """Map task-type names to their fixed indices in ``TASK_TYPES`` order.

    Args:
        task_types: Task-type name per sample, length B.
        device: Optional device for the returned tensor.

    Returns:
        Long tensor of task-type ids, shape (B,).

    Raises:
        ValueError: If any name is not a member of ``TASK_TYPES``.
    """
    unknown = sorted({t for t in task_types if t not in _TASK_TYPE_TO_ID})
    if unknown:
        raise ValueError(
            f"Unknown task_type(s) {unknown}; expected one of {TASK_TYPES}."
        )
    ids = [_TASK_TYPE_TO_ID[t] for t in task_types]
    return torch.tensor(ids, dtype=torch.long, device=device)


def geodesic_angle_from_matrices(
    rot_a: torch.Tensor, rot_b: torch.Tensor
) -> torch.Tensor:
    """Geodesic angle (radians) between two rotation-matrix tensors.

    Args:
        rot_a: Rotation matrices, shape (..., 3, 3).
        rot_b: Rotation matrices, shape (..., 3, 3).

    Returns:
        Angles in radians, shape (...,).
    """
    rel = torch.matmul(rot_a.transpose(-1, -2), rot_b)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))


def pose_errors_physical(
    pred: torch.Tensor,
    gt: torch.Tensor,
    trans_dim: int = 3,
    rot_dim: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Physical per-hand pose errors between broadcastable pose tensors.

    Args:
        pred: Predicted poses ``[trans, rot6d, joints]``, shape (..., D).
        gt: Ground-truth poses, shape broadcastable against ``pred``.
        trans_dim: Translation dimensionality.
        rot_dim: Rotation (6D) dimensionality.

    Returns:
        trans: Translation L2 distance (meters), shape (...,).
        rot: Rotation geodesic angle (radians), shape (...,).
        joint: Mean absolute joint error (radians), shape (...,).
    """
    p_t, g_t = pred[..., :trans_dim], gt[..., :trans_dim]
    p_r = pred[..., trans_dim : trans_dim + rot_dim]
    g_r = gt[..., trans_dim : trans_dim + rot_dim]
    p_j, g_j = pred[..., trans_dim + rot_dim :], gt[..., trans_dim + rot_dim :]
    trans = torch.norm(p_t - g_t, dim=-1)
    rot = geodesic_angle_from_matrices(
        rotation_6d_to_matrix(p_r), rotation_6d_to_matrix(g_r)
    )
    joint = torch.abs(p_j - g_j).mean(-1)
    return trans, rot, joint


def _check_slot_shapes(mask: torch.Tensor, sides: torch.Tensor, name: str) -> None:
    if mask.shape != sides.shape or mask.dim() != 2:
        raise ValueError(
            f"{name}: mask/sides must both be (B, S); got "
            f"{tuple(mask.shape)} / {tuple(sides.shape)}."
        )


def _check_unique_sides(mask: torch.Tensor, sides: torch.Tensor, name: str) -> None:
    num_slots = mask.shape[1]
    for a in range(num_slots):
        for b in range(a + 1, num_slots):
            dup = mask[:, a] & mask[:, b] & (sides[:, a] == sides[:, b])
            if bool(dup.any()):
                rows = dup.nonzero(as_tuple=True)[0].tolist()
                raise ValueError(
                    f"{name}: duplicate hand side within valid slots at batch "
                    f"rows {rows[:8]}; sides must be distinct per sample."
                )


def align_hands_by_side(
    pred_poses: torch.Tensor,
    pred_mask: torch.Tensor,
    pred_sides: torch.Tensor,
    gt_poses: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_sides: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align predicted hands to GT hand slots by hand side.

    Args:
        pred_poses: Predicted poses, shape (B, S, D).
        pred_mask: Predicted slot validity, shape (B, S) bool.
        pred_sides: Predicted hand side ids (-1 = pad), shape (B, S) long.
        gt_poses: GT poses, shape (B, S, D).
        gt_mask: GT slot validity, shape (B, S) bool.
        gt_sides: GT hand side ids (-1 = pad), shape (B, S) long.

    Returns:
        aligned_pred: Predicted poses reordered into GT slot order, zeros on
            unmatched GT slots, shape (B, S, D).
        matched: Whether each GT slot has a same-side prediction, shape
            (B, S) bool.
        align_idx: Source prediction slot per GT slot (-1 = unmatched), shape
            (B, S) long.

    Raises:
        ValueError: On shape mismatch or duplicate sides among valid slots.
    """
    if pred_poses.shape != gt_poses.shape:
        raise ValueError(
            f"pred_poses {tuple(pred_poses.shape)} != gt_poses "
            f"{tuple(gt_poses.shape)}."
        )
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    _check_slot_shapes(pred_mask, pred_sides, "pred")
    _check_slot_shapes(gt_mask, gt_sides, "gt")
    _check_unique_sides(pred_mask, pred_sides, "pred")
    _check_unique_sides(gt_mask, gt_sides, "gt")

    same = (
        (gt_sides.unsqueeze(-1) == pred_sides.unsqueeze(1))
        & gt_mask.unsqueeze(-1)
        & pred_mask.unsqueeze(1)
    )
    matched = same.any(dim=-1)
    align_idx = same.long().argmax(dim=-1)
    aligned = torch.gather(
        pred_poses, 1, align_idx.unsqueeze(-1).expand(-1, -1, pred_poses.shape[-1])
    )
    aligned = aligned * matched.unsqueeze(-1).to(aligned.dtype)
    align_idx = align_idx.masked_fill(~matched, -1)
    return aligned, matched, align_idx


def structure_metrics(
    pred_mask: torch.Tensor,
    pred_sides: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_sides: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample structural correctness of the emitted hand set.

    Args:
        pred_mask: Predicted slot validity, shape (B, S) bool.
        pred_sides: Predicted hand side ids (-1 = pad), shape (B, S) long.
        gt_mask: GT slot validity, shape (B, S) bool.
        gt_sides: GT hand side ids (-1 = pad), shape (B, S) long.

    Returns:
        count_ok: Predicted hand count equals GT hand count, shape (B,) bool.
        side_ok: Emitted side set equals the GT side set, shape (B,) bool.
        struct_ok: Both count and side set correct, shape (B,) bool.
    """
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    _check_slot_shapes(pred_mask, pred_sides, "pred")
    _check_slot_shapes(gt_mask, gt_sides, "gt")

    count_ok = pred_mask.sum(dim=1) == gt_mask.sum(dim=1)
    same = (
        (gt_sides.unsqueeze(-1) == pred_sides.unsqueeze(1))
        & gt_mask.unsqueeze(-1)
        & pred_mask.unsqueeze(1)
    )
    gt_matched = same.any(dim=-1)
    pred_matched = same.any(dim=1)
    side_ok = (gt_matched == gt_mask).all(dim=-1) & (pred_matched == pred_mask).all(dim=-1)
    struct_ok = count_ok & side_ok
    return count_ok, side_ok, struct_ok


def relative_pose_error(
    pred_pair: torch.Tensor,
    gt_pair: torch.Tensor,
    trans_dim: int = 3,
    rot_dim: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bimanual relative-pose errors between prediction and GT hand pairs.

    Both inputs must be side-aligned (slot s of ``pred_pair`` is the same hand
    side as slot s of ``gt_pair``) with both hands valid, in physical units.

    Args:
        pred_pair: Predicted hand pair, shape (B, 2, D).
        gt_pair: GT hand pair, shape (B, 2, D).
        trans_dim: Translation dimensionality.
        rot_dim: Rotation (6D) dimensionality.

    Returns:
        rel_trans: L2 error of the hand-1-relative-to-hand-0 translation,
            shape (B,).
        rel_rot: Geodesic error of the relative rotation R0^T R1, shape (B,).
    """
    if pred_pair.dim() != 3 or pred_pair.shape[1] != 2 or pred_pair.shape != gt_pair.shape:
        raise ValueError(
            f"Expected matching (B, 2, D) pairs, got {tuple(pred_pair.shape)} / "
            f"{tuple(gt_pair.shape)}."
        )
    dp = pred_pair[:, 1, :trans_dim] - pred_pair[:, 0, :trans_dim]
    dg = gt_pair[:, 1, :trans_dim] - gt_pair[:, 0, :trans_dim]
    rel_trans = torch.norm(dp - dg, dim=-1)

    rot = slice(trans_dim, trans_dim + rot_dim)
    rp0 = rotation_6d_to_matrix(pred_pair[:, 0, rot])
    rp1 = rotation_6d_to_matrix(pred_pair[:, 1, rot])
    rg0 = rotation_6d_to_matrix(gt_pair[:, 0, rot])
    rg1 = rotation_6d_to_matrix(gt_pair[:, 1, rot])
    rel_rot = geodesic_angle_from_matrices(
        torch.matmul(rp0.transpose(-1, -2), rp1),
        torch.matmul(rg0.transpose(-1, -2), rg1),
    )
    return rel_trans, rel_rot


class MultiTaskValAccumulator:
    """Fixed-layout per-task-type metric accumulator for DDP validation.

    All state lives in one flat float64 tensor, so distributed aggregation is
    a single ``all_reduce`` regardless of which task types each rank saw.
    Fields (first dim indexes ``TASK_TYPES``):

        sum_err          (T, 3)  summed trans/rot/joint errors over matched hands
        cnt_hand         (T,)    matched hand count behind ``sum_err``
        cnt_pose_samp    (T,)    samples that entered pose accumulation
        cnt_struct_miss  (T,)    group-min: predictions without a same-signature
                                 candidate (excluded from pose means)
        cnt_samp         (T,)    free-rollout decision samples
        cnt_count_ok     (T,)    samples with correct hand count
        cnt_side_ok      (T,)    samples with correct side set
        cnt_struct_ok    (T,)    samples with fully correct structure
        cnt_hand_gt      (T,)    GT hand count over decision samples
        sum_rel          (T, 2)  summed bimanual rel_trans/rel_rot errors
        cnt_rel          (T,)    samples behind ``sum_rel``
    """

    _LAYOUT: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("sum_err", (len(TASK_TYPES), 3)),
        ("cnt_hand", (len(TASK_TYPES),)),
        ("cnt_pose_samp", (len(TASK_TYPES),)),
        ("cnt_struct_miss", (len(TASK_TYPES),)),
        ("cnt_samp", (len(TASK_TYPES),)),
        ("cnt_count_ok", (len(TASK_TYPES),)),
        ("cnt_side_ok", (len(TASK_TYPES),)),
        ("cnt_struct_ok", (len(TASK_TYPES),)),
        ("cnt_hand_gt", (len(TASK_TYPES),)),
        ("sum_rel", (len(TASK_TYPES), 2)),
        ("cnt_rel", (len(TASK_TYPES),)),
    )

    def __init__(self, device: torch.device | str = "cpu"):
        """
        Args:
            device: Device holding the accumulator (must support all_reduce
                when used with DDP, i.e. the CUDA device of the rank).
        """
        total = sum(math.prod(shape) for _, shape in self._LAYOUT)
        self._flat = torch.zeros(total, dtype=torch.float64, device=device)
        self._views: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape in self._LAYOUT:
            numel = math.prod(shape)
            self._views[name] = self._flat[offset : offset + numel].view(shape)
            offset += numel

    def add_decision(
        self,
        tt_ids: torch.Tensor,
        count_ok: torch.Tensor,
        side_ok: torch.Tensor,
        struct_ok: torch.Tensor,
        gt_hand_count: torch.Tensor,
    ) -> None:
        """Accumulate free-rollout decision metrics.

        Args:
            tt_ids: Task-type ids, shape (B,) long.
            count_ok: Hand-count correctness, shape (B,) bool.
            side_ok: Side-set correctness, shape (B,) bool.
            struct_ok: Full structure correctness, shape (B,) bool.
            gt_hand_count: GT hand count per sample, shape (B,).
        """
        ones = torch.ones(tt_ids.shape[0], dtype=torch.float64, device=self._flat.device)
        tt_ids = tt_ids.to(self._flat.device)
        v = self._views
        v["cnt_samp"].index_add_(0, tt_ids, ones)
        v["cnt_count_ok"].index_add_(0, tt_ids, count_ok.to(self._flat))
        v["cnt_side_ok"].index_add_(0, tt_ids, side_ok.to(self._flat))
        v["cnt_struct_ok"].index_add_(0, tt_ids, struct_ok.to(self._flat))
        v["cnt_hand_gt"].index_add_(0, tt_ids, gt_hand_count.to(self._flat))

    def add_pose_errors(
        self,
        tt_ids: torch.Tensor,
        trans: torch.Tensor,
        rot: torch.Tensor,
        joint: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Accumulate per-slot pose errors over the valid (matched) hands.

        Args:
            tt_ids: Task-type ids, shape (B,) long.
            trans: Translation errors, shape (B, S).
            rot: Rotation errors, shape (B, S).
            joint: Joint errors, shape (B, S).
            valid: Which slots contribute, shape (B, S) bool.
        """
        tt_ids = tt_ids.to(self._flat.device)
        valid_f = valid.to(self._flat)
        err = torch.stack(
            [
                (trans.to(self._flat) * valid_f).sum(dim=1),
                (rot.to(self._flat) * valid_f).sum(dim=1),
                (joint.to(self._flat) * valid_f).sum(dim=1),
            ],
            dim=-1,
        )
        ones = torch.ones(tt_ids.shape[0], dtype=torch.float64, device=self._flat.device)
        v = self._views
        v["sum_err"].index_add_(0, tt_ids, err)
        v["cnt_hand"].index_add_(0, tt_ids, valid_f.sum(dim=1))
        v["cnt_pose_samp"].index_add_(0, tt_ids, ones)

    def add_pose_sample(
        self,
        tt_id: int,
        trans: torch.Tensor,
        rot: torch.Tensor,
        joint: torch.Tensor,
    ) -> None:
        """Accumulate one sample's per-hand pose errors (group-min path).

        Args:
            tt_id: Task-type id of the sample.
            trans: Per-hand translation errors, shape (H,).
            rot: Per-hand rotation errors, shape (H,).
            joint: Per-hand joint errors, shape (H,).
        """
        v = self._views
        i = int(tt_id)
        v["sum_err"][i, 0] += float(trans.sum())
        v["sum_err"][i, 1] += float(rot.sum())
        v["sum_err"][i, 2] += float(joint.sum())
        v["cnt_hand"][i] += float(trans.numel())
        v["cnt_pose_samp"][i] += 1.0

    def add_relative(
        self, tt_ids: torch.Tensor, rel_trans: torch.Tensor, rel_rot: torch.Tensor
    ) -> None:
        """Accumulate bimanual relative-pose errors.

        Args:
            tt_ids: Task-type ids, shape (B,) long.
            rel_trans: Relative-translation errors, shape (B,).
            rel_rot: Relative-rotation errors, shape (B,).
        """
        tt_ids = tt_ids.to(self._flat.device)
        rel = torch.stack([rel_trans, rel_rot], dim=-1).to(self._flat)
        ones = torch.ones(tt_ids.shape[0], dtype=torch.float64, device=self._flat.device)
        self._views["sum_rel"].index_add_(0, tt_ids, rel)
        self._views["cnt_rel"].index_add_(0, tt_ids, ones)

    def add_structure_miss(self, tt_id: int) -> None:
        """Count a group-min prediction whose signature has no candidate.

        Args:
            tt_id: Task-type id of the sample.
        """
        self._views["cnt_struct_miss"][int(tt_id)] += 1.0

    def all_reduce(self, distributed: bool) -> None:
        """Sum the accumulator across ranks with a single all_reduce.

        Args:
            distributed: Whether DDP is active; no-op when False.
        """
        if distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(self._flat, op=dist.ReduceOp.SUM)

    def summary(self) -> dict:
        """Compute per-task-type means, macro averages, and bimanual rel means.

        Returns:
            Dict with:
                per_task: ``{tt: {trans, rot, joint, count_acc, side_acc,
                    structure_acc, rel_trans, rel_rot, n_hand, n_pose,
                    n_decision, n_rel, n_struct_miss, n_hand_gt}}`` (means are
                    NaN when their count is zero).
                macro: Macro averages over task types with data
                    (trans/rot/joint/count_acc/side_acc/structure_acc).
                bimanual: Macro rel_trans/rel_rot over ``BIMANUAL`` types.
        """
        v = {name: t.detach().cpu() for name, t in self._views.items()}

        def ratio(num: float, den: float) -> float:
            return float(num) / float(den) if float(den) > 0 else float("nan")

        per_task: dict[str, dict[str, float]] = {}
        for i, tt in enumerate(TASK_TYPES):
            per_task[tt] = {
                "trans": ratio(v["sum_err"][i, 0], v["cnt_hand"][i]),
                "rot": ratio(v["sum_err"][i, 1], v["cnt_hand"][i]),
                "joint": ratio(v["sum_err"][i, 2], v["cnt_hand"][i]),
                "count_acc": ratio(v["cnt_count_ok"][i], v["cnt_samp"][i]),
                "side_acc": ratio(v["cnt_side_ok"][i], v["cnt_samp"][i]),
                "structure_acc": ratio(v["cnt_struct_ok"][i], v["cnt_samp"][i]),
                "rel_trans": ratio(v["sum_rel"][i, 0], v["cnt_rel"][i]),
                "rel_rot": ratio(v["sum_rel"][i, 1], v["cnt_rel"][i]),
                "n_hand": int(v["cnt_hand"][i]),
                "n_pose": int(v["cnt_pose_samp"][i]),
                "n_decision": int(v["cnt_samp"][i]),
                "n_rel": int(v["cnt_rel"][i]),
                "n_struct_miss": int(v["cnt_struct_miss"][i]),
                "n_hand_gt": int(v["cnt_hand_gt"][i]),
            }

        def macro_mean(key: str, count_key: str, types: Sequence[str]) -> float:
            vals = [per_task[tt][key] for tt in types if per_task[tt][count_key] > 0]
            return sum(vals) / len(vals) if vals else float("nan")

        macro = {
            "trans": macro_mean("trans", "n_hand", TASK_TYPES),
            "rot": macro_mean("rot", "n_hand", TASK_TYPES),
            "joint": macro_mean("joint", "n_hand", TASK_TYPES),
            "count_acc": macro_mean("count_acc", "n_decision", TASK_TYPES),
            "side_acc": macro_mean("side_acc", "n_decision", TASK_TYPES),
            "structure_acc": macro_mean("structure_acc", "n_decision", TASK_TYPES),
        }
        bimanual = {
            "rel_trans": macro_mean("rel_trans", "n_rel", BIMANUAL),
            "rel_rot": macro_mean("rel_rot", "n_rel", BIMANUAL),
        }
        return {"per_task": per_task, "macro": macro, "bimanual": bimanual}


def compose_score(
    per_tt_errors: dict[str, dict[str, float]],
    structure_acc: dict[str, float],
    rel_errors: dict[str, dict[str, float]],
    weights: dict[str, float] | None,
    reduce: str = "macro",
) -> float:
    """Combine per-task pose errors, structure accuracy, and rel errors.

    ``score = macro_mean_tt(w_t*trans + w_r*rot + w_j*joint)
              + w_struct * (1 - macro structure_acc)
              + w_rel * macro_mean_bimanual(rel_trans + rel_rot)``

    The relative term is omitted when ``rel_errors`` is empty (no bimanual
    samples were evaluated). Lower is better.

    Args:
        per_tt_errors: ``{tt: {"trans", "rot", "joint"}}`` for task types
            with pose data.
        structure_acc: ``{tt: structure accuracy}`` for task types with
            decision data.
        rel_errors: ``{tt: {"rel_trans", "rel_rot"}}`` for bimanual task
            types with rel data.
        weights: Score weights (keys ``translation``/``rotation``/``joint``/
            ``structure``/``relative``); None keeps the defaults
            (3.0/6.0/1.0/1.0/1.0).
        reduce: Reduction over task types; only ``macro`` is supported.

    Returns:
        Scalar score (lower is better).

    Raises:
        ValueError: On unsupported ``reduce``, unknown weight keys, empty or
            non-finite inputs.
    """
    if reduce != "macro":
        raise ValueError(f"Unsupported val_score_reduce '{reduce}' (only 'macro').")
    if not per_tt_errors:
        raise ValueError("compose_score: per_tt_errors is empty (no pose data).")
    if not structure_acc:
        raise ValueError("compose_score: structure_acc is empty (no decision data).")

    w = {"translation": 3.0, "rotation": 6.0, "joint": 1.0, "structure": 1.0, "relative": 1.0}
    if weights:
        unknown = set(weights) - set(_SCORE_WEIGHT_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown val_score_weights keys {sorted(unknown)}; "
                f"expected {_SCORE_WEIGHT_KEYS}."
            )
        w.update({k: float(v) for k, v in weights.items()})

    values = [x for e in per_tt_errors.values() for x in e.values()]
    values += list(structure_acc.values())
    values += [x for e in rel_errors.values() for x in e.values()]
    if any(not math.isfinite(x) for x in values):
        raise ValueError("compose_score: non-finite metric value in inputs.")

    pose_terms = [
        w["translation"] * e["trans"] + w["rotation"] * e["rot"] + w["joint"] * e["joint"]
        for e in per_tt_errors.values()
    ]
    score = sum(pose_terms) / len(pose_terms)
    score += w["structure"] * (1.0 - sum(structure_acc.values()) / len(structure_acc))
    if rel_errors:
        rel_terms = [e["rel_trans"] + e["rel_rot"] for e in rel_errors.values()]
        score += w["relative"] * (sum(rel_terms) / len(rel_terms))
    return float(score)
