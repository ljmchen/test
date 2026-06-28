"""Per-hand min-max pose normalizer (ported from the dexvlm LGBiDex pipeline).

Normalizes a single-hand 31-d pose ``[translation(3), rot6d(6), joints(22)]``:

  - **translation** and **joints** are min-max scaled to ``[-1, 1]`` using
    hand-specific (left/right) factor tables, so the flow-matching targets match
    the standard-normal prior far better than raw meters/radians;
  - the **6D rotation** is only re-orthonormalized (geodesically unchanged), so
    rotation metrics are identical in normalized and raw space.

Factor tables are copied verbatim from dexvlm's ``LGBidexPoseNormalizer``
(``minmax11``, hand-specific) so both reproductions share the same data scaling.
Translation rows are hand-tuned; joint rows are the Shadow Hand ROM bounds.
"""

from __future__ import annotations

import torch

# 28 rows = translation(3) + axis-angle(3, unused for 6d) + joints(22), each [min, max].
_LEFT_FACTOR_MINMAX = torch.tensor(
    [
        [-0.28, 0.27], [-0.3, 0.34], [-0.13, 0.28],
        [-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14],
        [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [0.0, 0.785], [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [-1.047, 1.047], [0.0, 1.222], [-0.209, 0.209], [-0.524, 0.524], [-1.571, 0.0],
    ],
    dtype=torch.float32,
)
_RIGHT_FACTOR_MINMAX = torch.tensor(
    [
        [-0.27, 0.24], [-0.3, 0.35], [-0.29, 0.22],
        [-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14],
        [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [0.0, 0.785], [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [-1.047, 1.047], [0.0, 1.222], [-0.209, 0.209], [-0.524, 0.524], [-1.571, 0.0],
    ],
    dtype=torch.float32,
)


class LgbidexPoseNormalizer:
    """Min-max[-1,1] normalizer for one hand's 31-d pose."""

    def __init__(self, hand_side: str = "right", joint_dim: int = 22, mode: str = "minmax11"):
        if hand_side not in {"left", "right"}:
            raise ValueError(f"Unsupported hand_side '{hand_side}'.")
        if mode != "minmax11":
            raise ValueError(f"Only 'minmax11' is supported, got '{mode}'.")
        self.hand_side = hand_side
        self.joint_dim = int(joint_dim)
        self.rotation_dim = 6
        self.single_hand_pose_dim = 3 + self.rotation_dim + self.joint_dim
        factors = _LEFT_FACTOR_MINMAX if hand_side == "left" else _RIGHT_FACTOR_MINMAX
        self.trans_min, self.trans_max = factors[:3, 0], factors[:3, 1]
        self.joint_min = factors[6 : 6 + self.joint_dim, 0]
        self.joint_max = factors[6 : 6 + self.joint_dim, 1]

    def _split(self, pose: torch.Tensor):
        if pose.shape[-1] != self.single_hand_pose_dim:
            raise ValueError(
                f"Expected pose dim {self.single_hand_pose_dim}, got {pose.shape[-1]}."
            )
        return pose[..., :3], pose[..., 3:9], pose[..., 9:]

    def _factors(self, device, dtype):
        return (
            self.trans_min.to(device=device, dtype=dtype),
            self.trans_max.to(device=device, dtype=dtype),
            self.joint_min.to(device=device, dtype=dtype),
            self.joint_max.to(device=device, dtype=dtype),
        )

    def normalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        # Only translation + joints are min-max scaled. The 6D rotation is passed
        # through: training targets are already orthonormal (see _to_target_pose)
        # and model outputs get re-orthonormalized downstream by rotation_6d_to_matrix,
        # so the explicit round-trip here is redundant (and was the hot cost).
        trans, rot6d, joints = self._split(pose)
        tmin, tmax, jmin, jmax = self._factors(pose.device, pose.dtype)
        trans = 2.0 * (trans - tmin) / (tmax - tmin + 1e-8) - 1.0
        joints = 2.0 * (joints - jmin) / (jmax - jmin + 1e-8) - 1.0
        return torch.cat([trans, rot6d, joints], dim=-1)

    def denormalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        trans, rot6d, joints = self._split(pose)
        tmin, tmax, jmin, jmax = self._factors(pose.device, pose.dtype)
        trans = 0.5 * (trans + 1.0) * (tmax - tmin) + tmin
        joints = 0.5 * (joints + 1.0) * (jmax - jmin) + jmin
        return torch.cat([trans, rot6d, joints], dim=-1)


def build_hand_normalizers(
    normalization: dict | None, joint_dim: int = 22
) -> dict[str, LgbidexPoseNormalizer] | None:
    """Construct left/right normalizers from a config dict, or None if disabled.

    ``normalization`` example: ``{"enabled": true, "mode": "minmax11"}``.
    """
    cfg = dict(normalization or {})
    if not cfg.get("enabled", False):
        return None
    mode = str(cfg.get("mode", "minmax11"))
    return {
        side: LgbidexPoseNormalizer(hand_side=side, joint_dim=joint_dim, mode=mode)
        for side in ("left", "right")
    }
