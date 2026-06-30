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

# ════════════════════════════════════════════════════════════════════════════
# train_v4.json measured statistics (826,325 grasps). 切换归一化参数时参考。
#
# ⚠️ v4 的抓取已经是物体中心化的（raw 平移 ≈ 0），不像 v3（世界坐标 ≈ obj_pose）。
#    所以 center_on_object=True 会把 v4 抓取再减一次 obj 平移(~-0.75 in x)、与点云错位。
#    v4 应让点云照常中心化、但抓取不再 shift；下面的数字就是 v4 抓取的真实(已中心化)分布。
#
# 中心化平移 1%/99% 分位（可直接替换下面 minmax 表的前 3 行 translation [min,max]）：
#   LEFT : x [-0.1471, 0.1511]   y [-0.1626, 0.1444]   z [-0.2033, 0.1828]
#   RIGHT: x [-0.1486, 0.1420]   y [-0.1591, 0.1831]   z [-0.1374, 0.1583]
#
# 中心化全 28 维 [trans(3), axis_angle(3), joints(22)] 均值 / 标准差
#   （meastd 取 dims 0-2=平移、6-27=关节；axis_angle(3-5) 跳过，因 6D 旋转直通）：
#   LEFT  mean = [ 0.0017,-0.0228,-0.0222, -0.1749,-0.0935,-0.0583, -0.0772, 0.2850, 0.3136,
#                  0.3147,-0.0199, 0.3897, 0.3230, 0.3235, 0.0041, 0.5409, 0.2909, 0.2903,
#                  0.2656,-0.0770, 0.4679, 0.2457, 0.2474, 0.3107, 0.9876, 0.0035, 0.0354, 0.0866]
#   LEFT  std  = [ 0.0842, 0.0896, 0.1037, 1.3569, 1.1725, 1.4571, 0.1404, 0.1706, 0.1428,
#                  0.1428, 0.1253, 0.1455, 0.1214, 0.1193, 0.1336, 0.1438, 0.0960, 0.0940,
#                  0.1093, 0.1402, 0.1171, 0.0878, 0.0872, 0.1877, 0.0942, 0.0879, 0.1009, 0.0933]
#   RIGHT mean = [ 0.0009, 0.0274,-0.0046, 0.5724,-0.6281, 0.1485, 0.1356, 0.5301, 0.3443,
#                  0.4587, 0.2239, 0.6730, 0.4273, 0.4470,-0.2317, 0.7304, 0.5002, 0.5549,
#                  0.1095,-0.2583, 0.8344, 0.4587, 0.5240, 0.3923, 0.6693, 0.1189,-0.0502, 0.1183]
#   RIGHT std  = [ 0.0833, 0.1079, 0.0681, 0.8953, 0.9212, 1.3979, 0.1687, 0.3412, 0.3086,
#                  0.3113, 0.1481, 0.3383, 0.2927, 0.3146, 0.1426, 0.3119, 0.3206, 0.3318,
#                  0.1320, 0.1316, 0.3102, 0.3078, 0.3187, 0.2619, 0.2623, 0.1031, 0.2292, 0.2137]
#   （完整数值见 tools/pose_stats_v4.json，由 tools/compute_pose_stats.py 生成。）
# ════════════════════════════════════════════════════════════════════════════

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

    def __init__(
        self,
        hand_side: str = "right",
        joint_dim: int = 22,
        mode: str = "minmax11",
        meastd: dict | None = None,
    ):
        if hand_side not in {"left", "right"}:
            raise ValueError(f"Unsupported hand_side '{hand_side}'.")
        if mode not in {"minmax11", "meastd11"}:
            raise ValueError(f"Unsupported mode '{mode}' (use minmax11 | meastd11).")
        self.hand_side = hand_side
        self.joint_dim = int(joint_dim)
        self.rotation_dim = 6
        self.single_hand_pose_dim = 3 + self.rotation_dim + self.joint_dim
        self.mode = mode

        if mode == "minmax11":
            factors = _LEFT_FACTOR_MINMAX if hand_side == "left" else _RIGHT_FACTOR_MINMAX
            self.trans_lo, self.trans_hi = factors[:3, 0], factors[:3, 1]
            self.joint_lo = factors[6 : 6 + self.joint_dim, 0]
            self.joint_hi = factors[6 : 6 + self.joint_dim, 1]
        else:  # meastd11: standardize translation+joints by (x-mean)/(2*std)
            if meastd is None:
                raise ValueError(
                    "meastd11 needs per-hand mean/std factors. Run "
                    "tools/compute_norm_stats.py and point "
                    "data.normalization.stats_path at the output JSON."
                )
            self.trans_mean = torch.as_tensor(meastd["translation_mean"], dtype=torch.float32)
            self.trans_std = torch.as_tensor(meastd["translation_std"], dtype=torch.float32)
            self.joint_mean = torch.as_tensor(meastd["joint_mean"], dtype=torch.float32)
            self.joint_std = torch.as_tensor(meastd["joint_std"], dtype=torch.float32)
            if self.joint_mean.numel() != self.joint_dim:
                raise ValueError(
                    f"meastd joint factors have {self.joint_mean.numel()} dims, "
                    f"expected joint_dim={self.joint_dim}."
                )

    def _split(self, pose: torch.Tensor):
        if pose.shape[-1] != self.single_hand_pose_dim:
            raise ValueError(
                f"Expected pose dim {self.single_hand_pose_dim}, got {pose.shape[-1]}."
            )
        return pose[..., :3], pose[..., 3:9], pose[..., 9:]

    def normalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        # Only translation + joints are scaled. The 6D rotation is passed through:
        # training targets are already orthonormal (see _to_target_pose) and model
        # outputs get re-orthonormalized downstream by rotation_6d_to_matrix.
        trans, rot6d, joints = self._split(pose)
        d, t = pose.device, pose.dtype
        if self.mode == "minmax11":
            tlo, thi = self.trans_lo.to(d, t), self.trans_hi.to(d, t)
            jlo, jhi = self.joint_lo.to(d, t), self.joint_hi.to(d, t)
            trans = 2.0 * (trans - tlo) / (thi - tlo + 1e-8) - 1.0
            joints = 2.0 * (joints - jlo) / (jhi - jlo + 1e-8) - 1.0
        else:  # meastd11
            trans = (trans - self.trans_mean.to(d, t)) / (2.0 * self.trans_std.to(d, t) + 1e-8)
            joints = (joints - self.joint_mean.to(d, t)) / (2.0 * self.joint_std.to(d, t) + 1e-8)
        return torch.cat([trans, rot6d, joints], dim=-1)

    def denormalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        trans, rot6d, joints = self._split(pose)
        d, t = pose.device, pose.dtype
        if self.mode == "minmax11":
            tlo, thi = self.trans_lo.to(d, t), self.trans_hi.to(d, t)
            jlo, jhi = self.joint_lo.to(d, t), self.joint_hi.to(d, t)
            trans = 0.5 * (trans + 1.0) * (thi - tlo) + tlo
            joints = 0.5 * (joints + 1.0) * (jhi - jlo) + jlo
        else:  # meastd11
            trans = trans * (2.0 * self.trans_std.to(d, t)) + self.trans_mean.to(d, t)
            joints = joints * (2.0 * self.joint_std.to(d, t)) + self.joint_mean.to(d, t)
        return torch.cat([trans, rot6d, joints], dim=-1)


def _load_meastd_factors(stats_path: str) -> dict:
    """Load per-hand mean/std factors (produced by tools/compute_norm_stats.py)."""
    from pathlib import Path

    path = Path(stats_path).expanduser()
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        if (repo_root / path).exists():
            path = repo_root / path
    if not path.exists():
        raise FileNotFoundError(
            f"meastd stats file not found: {path}. Run tools/compute_norm_stats.py first."
        )
    import json

    with path.open("r") as f:
        return json.load(f)


def build_hand_normalizers(
    normalization: dict | None, joint_dim: int = 22
) -> dict[str, LgbidexPoseNormalizer] | None:
    """Construct left/right normalizers from a config dict, or None if disabled.

    ``normalization`` examples::

        {"enabled": true, "mode": "minmax11"}
        {"enabled": true, "mode": "meastd11", "stats_path": "tools/meastd_factors.json"}
    """
    cfg = dict(normalization or {})
    if not cfg.get("enabled", False):
        return None
    mode = str(cfg.get("mode", "minmax11"))
    meastd_by_side = {"left": None, "right": None}
    if mode == "meastd11":
        stats = _load_meastd_factors(str(cfg.get("stats_path", "tools/meastd_factors.json")))
        meastd_by_side = {"left": stats["left"], "right": stats["right"]}
    return {
        side: LgbidexPoseNormalizer(
            hand_side=side, joint_dim=joint_dim, mode=mode, meastd=meastd_by_side[side]
        )
        for side in ("left", "right")
    }
