"""Dataset for LGBiDex bimanual dexterous grasping with language instructions.

This adapts the LGBiDex data format (as consumed by the dexvlm pipeline's
``LgbiDexDataset``) to the test model's input contract. Input processing
mirrors dexvlm exactly so the two reproductions share coordinate conventions:

  - object placement comes from a 7-dim ``obj_pose`` (translation + wxyz quat)
    plus ``obj_scale``;
  - point clouds load from colored ``pc_part/simplified_part_{N}.ply`` and fall
    back to surface-sampling ``mesh/simplified.obj``;
  - the point cloud is bbox-centered and placed by the object rotation/scale so
    the object sits at the origin (``center_on_object``);
  - each hand's 28-dim grasp ``[translation(3), axis_angle(3), joints(22)]`` is
    shifted into the same object-centered frame and the axis-angle rotation is
    converted to the 6D representation the model consumes.

Each record (JSON list element) provides::

    {
        "obj_id": str,
        "obj_pose": [tx, ty, tz, qw, qx, qy, qz],
        "obj_scale": float,
        "cate_id": str,
        "guidance": str,
        "dex_grasp_left":  [tx, ty, tz, ax, ay, az, j0..j21],   # 28-dim
        "dex_grasp_right": [tx, ty, tz, ax, ay, az, j0..j21],   # 28-dim
        ...
    }

Output per item matches the model contract::

    xyz(N, 3), rgb(N, 3), text(str), pose_left(31), pose_right(31)
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.lgbidex_io import (
    axis_angle_to_matrix,
    canonicalize_axis_angle,
    load_mesh_geometry,
    load_mesh_vertices,
    load_point_cloud_from_ply,
    make_constant_color,
    matrix_to_rotation_6d,
    normalize_point_cloud_color_mode,
    normalize_rotation_6d,
    quaternion_to_matrix,
    sample_uniform_points_from_mesh,
    select_point_cloud_colors,
    shift_pose_translation,
    transform_points,
)
from data.pose_normalizer import build_hand_normalizers

HAND_SIDES = ("left", "right")
SINGLE_HAND_RAW_POSE_DIM = 28  # [trans(3), axis_angle(3), joints(22)]


def sample_point_cloud_from_mesh(
    mesh_path: str,
    n_points: int = 4096,
    scale: float = 0.1,
    color_fill: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a bbox-centered, scaled colored point cloud from a single mesh.

    Convenience helper for single-object inference. Mirrors the dataset's
    object-centered convention (bbox center at origin, scaled by ``scale``);
    no object rotation/translation is available for a bare mesh, so the
    canonical mesh frame is used. Colors are a constant fill.

    Returns:
        xyz: ``(n_points, 3)`` float32 positions.
        rgb: ``(n_points, 3)`` float32 colors in [0, 1].
    """
    vertices, faces = load_mesh_geometry(Path(mesh_path))
    bbox_center = 0.5 * (vertices.min(dim=0).values + vertices.max(dim=0).values)
    xyz = sample_uniform_points_from_mesh(vertices, faces, int(n_points))
    xyz = (xyz - bbox_center) * float(scale)
    rgb = make_constant_color(int(n_points), xyz.dtype, color_fill)
    return xyz.numpy().astype(np.float32), rgb.numpy().astype(np.float32)


class DexGraspDataset(Dataset):
    """LGBiDex bimanual dexterous grasping dataset.

    Each sample provides a colored point cloud of the object, a language
    instruction, and bimanual grasp poses (left + right) in object-centered
    coordinates, with rotation expressed in the 6D representation.
    """

    def __init__(
        self,
        data_path: str,
        mesh_root: str,
        n_points: int = 4096,
        joint_dim: int = 22,
        augment: bool = False,
        point_cloud_source: str = "auto",
        point_cloud_color_mode: str = "real",
        point_cloud_color_fill: float = 0.4,
        point_cloud_file_points: int | str | None = None,
        center_on_object: bool = True,
        obj_pose_quaternion_order: str = "wxyz",
        normalization: dict | None = None,
    ):
        """
        Args:
            data_path: Path to the LGBiDex JSON split file.
            mesh_root: Root directory of processed objects. Per object:
                ``{mesh_root}/{obj_id}/pc_part/simplified_part_{N}.ply`` and
                ``{mesh_root}/{obj_id}/mesh/simplified.obj``.
            n_points: Number of points per object.
            joint_dim: Joint angles per hand (22 for the Shadow Hand).
            augment: Apply (pose-preserving) color/position jitter.
            point_cloud_source: ``auto`` | ``colored_ply`` | ``mesh_sample``.
            point_cloud_color_mode: ``real`` (use PLY RGB) | ``fill`` (constant).
            point_cloud_color_fill: Constant color value when filling.
            point_cloud_file_points: PLY point count tag (defaults to n_points).
            center_on_object: Center point cloud + grasps on the object.
            obj_pose_quaternion_order: ``wxyz`` | ``xyzw``.
            normalization: ``{"enabled": bool, "mode": "minmax11"}``. When
                enabled, target poses are min-max normalized to ``[-1, 1]``
                per hand (translation + joints); the flow model then trains on
                normalized poses and callers must denormalize predictions.
        """
        if point_cloud_source not in {"auto", "colored_ply", "mesh_sample"}:
            raise ValueError(f"Unsupported point_cloud_source '{point_cloud_source}'.")
        if obj_pose_quaternion_order not in {"wxyz", "xyzw"}:
            raise ValueError(f"Unsupported quaternion order '{obj_pose_quaternion_order}'.")

        with open(data_path, "r") as f:
            self.data = json.load(f)
        if not isinstance(self.data, list):
            raise ValueError(f"LGBiDex split at {data_path} must be a JSON array.")

        self.mesh_root = Path(mesh_root).expanduser().resolve()
        self.n_points = int(n_points)
        self.joint_dim = int(joint_dim)
        self.augment = augment
        self.point_cloud_source = point_cloud_source
        self.point_cloud_color_mode = normalize_point_cloud_color_mode(point_cloud_color_mode)
        self.point_cloud_color_fill = float(point_cloud_color_fill)
        self.point_cloud_file_points = str(self.n_points if point_cloud_file_points is None else point_cloud_file_points)
        self.center_on_object = bool(center_on_object)
        self.obj_pose_quaternion_order = obj_pose_quaternion_order
        # per-hand pose normalizers (None when normalization disabled)
        self.normalizers = build_hand_normalizers(normalization, joint_dim=self.joint_dim)

        # Cache the raw (pre-transform) per-object geometry so the cache size is
        # bounded by the number of unique objects, not the number of records.
        self._raw_pc_cache: dict[str, tuple[torch.Tensor, bool]] = {}
        self._bbox_center_cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.data)

    # ─── Path resolution ──────────────────────────────────────────────────

    def _resolve_point_cloud_path(self, obj_id: str) -> Path:
        path = self.mesh_root / obj_id / "pc_part" / f"simplified_part_{self.point_cloud_file_points}.ply"
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(
            f"Unable to find point cloud '{path.name}' for '{obj_id}' under {path.parent}."
        )

    def _resolve_mesh_path(self, obj_id: str) -> Path:
        path = self.mesh_root / obj_id / "mesh" / "simplified.obj"
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"Unable to find simplified.obj for '{obj_id}' under {path.parent}.")

    # ─── Object placement ─────────────────────────────────────────────────

    def _build_object_pose(self, record: dict) -> tuple[torch.Tensor, torch.Tensor, float]:
        obj_pose = torch.tensor(record["obj_pose"], dtype=torch.float32)
        if obj_pose.shape[-1] != 7:
            raise ValueError(f"Expected obj_pose dim 7, got {obj_pose.shape[-1]}.")
        translation = obj_pose[:3]
        rotation_matrix = quaternion_to_matrix(obj_pose[3:], order=self.obj_pose_quaternion_order)
        scale = float(record.get("obj_scale", 0.1))
        return translation, rotation_matrix, scale

    def _load_mesh_bbox_center(self, obj_id: str) -> torch.Tensor:
        if obj_id not in self._bbox_center_cache:
            vertices = load_mesh_vertices(self._resolve_mesh_path(obj_id))
            self._bbox_center_cache[obj_id] = 0.5 * (vertices.min(dim=0).values + vertices.max(dim=0).values)
        return self._bbox_center_cache[obj_id]

    def _build_object_center_offset(self, record: dict) -> torch.Tensor:
        bbox_center = self._load_mesh_bbox_center(str(record["obj_id"]))
        _, rotation, scale = self._build_object_pose(record)
        return transform_points(bbox_center, rotation_matrix=rotation, scale=scale)

    # ─── Point cloud ──────────────────────────────────────────────────────

    def _load_raw_point_cloud(self, obj_id: str) -> tuple[torch.Tensor, bool]:
        """Load the raw, untransformed point cloud (xyz[+rgb]) for an object."""
        if obj_id in self._raw_pc_cache:
            return self._raw_pc_cache[obj_id]

        if self.point_cloud_source == "mesh_sample":
            raw_pc, has_real_color = self._sample_point_cloud_from_mesh(obj_id), False
        elif self.point_cloud_source == "colored_ply":
            raw_pc, has_real_color = load_point_cloud_from_ply(
                self._resolve_point_cloud_path(obj_id), return_color_status=True
            )
            self._check_point_count(raw_pc, obj_id)
        else:  # auto: prefer colored PLY, fall back to mesh sampling
            try:
                raw_pc, has_real_color = load_point_cloud_from_ply(
                    self._resolve_point_cloud_path(obj_id), return_color_status=True
                )
                self._check_point_count(raw_pc, obj_id)
            except (FileNotFoundError, ValueError):
                raw_pc, has_real_color = self._sample_point_cloud_from_mesh(obj_id), False

        self._raw_pc_cache[obj_id] = (raw_pc, has_real_color)
        return raw_pc, has_real_color

    def _sample_point_cloud_from_mesh(self, obj_id: str) -> torch.Tensor:
        vertices, faces = load_mesh_geometry(self._resolve_mesh_path(obj_id))
        sampled_xyz = sample_uniform_points_from_mesh(vertices, faces, self.n_points)
        colors = make_constant_color(self.n_points, sampled_xyz.dtype, self.point_cloud_color_fill)
        return torch.cat([sampled_xyz, colors], dim=-1)

    def _check_point_count(self, raw_pc: torch.Tensor, obj_id: str) -> None:
        if raw_pc.shape[0] != self.n_points:
            raise ValueError(
                f"Point cloud for '{obj_id}' has {raw_pc.shape[0]} points, expected {self.n_points}."
            )

    def _build_point_cloud(self, record: dict) -> tuple[torch.Tensor, torch.Tensor]:
        obj_id = str(record["obj_id"])
        raw_pc, has_real_color = self._load_raw_point_cloud(obj_id)
        bbox_center = self._load_mesh_bbox_center(obj_id).to(dtype=raw_pc.dtype)
        translation, rotation, scale = self._build_object_pose(record)

        xyz = raw_pc[:, :3] - bbox_center
        xyz = transform_points(xyz, rotation_matrix=rotation, scale=scale)
        if not self.center_on_object:
            xyz = transform_points(xyz, translation=translation)

        rgb = select_point_cloud_colors(
            raw_pc,
            color_mode=self.point_cloud_color_mode,
            color_fill=self.point_cloud_color_fill,
            has_real_color=has_real_color,
        )
        return xyz.clone(), rgb.clone()

    # ─── Grasp poses ──────────────────────────────────────────────────────

    def _center_hand_pose(self, pose: torch.Tensor, record: dict) -> torch.Tensor:
        if pose.shape[-1] != SINGLE_HAND_RAW_POSE_DIM:
            raise ValueError(
                f"Expected hand pose dim {SINGLE_HAND_RAW_POSE_DIM}, got {pose.shape[-1]}."
            )
        if not self.center_on_object:
            return pose
        translation, _, _ = self._build_object_pose(record)
        center_offset = self._build_object_center_offset(record)
        return shift_pose_translation(pose, translation + center_offset)

    def _to_target_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """Convert a centered 28-dim grasp to ``[trans(3), rot6d(6), joints(J)]``."""
        translation = pose[:3]
        axis_angle = canonicalize_axis_angle(pose[3:6])
        joints = pose[6 : 6 + self.joint_dim]
        if joints.shape[-1] < self.joint_dim:
            joints = torch.nn.functional.pad(joints, (0, self.joint_dim - joints.shape[-1]))
        rotation = normalize_rotation_6d(matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle)))
        return torch.cat([translation, rotation, joints], dim=-1)

    def _build_hand_pose(self, record: dict, side: str) -> torch.Tensor:
        raw = torch.tensor(record[f"dex_grasp_{side}"], dtype=torch.float32)
        centered = self._center_hand_pose(raw, record)
        pose = self._to_target_pose(centered)
        if self.normalizers is not None:
            pose = self.normalizers[side].normalize_pose(pose)
        return pose

    # ─── Augmentation (pose-preserving) ───────────────────────────────────

    def _augment_point_cloud(
        self, xyz: torch.Tensor, rgb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Light jitter that does not break point-cloud/grasp correspondence.

        Note: no random rotation is applied here. Rotating only the point cloud
        (as the legacy version did) silently desynchronizes the grasp labels,
        so we keep augmentation to small per-point position/color noise.
        """
        xyz = xyz + torch.randn_like(xyz) * 0.005
        rgb = torch.clamp(rgb + torch.randn_like(rgb) * 0.02, 0.0, 1.0)
        return xyz, rgb

    def _build_instruction(self, record: dict) -> str:
        guidance = str(record.get("guidance", "")).strip()
        if not guidance:
            action = record.get("action", "grasp")
            cate = record.get("cate_id", "object")
            guidance = f"{action} the {cate}"
        return guidance

    def get_inference_item(self, idx: int) -> dict:
        """Per-record inputs + world-restore offsets for batched inference.

        Returns the point cloud, instruction, and the object translation /
        bbox-center offset needed to map object-centered predictions back to
        the world frame (mirrors dexvlm's ``get_visualization_item``).
        """
        record = self.data[idx]
        xyz, rgb = self._build_point_cloud(record)
        translation, _, _ = self._build_object_pose(record)
        center_offset = self._build_object_center_offset(record)
        return {
            "xyz": xyz,
            "rgb": rgb,
            "text": self._build_instruction(record),
            "object_translation": translation,
            "object_center_offset": center_offset,
            "record": record,
        }

    def __getitem__(self, idx: int) -> dict:
        record = self.data[idx]
        xyz, rgb = self._build_point_cloud(record)
        if self.augment:
            xyz, rgb = self._augment_point_cloud(xyz, rgb)

        pose_left = self._build_hand_pose(record, "left")
        pose_right = self._build_hand_pose(record, "right")

        guidance = self._build_instruction(record)

        return {
            "xyz": xyz,
            "rgb": rgb,
            "text": guidance,
            "pose_left": pose_left,
            "pose_right": pose_right,
            "obj_id": str(record.get("obj_id", "")),
            "cate_id": str(record.get("cate_id", "")),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that stacks tensors and keeps text fields as lists."""
    result = {}
    result["xyz"] = torch.stack([b["xyz"] for b in batch])
    result["rgb"] = torch.stack([b["rgb"] for b in batch])
    result["texts"] = [b["text"] for b in batch]
    pose_left = torch.stack([b["pose_left"] for b in batch])
    pose_right = torch.stack([b["pose_right"] for b in batch])
    result["gt_poses"] = torch.stack([pose_left, pose_right], dim=1)
    result["obj_ids"] = [b["obj_id"] for b in batch]
    result["cate_ids"] = [b["cate_id"] for b in batch]
    return result


def create_dataloader(
    data_path: str,
    mesh_root: str,
    batch_size: int = 32,
    n_points: int = 4096,
    joint_dim: int = 22,
    augment: bool = False,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    """Create a DataLoader for the LGBiDex dexterous grasp dataset."""
    dataset = DexGraspDataset(
        data_path=data_path,
        mesh_root=mesh_root,
        n_points=n_points,
        joint_dim=joint_dim,
        augment=augment,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
