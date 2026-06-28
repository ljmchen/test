"""Dataset for bimanual dexterous grasping with language instructions.

Loads object meshes, samples colored point clouds, and prepares
bimanual grasp pose labels in object-centered coordinates.

Expected data format (JSON list):
    {
        "obj_id": str,
        "obj_pose": [x, y, z, qw, qx, qy, qz],
        "obj_scale": float,
        "cate_id": str,
        "action": str,
        "contact_area": [...],
        "dex_grasp_left": [tx, ty, tz, qw, qx, qy, qz, j0, j1, ...],
        "dex_grasp_right": [tx, ty, tz, qw, qx, qy, qz, j0, j1, ...],
        "guidance": str
    }
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset, DataLoader

from utils.rotation import quaternion_to_rotation_6d


def farthest_point_sample_np(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Farthest point sampling on numpy arrays."""
    N = points.shape[0]
    if N <= n_samples:
        pad = np.random.choice(N, n_samples - N, replace=True)
        return np.concatenate([np.arange(N), pad])

    centroids = np.zeros(n_samples, dtype=np.int64)
    distance = np.full(N, np.inf)
    centroids[0] = np.random.randint(N)

    for i in range(1, n_samples):
        centroid = points[centroids[i - 1]]
        dist = np.sum((points - centroid) ** 2, axis=-1)
        distance = np.minimum(distance, dist)
        centroids[i] = np.argmax(distance)

    return centroids


def sample_point_cloud_from_mesh(
    mesh_path: str,
    n_points: int = 10000,
    scale: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a colored point cloud from an OBJ mesh.

    Args:
        mesh_path: Path to the mesh file.
        n_points: Number of points to sample.
        scale: Scale factor for the mesh.

    Returns:
        xyz: Point positions, shape (n_points, 3).
        rgb: Point colors normalized to [0, 1], shape (n_points, 3).
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(scale)

    points, face_indices = trimesh.sample.sample_surface(mesh, n_points * 2)

    if mesh.visual.kind == "face" or mesh.visual.kind == "vertex":
        try:
            colors = mesh.visual.interpolated_face_color(face_indices)[:, :3]
        except Exception:
            colors = np.full((len(points), 3), 128, dtype=np.uint8)
    else:
        colors = np.full((len(points), 3), 128, dtype=np.uint8)

    idx = farthest_point_sample_np(points, n_points)
    xyz = points[idx].astype(np.float32)
    rgb = colors[idx].astype(np.float32) / 255.0

    return xyz, rgb


def apply_pose_to_points(
    points: np.ndarray, position: np.ndarray, quaternion: np.ndarray
) -> np.ndarray:
    """Transform points by a pose (position + quaternion wxyz)."""
    w, x, y, z = quaternion
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return (rot @ points.T).T + position


class DexGraspDataset(Dataset):
    """Dataset for bimanual dexterous grasping.

    Each sample provides:
        - Colored point cloud of the object
        - Language instruction for grasping
        - Bimanual grasp poses (left + right) in object-centered coordinates
    """

    def __init__(
        self,
        data_path: str,
        mesh_root: str,
        n_points: int = 10000,
        joint_dim: int = 22,
        augment: bool = False,
    ):
        """
        Args:
            data_path: Path to the JSON data file.
            mesh_root: Root directory for object meshes.
                Meshes at: {mesh_root}/{obj_id}/mesh/simplified.obj
            n_points: Number of points to sample per object.
            joint_dim: Number of joint angles per hand.
            augment: Whether to apply data augmentation.
        """
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.mesh_root = mesh_root
        self.n_points = n_points
        self.joint_dim = joint_dim
        self.augment = augment
        self._mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.data)

    def _get_mesh_points(
        self, obj_id: str, scale: float
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_key = f"{obj_id}_{scale}"
        if cache_key not in self._mesh_cache:
            mesh_path = os.path.join(
                self.mesh_root, obj_id, "mesh", "simplified.obj"
            )
            xyz, rgb = sample_point_cloud_from_mesh(
                mesh_path, self.n_points, scale
            )
            self._mesh_cache[cache_key] = (xyz, rgb)
        else:
            xyz, rgb = self._mesh_cache[cache_key]
            idx = farthest_point_sample_np(xyz, self.n_points)
            xyz, rgb = xyz[idx], rgb[idx]
        return xyz.copy(), rgb.copy()

    def _parse_grasp(
        self, grasp_params: list[float], obj_position: np.ndarray
    ) -> np.ndarray:
        """Parse grasp parameters and center at object position.

        Args:
            grasp_params: [tx, ty, tz, qw, qx, qy, qz, j0, j1, ...].
            obj_position: Object position for centering.

        Returns:
            Pose vector [trans(3), rot6d(6), joints(J)] as float32 array.
        """
        params = np.array(grasp_params, dtype=np.float32)
        trans = params[:3] - obj_position
        quat = params[3:7]
        joints = params[7 : 7 + self.joint_dim]

        if len(joints) < self.joint_dim:
            joints = np.pad(joints, (0, self.joint_dim - len(joints)))

        quat_tensor = torch.from_numpy(quat).unsqueeze(0)
        rot6d = quaternion_to_rotation_6d(quat_tensor).squeeze(0).numpy()

        pose = np.concatenate([trans, rot6d, joints])
        return pose

    def _augment_point_cloud(
        self, xyz: np.ndarray, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random rotation around Z-axis and jittering."""
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        xyz = xyz @ rot.T

        xyz += np.random.normal(0, 0.005, xyz.shape).astype(np.float32)
        rgb = np.clip(rgb + np.random.normal(0, 0.02, rgb.shape), 0, 1).astype(
            np.float32
        )
        return xyz, rgb

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        obj_id = item["obj_id"]
        obj_pose = np.array(item["obj_pose"], dtype=np.float32)
        obj_position = obj_pose[:3]
        obj_quat = obj_pose[3:7]
        obj_scale = item.get("obj_scale", 0.1)

        xyz, rgb = self._get_mesh_points(obj_id, obj_scale)
        xyz = apply_pose_to_points(xyz, obj_position, obj_quat)

        xyz = xyz - obj_position

        if self.augment:
            xyz, rgb = self._augment_point_cloud(xyz, rgb)

        pose_left = self._parse_grasp(item["dex_grasp_left"], obj_position)
        pose_right = self._parse_grasp(item["dex_grasp_right"], obj_position)

        guidance = item.get("guidance", "")
        if not guidance:
            action = item.get("action", "grasp")
            cate = item.get("cate_id", "object")
            guidance = f"Grasp the {cate} object with action {action}"

        contact = item.get("contact_area", [])
        if contact:
            if isinstance(contact, list):
                contact_str = ", ".join(str(c) for c in contact)
            else:
                contact_str = str(contact)
            guidance = f"{guidance}, with contacts on {contact_str}"

        return {
            "xyz": torch.from_numpy(xyz),
            "rgb": torch.from_numpy(rgb),
            "text": guidance,
            "pose_left": torch.from_numpy(pose_left),
            "pose_right": torch.from_numpy(pose_right),
            "obj_id": obj_id,
            "cate_id": item.get("cate_id", ""),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function that handles text fields."""
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
    n_points: int = 10000,
    joint_dim: int = 22,
    augment: bool = False,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for the dexterous grasp dataset."""
    dataset = DexGraspDataset(
        data_path=data_path,
        mesh_root=mesh_root,
        n_points=n_points,
        joint_dim=joint_dim,
        augment=augment,
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
