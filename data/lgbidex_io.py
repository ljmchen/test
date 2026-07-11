"""Geometry / mesh / point-cloud IO helpers for the LGBiDex data format.

These routines are ported (near verbatim) from the dexvlm pipeline's
``project/data/datasets_v2.py`` so that the test reproduction consumes the
exact same on-disk format and coordinate conventions:

  - object placement uses a 7-dim ``obj_pose`` (translation + quaternion);
  - point clouds are loaded from colored ``.ply`` files or sampled from
    ``.obj`` / ``.ply`` meshes;
  - single-hand grasps are 28-dim ``[translation(3), axis_angle(3), joints(22)]``.

Keeping these helpers self-contained makes the test project runnable without
importing across project roots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# ─── Rotation conversions ──────────────────────────────────────────────────


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices."""
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / theta.clamp_min(1e-8)
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    identity = identity.expand(axis_angle.shape[:-1] + (3, 3))
    axis_outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    return cos_theta * identity + (1.0 - cos_theta) * axis_outer + sin_theta * skew


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to the 6D rotation representation."""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def normalize_rotation_6d(rotation_6d: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(rotation_6d_to_matrix(rotation_6d))


def canonicalize_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / theta.clamp_min(1e-8)
    wrapped = torch.remainder(theta + torch.pi, 2.0 * torch.pi) - torch.pi
    canonical = axis * wrapped
    return torch.where(theta > 1e-8, canonical, torch.zeros_like(axis_angle))


def quaternion_to_matrix(quaternion: torch.Tensor, order: str = "wxyz") -> torch.Tensor:
    """Convert a quaternion to a rotation matrix (default order: wxyz)."""
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion dim 4, got {quaternion.shape[-1]}.")
    if order not in {"xyzw", "wxyz"}:
        raise ValueError(f"Unsupported quaternion order '{order}'.")
    if order == "xyzw":
        quaternion = torch.cat([quaternion[..., 3:4], quaternion[..., :3]], dim=-1)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(dim=-1)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return torch.stack(
        [
            torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1),
            torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1),
            torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )


# ─── Record field resolution ───────────────────────────────────────────────


def resolve_obj_scale(record: dict) -> float:
    """Resolve the object scale of a split record.

    Args:
        record: Split record; uses ``obj_scale`` when present, else ``scale_id / 100``.

    Returns:
        Object scale as a float.

    Raises:
        ValueError: If the record has neither ``obj_scale`` nor ``scale_id``.
    """
    if record.get("obj_scale") is not None:
        return float(record["obj_scale"])
    if record.get("scale_id") is not None:
        return float(record["scale_id"]) / 100.0
    raise ValueError(
        f"Record obj_id={record.get('obj_id', '<unknown>')!r} has neither "
        f"'obj_scale' nor 'scale_id'; cannot resolve object scale."
    )


# ─── Point / pose transforms ───────────────────────────────────────────────


def transform_points(
    points: torch.Tensor,
    rotation_matrix: torch.Tensor | None = None,
    translation: torch.Tensor | None = None,
    scale: float | torch.Tensor | None = None,
) -> torch.Tensor:
    transformed = points
    if scale is not None:
        transformed = transformed * torch.as_tensor(scale, dtype=points.dtype, device=points.device)
    if rotation_matrix is not None:
        transformed = transformed @ rotation_matrix.transpose(-1, -2)
    if translation is not None:
        transformed = transformed + translation
    return transformed


def shift_pose_translation(pose: torch.Tensor, translation_offset: torch.Tensor) -> torch.Tensor:
    shifted_pose = pose.clone()
    shifted_pose[..., :3] = shifted_pose[..., :3] - translation_offset.to(device=pose.device, dtype=pose.dtype)
    return shifted_pose


def sample_uniform_points(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Uniformly sample a fixed number of points from a point set."""
    if points.size(0) == 0:
        raise ValueError("Cannot sample from an empty point set.")
    if points.size(0) >= num_samples:
        indices = torch.randperm(points.size(0), device=points.device)[:num_samples]
        return points.index_select(0, indices)

    extra_indices = torch.randint(
        points.size(0),
        (num_samples - points.size(0),),
        device=points.device,
    )
    indices = torch.cat(
        [torch.arange(points.size(0), device=points.device), extra_indices],
        dim=0,
    )
    return points.index_select(0, indices)


def sample_uniform_points_from_mesh(vertices: torch.Tensor, faces: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Area-weighted uniform surface sampling of a triangle mesh."""
    if vertices.size(0) == 0:
        raise ValueError("Cannot sample from an empty mesh.")
    if faces.numel() == 0:
        return sample_uniform_points(vertices, num_samples)

    triangles = vertices[faces]
    face_normals = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1)
    face_areas = torch.linalg.norm(face_normals, dim=-1)
    valid_faces = face_areas > 1e-12
    if not torch.any(valid_faces):
        return sample_uniform_points(vertices, num_samples)

    triangles = triangles[valid_faces]
    face_areas = face_areas[valid_faces]
    sampled_face_indices = torch.multinomial(face_areas, num_samples, replacement=True)
    sampled_triangles = triangles.index_select(0, sampled_face_indices)

    u = torch.rand(num_samples, 1, dtype=vertices.dtype, device=vertices.device)
    v = torch.rand(num_samples, 1, dtype=vertices.dtype, device=vertices.device)
    sqrt_u = torch.sqrt(u)
    barycentric_a = 1.0 - sqrt_u
    barycentric_b = sqrt_u * (1.0 - v)
    barycentric_c = sqrt_u * v
    return (
        sampled_triangles[:, 0] * barycentric_a
        + sampled_triangles[:, 1] * barycentric_b
        + sampled_triangles[:, 2] * barycentric_c
    )


def _triangulate_face(indices: list[int]) -> list[list[int]]:
    if len(indices) < 3:
        return []
    if len(indices) == 3:
        return [indices]
    return [[indices[0], indices[i], indices[i + 1]] for i in range(1, len(indices) - 1)]


# ─── Mesh / point-cloud loaders ────────────────────────────────────────────


def load_mesh_geometry(mesh_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load an OakInk ``.obj`` / ``.ply`` mesh and return vertices and faces."""
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        with mesh_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    continue
                if line.startswith("f "):
                    raw_indices: list[int] = []
                    for token in line.strip().split()[1:]:
                        vertex_token = token.split("/")[0]
                        if not vertex_token:
                            continue
                        index = int(vertex_token)
                        if index < 0:
                            index = len(vertices) + index
                        else:
                            index = index - 1
                        raw_indices.append(index)
                    faces.extend(_triangulate_face(raw_indices))
        if not vertices:
            raise ValueError(f"No vertices found in OBJ file {mesh_path}.")
        vertex_tensor = torch.tensor(vertices, dtype=torch.float32)
        face_tensor = torch.tensor(faces, dtype=torch.long) if faces else torch.empty((0, 3), dtype=torch.long)
        return vertex_tensor, face_tensor

    if suffix == ".ply":
        with mesh_path.open("rb") as handle:
            header_lines: list[str] = []
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError(f"Invalid PLY file without end_header: {mesh_path}")
                decoded = line.decode("ascii").strip()
                header_lines.append(decoded)
                if decoded == "end_header":
                    break

            format_name = None
            vertex_count = None
            face_count = 0
            vertex_properties: list[tuple[str, str]] = []
            in_vertex = False
            in_face = False
            for line in header_lines:
                if line.startswith("format "):
                    format_name = line.split()[1]
                    continue
                if line.startswith("element vertex "):
                    vertex_count = int(line.split()[-1])
                    in_vertex = True
                    in_face = False
                    continue
                if line.startswith("element face "):
                    face_count = int(line.split()[-1])
                    in_vertex = False
                    in_face = True
                    continue
                if line.startswith("element "):
                    in_vertex = False
                    in_face = False
                    continue
                if in_vertex and line.startswith("property "):
                    _, prop_type, prop_name = line.split()[:3]
                    if prop_type == "list":
                        raise ValueError(f"List vertex property is not supported in {mesh_path}.")
                    vertex_properties.append((prop_name, prop_type))
                if in_face and line.startswith("property list "):
                    _, _, count_type, index_type, prop_name = line.split()[:5]
                    if prop_name != "vertex_indices":
                        raise ValueError(f"Unsupported face property '{prop_name}' in {mesh_path}.")
            if vertex_count is None:
                raise ValueError(f"PLY file {mesh_path} is missing vertex count.")
            if format_name is None:
                raise ValueError(f"PLY file {mesh_path} is missing format declaration.")

            dtype_map = {
                "char": "i1",
                "uchar": "u1",
                "short": "i2",
                "ushort": "u2",
                "int": "i4",
                "uint": "u4",
                "float": "f4",
                "double": "f8",
            }

            if format_name == "ascii":
                vertices: list[list[float]] = []
                for _ in range(vertex_count):
                    parts = handle.readline().decode("ascii").strip().split()
                    values = {name: float(parts[idx]) for idx, (name, _) in enumerate(vertex_properties)}
                    vertices.append([values["x"], values["y"], values["z"]])
                faces: list[list[int]] = []
                for _ in range(face_count):
                    parts = handle.readline().decode("ascii").strip().split()
                    count = int(parts[0])
                    indices = [int(value) for value in parts[1 : 1 + count]]
                    faces.extend(_triangulate_face(indices))
                return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.long) if faces else torch.empty((0, 3), dtype=torch.long)

            if format_name not in {"binary_little_endian", "binary_big_endian"}:
                raise ValueError(f"Unsupported PLY format '{format_name}' for {mesh_path}.")
            endian_prefix = "<" if format_name == "binary_little_endian" else ">"
            fields = [(prop_name, endian_prefix + dtype_map[prop_type]) for prop_name, prop_type in vertex_properties]
            vertex_array = np.fromfile(handle, dtype=np.dtype(fields), count=vertex_count)
            for axis in ("x", "y", "z"):
                if axis not in vertex_array.dtype.names:
                    raise ValueError(f"PLY file {mesh_path} is missing '{axis}' vertex field.")
            xyz = np.stack([vertex_array["x"], vertex_array["y"], vertex_array["z"]], axis=-1)

            count_dtype = np.dtype(endian_prefix + dtype_map["uchar"])
            index_dtype = np.dtype(endian_prefix + dtype_map["int"])
            faces: list[list[int]] = []
            for _ in range(face_count):
                raw_count = np.fromfile(handle, dtype=count_dtype, count=1)
                if raw_count.size != 1:
                    raise ValueError(f"Unexpected EOF while reading face count from {mesh_path}.")
                count = int(raw_count[0])
                indices = np.fromfile(handle, dtype=index_dtype, count=count)
                if indices.size != count:
                    raise ValueError(f"Unexpected EOF while reading face indices from {mesh_path}.")
                faces.extend(_triangulate_face(indices.astype(np.int64).tolist()))
            return torch.tensor(xyz, dtype=torch.float32), torch.tensor(faces, dtype=torch.long) if faces else torch.empty((0, 3), dtype=torch.long)

    raise ValueError(f"Unsupported mesh format '{mesh_path.suffix}' for {mesh_path}.")


def load_mesh_vertices(mesh_path: Path) -> torch.Tensor:
    """Load OakInk ``.obj`` / ``.ply`` vertices and return ``[N, 3]``."""
    vertices, _ = load_mesh_geometry(mesh_path)
    return vertices


def load_point_cloud_from_ply(
    ply_path: Path,
    return_color_status: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, bool]:
    """Load a colored PLY point cloud and return ``[N, 6]`` (xyz + rgb)."""
    if ply_path.suffix.lower() != ".ply":
        raise ValueError(f"Unsupported point cloud format '{ply_path.suffix}' for {ply_path}.")

    def _normalize_colors(colors: np.ndarray) -> np.ndarray:
        colors = colors.astype(np.float32)
        if colors.size == 0:
            return colors
        if float(colors.max()) > 1.0:
            colors = colors / 255.0
        return np.clip(colors, 0.0, 1.0)

    def _extract_colors_from_mapping(values: dict[str, Any]) -> list[float]:
        if all(channel in values for channel in ("red", "green", "blue")):
            return [float(values["red"]), float(values["green"]), float(values["blue"])]
        if all(channel in values for channel in ("r", "g", "b")):
            return [float(values["r"]), float(values["g"]), float(values["b"])]
        return [0.0, 0.0, 0.0]

    def _with_color_status(point_cloud: torch.Tensor, has_real_color: bool):
        if return_color_status:
            return point_cloud, has_real_color
        return point_cloud

    with ply_path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY file without end_header: {ply_path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        format_name = None
        vertex_count = None
        vertex_properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_lines:
            if line.startswith("format "):
                format_name = line.split()[1]
                continue
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex = True
                continue
            if line.startswith("element "):
                in_vertex = False
                continue
            if in_vertex and line.startswith("property "):
                _, prop_type, prop_name = line.split()[:3]
                if prop_type == "list":
                    raise ValueError(f"List vertex property is not supported in {ply_path}.")
                vertex_properties.append((prop_name, prop_type))

        if vertex_count is None:
            raise ValueError(f"PLY file {ply_path} is missing vertex count.")
        if format_name is None:
            raise ValueError(f"PLY file {ply_path} is missing format declaration.")

        dtype_map = {
            "char": "i1",
            "uchar": "u1",
            "short": "i2",
            "ushort": "u2",
            "int": "i4",
            "uint": "u4",
            "float": "f4",
            "double": "f8",
        }

        if format_name == "ascii":
            property_names = {name for name, _ in vertex_properties}
            has_real_color = all(channel in property_names for channel in ("red", "green", "blue")) or all(
                channel in property_names for channel in ("r", "g", "b")
            )
            xyz_rows: list[list[float]] = []
            color_rows: list[list[float]] = []
            for _ in range(vertex_count):
                parts = handle.readline().decode("ascii").strip().split()
                values = {
                    name: float(parts[idx])
                    for idx, (name, _) in enumerate(vertex_properties)
                }
                xyz_rows.append([values["x"], values["y"], values["z"]])
                color_rows.append(_extract_colors_from_mapping(values))
            xyz = np.asarray(xyz_rows, dtype=np.float32)
            colors = _normalize_colors(np.asarray(color_rows, dtype=np.float32))
            return _with_color_status(torch.from_numpy(np.concatenate([xyz, colors], axis=-1)), has_real_color)

        if format_name not in {"binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"Unsupported PLY format '{format_name}' for {ply_path}.")
        endian_prefix = "<" if format_name == "binary_little_endian" else ">"
        fields = [(prop_name, endian_prefix + dtype_map[prop_type]) for prop_name, prop_type in vertex_properties]
        vertex_array = np.fromfile(handle, dtype=np.dtype(fields), count=vertex_count)
        for axis in ("x", "y", "z"):
            if axis not in vertex_array.dtype.names:
                raise ValueError(f"PLY file {ply_path} is missing '{axis}' vertex field.")
        xyz = np.stack([vertex_array["x"], vertex_array["y"], vertex_array["z"]], axis=-1).astype(np.float32)

        color_field_names = None
        if all(channel in vertex_array.dtype.names for channel in ("red", "green", "blue")):
            color_field_names = ("red", "green", "blue")
        elif all(channel in vertex_array.dtype.names for channel in ("r", "g", "b")):
            color_field_names = ("r", "g", "b")

        if color_field_names is None:
            has_real_color = False
            colors = np.zeros((vertex_count, 3), dtype=np.float32)
        else:
            has_real_color = True
            colors = np.stack([vertex_array[name] for name in color_field_names], axis=-1)
            colors = _normalize_colors(colors)
        return _with_color_status(
            torch.from_numpy(np.concatenate([xyz, colors], axis=-1).astype(np.float32)),
            has_real_color,
        )


# ─── Color selection ───────────────────────────────────────────────────────


def make_constant_color(point_count: int, dtype: torch.dtype, fill_value: float) -> torch.Tensor:
    return torch.full((point_count, 3), float(fill_value), dtype=dtype)


def normalize_point_cloud_color_mode(value: str | None) -> str:
    normalized = "fill" if value is None else str(value).strip().lower()
    aliases = {
        "constant": "fill",
        "const": "fill",
        "filled": "fill",
        "rgb": "real",
        "true": "real",
        "true_color": "real",
        "real_color": "real",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"fill", "real"}:
        raise ValueError(f"point_cloud_color_mode must be 'fill' or 'real', got {value!r}.")
    return normalized


def select_point_cloud_colors(
    raw_point_cloud: torch.Tensor,
    *,
    color_mode: str,
    color_fill: float,
    has_real_color: bool | None = None,
) -> torch.Tensor:
    mode = normalize_point_cloud_color_mode(color_mode)
    if mode == "real" and raw_point_cloud.shape[-1] >= 6 and has_real_color is not False:
        return raw_point_cloud[:, 3:6]
    return make_constant_color(raw_point_cloud.size(0), raw_point_cloud.dtype, color_fill)
