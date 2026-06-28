"""Rotation conversion utilities between quaternion, rotation matrix, and 6D representations."""

import torch
import torch.nn.functional as F


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (wxyz) to rotation matrices.

    Args:
        quaternions: Quaternions with real part first, shape (..., 4).

    Returns:
        Rotation matrices, shape (..., 3, 3).
    """
    q = F.normalize(quaternions, dim=-1)
    w, x, y, z = q.unbind(-1)

    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z

    matrix = torch.stack(
        [
            1.0 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1.0 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1.0 - (txx + tyy),
        ],
        dim=-1,
    )
    return matrix.reshape(*quaternions.shape[:-1], 3, 3)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to 6D rotation representation (first two columns).

    Args:
        matrix: Rotation matrices, shape (..., 3, 3).

    Returns:
        6D rotation representation, shape (..., 6).
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrices via Gram-Schmidt.

    Args:
        d6: 6D rotation representation, shape (..., 6).

    Returns:
        Rotation matrices, shape (..., 3, 3).
    """
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions (wxyz).

    Args:
        matrix: Rotation matrices, shape (..., 3, 3).

    Returns:
        Quaternions with real part first, shape (..., 4).
    """
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    batch_size = m.shape[0]

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    quat = torch.zeros(batch_size, 4, device=matrix.device, dtype=matrix.dtype)

    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2.0
    mask = trace > 0
    quat[mask, 0] = 0.25 * s[mask]
    quat[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s[mask]
    quat[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s[mask]
    quat[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s[mask]

    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cond = (~mask) & (m[:, i, i] > m[:, j, j]) & (m[:, i, i] > m[:, k, k])
        if not cond.any():
            continue
        s_i = torch.sqrt(
            torch.clamp(1.0 + m[cond, i, i] - m[cond, j, j] - m[cond, k, k], min=1e-10)
        ) * 2.0
        quat[cond, 0] = (m[cond, k, j] - m[cond, j, k]) / s_i
        quat[cond, i + 1] = 0.25 * s_i
        quat[cond, j + 1] = (m[cond, j, i] + m[cond, i, j]) / s_i
        quat[cond, k + 1] = (m[cond, k, i] + m[cond, i, k]) / s_i

    return F.normalize(quat, dim=-1).reshape(*batch_shape, 4)


def quaternion_to_rotation_6d(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (wxyz) to 6D rotation representation."""
    return matrix_to_rotation_6d(quaternion_to_matrix(quaternions))


def rotation_6d_to_quaternion(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to quaternions (wxyz)."""
    return matrix_to_quaternion(rotation_6d_to_matrix(d6))
