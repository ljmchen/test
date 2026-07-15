"""PointNet++ encoder for colored point cloud feature extraction.

Replaces Uni3D from the original DexVLG architecture. PointNet++ uses
hierarchical set abstraction to progressively downsample and aggregate
local geometric features, producing per-point and global features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """K-nearest neighbors using pairwise distances.

    Args:
        x: Point positions, shape (B, N, 3).
        k: Number of neighbors.

    Returns:
        Indices of k nearest neighbors, shape (B, N, k).
    """
    dist = torch.cdist(x, x)
    return dist.topk(k, largest=False).indices


def farthest_point_sample(
    xyz: torch.Tensor, npoint: int, deterministic: bool = False
) -> torch.Tensor:
    """Farthest point sampling.

    Args:
        xyz: Point positions, shape (B, N, 3).
        npoint: Number of points to sample.
        deterministic: Start from point index 0 instead of a random point.
            Used at eval time so inference is reproducible and independent
            of batch composition; training keeps the random start (acts as
            a mild augmentation).

    Returns:
        Indices of sampled points, shape (B, npoint).
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    if deterministic:
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    else:
        farthest = torch.randint(0, N, (B,), device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)

    return centroids


def ball_query(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    radius: float,
    nsample: int,
) -> torch.Tensor:
    """Ball query: find points within radius, fallback to KNN if fewer than nsample.

    Args:
        xyz: All points, shape (B, N, 3).
        new_xyz: Query points, shape (B, S, 3).
        radius: Search radius.
        nsample: Max number of neighbors.

    Returns:
        Group indices, shape (B, S, nsample).
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    device = xyz.device

    dist = torch.cdist(new_xyz, xyz)
    _, idx = dist.topk(nsample, largest=False)

    mask = dist.gather(-1, idx) > radius
    first_idx = idx[:, :, 0:1].expand_as(idx)
    idx[mask] = first_idx[mask]

    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index into points tensor using idx.

    Args:
        points: Input points, shape (B, N, C).
        idx: Index tensor, shape (B, S, ...).

    Returns:
        Indexed points, shape (B, S, ..., C).
    """
    B = points.shape[0]
    batch_indices = torch.arange(B, device=points.device).view(B, *([1] * (idx.dim() - 1)))
    batch_indices = batch_indices.expand_as(idx)
    return points[batch_indices, idx]


class SetAbstraction(nn.Module):
    """PointNet++ Set Abstraction layer."""

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp_channels: list[int],
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []
        last_channel = in_channel + 3
        for out_channel in mlp_channels:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: Point positions, shape (B, N, 3).
            features: Point features, shape (B, N, C) or None.

        Returns:
            new_xyz: Sampled point positions, shape (B, npoint, 3).
            new_features: Aggregated features, shape (B, npoint, C').
        """
        B, N, _ = xyz.shape

        fps_idx = farthest_point_sample(
            xyz, self.npoint, deterministic=not self.training
        )
        new_xyz = index_points(xyz, fps_idx)

        group_idx = ball_query(xyz, new_xyz, self.radius, self.nsample)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz -= new_xyz.unsqueeze(2)

        if features is not None:
            grouped_features = index_points(features, group_idx)
            grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
        else:
            grouped_features = grouped_xyz

        grouped_features = grouped_features.reshape(
            B * self.npoint, self.nsample, -1
        ).permute(0, 2, 1)
        grouped_features = self.mlp(grouped_features)
        new_features = grouped_features.max(dim=-1).values
        new_features = new_features.reshape(B, self.npoint, -1)

        return new_xyz, new_features


class PointNet2Encoder(nn.Module):
    """PointNet++ encoder that processes colored point clouds into token features.

    Replaces Uni3D in the original DexVLG. Produces a set of point-wise
    feature tokens suitable for projection into the language model's
    embedding space.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_output_tokens: int = 64,
        output_dim: int = 256,
    ):
        """
        Args:
            in_channels: Input feature channels (3 for xyz-only, 6 for xyz+rgb).
            num_output_tokens: Number of output feature tokens.
            output_dim: Dimension of each output token.
        """
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.feature_dim = in_channels - 3

        self.sa1 = SetAbstraction(512, 0.1, 32, self.feature_dim, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.2, 64, 128, [128, 128, 256])
        self.sa3 = SetAbstraction(num_output_tokens, 0.4, 64, 256, [256, 256, output_dim])

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor | None = None,
        return_centers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: Point positions, shape (B, N, 3).
            features: Point features (e.g., RGB), shape (B, N, C). Ignored when
                the encoder is configured xyz-only (in_channels == 3), so the
                caller may pass colors unconditionally.
            return_centers: When True, also return the final set-abstraction
                token centers ``xyz3`` (B, num_output_tokens, 3) in the SAME
                (raw, object-centered) frame as the input ``xyz``. These are the
                FPS centers of the output tokens (index-aligned with the
                returned features) and are the visualizable substrate for a
                grounded affordance/contact heatmap. Default False keeps the
                legacy single-tensor return bit-identical.

        Returns:
            Token features, shape (B, num_output_tokens, output_dim); or, when
            ``return_centers``, ``(features, xyz3)`` with ``xyz3`` (B, num_output_tokens, 3).
        """
        if self.feature_dim == 0:
            features = None  # xyz-only: drop colors to match the first SA layer
        xyz1, feat1 = self.sa1(xyz, features)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        xyz3, feat3 = self.sa3(xyz2, feat2)
        if return_centers:
            return feat3, xyz3
        return feat3
