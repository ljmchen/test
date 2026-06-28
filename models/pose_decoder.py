"""Pose decoder that converts flow-matching output to structured grasp parameters.

Decodes the raw output from the flow-matching transformer into
translation (T), rotation (R in 6D representation), and joint angles (theta).
"""

import torch
import torch.nn as nn


class PoseDecoder(nn.Module):
    """Decodes pose features into structured grasp parameters.

    For each hand, outputs:
        - Translation T: 3D position
        - Rotation R: 6D rotation representation
        - Joint angles theta: N-dimensional joint configuration
    """

    def __init__(
        self,
        pose_dim: int,
        trans_dim: int = 3,
        rot_dim: int = 6,
        joint_dim: int = 22,
        hidden_dim: int = 512,
    ):
        """
        Args:
            pose_dim: Input pose feature dimension from flow matching.
            trans_dim: Translation dimensions (3).
            rot_dim: Rotation dimensions (6 for 6D representation).
            joint_dim: Number of joint angles.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.trans_dim = trans_dim
        self.rot_dim = rot_dim
        self.joint_dim = joint_dim
        self.output_dim = trans_dim + rot_dim + joint_dim

        self.decoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Pose features, shape (B, pose_dim).

        Returns:
            Dictionary with 'translation' (B, 3), 'rotation' (B, 6), 'joints' (B, J).
        """
        out = self.decoder(x)
        t = out[..., : self.trans_dim]
        r = out[..., self.trans_dim : self.trans_dim + self.rot_dim]
        theta = out[..., self.trans_dim + self.rot_dim :]
        return {"translation": t, "rotation": r, "joints": theta}

    @property
    def total_dim(self) -> int:
        return self.output_dim
