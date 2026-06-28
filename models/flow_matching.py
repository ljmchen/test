"""Flow-Matching Transformer for conditional pose generation.

Implements a DiT-style transformer that learns a velocity field v(x_t, t, c)
for conditional flow matching. The condition c comes from cross-attention
to the fused point-cloud + language features from the backbone.
"""

import math
import torch
import torch.nn as nn
from einops import rearrange


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timestep."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AdaLayerNorm(nn.Module):
    """Adaptive layer normalization conditioned on timestep embedding."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class FlowMatchingBlock(nn.Module):
    """Single transformer block with self-attention + cross-attention + FFN."""

    def __init__(self, dim: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = AdaLayerNorm(dim, cond_dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm3 = AdaLayerNorm(dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attn(*([self.norm1(x, time_emb)] * 3))[0]
        x = x + self.cross_attn(self.norm2(x, time_emb), cond_tokens, cond_tokens)[0]
        x = x + self.ffn(self.norm3(x, time_emb))
        return x


class FlowMatchingTransformer(nn.Module):
    """Transformer-based velocity field for conditional flow matching.

    Takes noisy pose queries and denoises them conditioned on fused
    vision-language features via cross-attention.
    """

    def __init__(
        self,
        pose_dim: int,
        num_queries: int = 1,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        cond_dim: int = 768,
        time_dim: int = 256,
        mlp_ratio: float = 4.0,
    ):
        """
        Args:
            pose_dim: Dimension of the pose vector.
            num_queries: Number of noise queries (1 for single-hand, 2 for bimanual).
            dim: Hidden dimension of the transformer.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            cond_dim: Dimension of condition tokens from the backbone.
            time_dim: Dimension of timestep embedding.
            mlp_ratio: MLP expansion ratio.
        """
        super().__init__()
        self.pose_dim = pose_dim
        self.num_queries = num_queries

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.input_proj = nn.Linear(pose_dim, dim)
        self.query_pos = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.cond_proj = nn.Linear(cond_dim, dim) if cond_dim != dim else nn.Identity()

        self.blocks = nn.ModuleList(
            [FlowMatchingBlock(dim, num_heads, dim, mlp_ratio) for _ in range(depth)]
        )

        self.final_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, pose_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field v(x_t, t, c).

        Args:
            x_t: Noisy pose, shape (B, num_queries, pose_dim) or (B, pose_dim).
            t: Timestep in [0, 1], shape (B,).
            cond_tokens: Condition features, shape (B, L, cond_dim).

        Returns:
            Predicted velocity, same shape as x_t.
        """
        squeeze = False
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
            squeeze = True

        time_emb = sinusoidal_embedding(t, self.time_mlp[0].in_features)
        time_emb = self.time_mlp(time_emb)

        x = self.input_proj(x_t) + self.query_pos[:, : x_t.shape[1]]
        cond = self.cond_proj(cond_tokens)

        for block in self.blocks:
            x = block(x, cond, time_emb)

        x = self.final_norm(x)
        v = self.output_proj(x)

        if squeeze:
            v = v.squeeze(1)
        return v
