"""Autoregressive latent reasoner for hand-sequential grasp generation.

Implements a Coconut-style continuous latent thinking chain followed by
per-hand decision steps. The reasoner unrolls a causal decoder over a short
sequence:

    [BOT] -> z_1..z_K (latent thinking: each position's output hidden is fed
             back, through a feedback LayerNorm, as the next position's input)
          -> hand step 0: act_embed + step_embed[0]
          -> hand step 1: act_embed + step_embed[1] + embed_prev_hand(side_0, pose_0)

Each hand step emits a hidden state h_s used to condition a per-hand
flow-matching head, plus presence (emit-or-stop) and side (left/right) logits.
Causal masking guarantees that recomputing the full prefix reproduces every
earlier position bit-exactly, which keeps teacher-forced and incremental
rollouts numerically identical.
"""

import torch
import torch.nn as nn


class ReasonerBlock(nn.Module):
    """Pre-LN decoder block: causal self-attention, cross-attention, FFN."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
        mem_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Sequence hidden states, shape (B, T, dim).
            memory: Cross-attention memory, shape (B, L, dim).
            causal_mask: Bool mask, shape (T, T), True = disallowed.
            mem_key_padding_mask: Optional bool mask over memory, shape
                (B, L); True = pad (excluded from cross-attention). None
                keeps the legacy all-visible behavior.

        Returns:
            Updated hidden states, shape (B, T, dim).
        """
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, attn_mask=causal_mask)[0]
        x = x + self.cross_attn(
            self.norm2(x), memory, memory, key_padding_mask=mem_key_padding_mask
        )[0]
        x = x + self.ffn(self.norm3(x))
        return x


class LatentReasoner(nn.Module):
    """Autoregressive latent thinking + per-hand decision decoder.

    Produces K latent thinking tokens (Coconut-style feedback loop) and up to
    ``max_hands`` hand hidden states with presence/side decision logits, all
    conditioned on fused vision-language memory via cross-attention.
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        mem_dim: int = 768,
        pose_dim: int = 31,
        num_thinking_tokens: int = 6,
        max_hands: int = 2,
        feedback_norm: bool = True,
        mlp_ratio: float = 4.0,
        aux_probe: dict | None = None,
    ):
        """
        Args:
            dim: Hidden dimension of the reasoner.
            depth: Number of decoder blocks.
            num_heads: Number of attention heads.
            mem_dim: Dimension of the cross-attention memory tokens.
            pose_dim: Dimension of a single-hand pose vector.
            num_thinking_tokens: Number K of latent thinking tokens.
            max_hands: Maximum number of hand steps.
            feedback_norm: Apply LayerNorm to the Coconut feedback hidden.
            mlp_ratio: FFN expansion ratio.
            aux_probe: Optional probe config dict with keys ``enabled``
                (default False), ``hidden_dim`` (default 256),
                ``num_task_types`` (default 4). Disabled probes create no
                parameters.
        """
        super().__init__()
        self.dim = dim
        self.pose_dim = pose_dim
        self.num_thinking_tokens = num_thinking_tokens
        self.max_hands = max_hands

        self.bot_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.act_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.step_embed = nn.Parameter(torch.randn(1, max_hands, dim) * 0.02)
        self.side_embed = nn.Embedding(2, dim)

        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        self.mem_proj = nn.Linear(mem_dim, dim)
        self.blocks = nn.ModuleList(
            [ReasonerBlock(dim, num_heads, mlp_ratio) for _ in range(depth)]
        )

        self.feedback_ln = nn.LayerNorm(dim) if feedback_norm else nn.Identity()
        self.final_norm = nn.LayerNorm(dim)
        self.presence_head = nn.Linear(dim, 1)
        self.side_head = nn.Linear(dim, 2)

        probe_cfg = aux_probe or {}
        self.probe_enabled = bool(probe_cfg.get("enabled", False))
        if self.probe_enabled:
            probe_hidden = int(probe_cfg.get("hidden_dim", 256))
            num_task_types = int(probe_cfg.get("num_task_types", 4))
            self.probe_trunk = nn.Sequential(
                nn.Linear(dim, probe_hidden),
                nn.GELU(),
            )
            self.probe_task_head = nn.Linear(probe_hidden, num_task_types)
            self.probe_lobj_head = nn.Linear(probe_hidden, 1)

    def embed_prev_hand(self, side: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Embed the previous hand's decision for conditioning the next step.

        Args:
            side: Hand side ids (0=left, 1=right), shape (B,).
            pose: Hand pose in normalized space, shape (B, pose_dim).

        Returns:
            Previous-hand embedding, shape (B, dim).
        """
        return self.side_embed(side) + self.pose_mlp(pose)

    def _forward_prefix(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        mem_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run all decoder blocks over an input prefix with causal masking.

        Args:
            inputs: Input token embeddings, shape (B, T, dim).
            memory: Raw cross-attention memory, shape (B, L, mem_dim).
            mem_key_padding_mask: Optional bool mask over memory, shape
                (B, L); True = pad.

        Returns:
            Raw hidden states (no final norm), shape (B, T, dim).
        """
        T = inputs.shape[1]
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=inputs.device), diagonal=1
        )
        mem = self.mem_proj(memory)
        x = inputs
        for block in self.blocks:
            x = block(x, mem, causal_mask, mem_key_padding_mask=mem_key_padding_mask)
        return x

    def _run_thinking(
        self,
        memory: torch.Tensor,
        mem_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Unroll the Coconut thinking chain.

        Args:
            memory: Cross-attention memory, shape (B, L, mem_dim).
            mem_key_padding_mask: Optional bool mask over memory, shape
                (B, L); True = pad.

        Returns:
            Input prefix embeddings [BOT, z_1..z_K], shape (B, 1+K, dim).
        """
        B = memory.shape[0]
        inputs = self.bot_embed.expand(B, -1, -1)
        for _ in range(self.num_thinking_tokens):
            h_raw = self._forward_prefix(
                inputs, memory, mem_key_padding_mask=mem_key_padding_mask
            )
            feedback = self.feedback_ln(h_raw[:, -1:])
            inputs = torch.cat([inputs, feedback], dim=1)
        return inputs

    def _hand_step_token(
        self,
        step_idx: int,
        batch_size: int,
        prev_side: torch.Tensor | None,
        prev_pose: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build the input token for hand step ``step_idx``.

        Args:
            step_idx: Hand step index in [0, max_hands).
            batch_size: Batch size B.
            prev_side: Previous hand side ids, shape (B,); required for step_idx >= 1.
            prev_pose: Previous hand pose, shape (B, pose_dim); required for step_idx >= 1.

        Returns:
            Input token, shape (B, 1, dim).
        """
        tok = (self.act_embed + self.step_embed[:, step_idx : step_idx + 1]).expand(
            batch_size, -1, -1
        )
        if step_idx >= 1:
            if prev_side is None or prev_pose is None:
                raise ValueError(
                    f"hand step {step_idx} requires prev_side and prev_pose; got "
                    f"prev_side={'None' if prev_side is None else 'set'}, "
                    f"prev_pose={'None' if prev_pose is None else 'set'}"
                )
            tok = tok + self.embed_prev_hand(prev_side, prev_pose).unsqueeze(1)
        return tok

    def rollout_teacher_forced(
        self,
        memory: torch.Tensor,
        gt_prev_pose: torch.Tensor,
        gt_prev_side: torch.Tensor,
        mem_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced rollout over thinking tokens and all hand steps.

        Args:
            memory: Cross-attention memory, shape (B, L, mem_dim).
            gt_prev_pose: GT pose of the first hand (optionally noised),
                shape (B, pose_dim); conditions hand steps s >= 1.
            gt_prev_side: GT side id of the first hand, shape (B,).
            mem_key_padding_mask: Optional bool mask over memory, shape
                (B, L); True = pad.

        Returns:
            Dict with:
                z_tokens: (B, K, dim) thinking hidden states (final-normed).
                hand_hidden: (B, max_hands, dim) hand hidden states.
                presence_logits: (B, max_hands).
                side_logits: (B, max_hands, 2).
        """
        B = memory.shape[0]
        inputs = self._run_thinking(memory, mem_key_padding_mask=mem_key_padding_mask)

        hand_hidden, presence_logits, side_logits = [], [], []
        for s in range(self.max_hands):
            prev_side = gt_prev_side if s >= 1 else None
            prev_pose = gt_prev_pose if s >= 1 else None
            tok = self._hand_step_token(s, B, prev_side, prev_pose)
            inputs = torch.cat([inputs, tok], dim=1)
            h_raw = self._forward_prefix(
                inputs, memory, mem_key_padding_mask=mem_key_padding_mask
            )
            h_s = self.final_norm(h_raw[:, -1])
            hand_hidden.append(h_s)
            presence_logits.append(self.presence_head(h_s).squeeze(-1))
            side_logits.append(self.side_head(h_s))

        K = self.num_thinking_tokens
        z_tokens = self.final_norm(h_raw[:, 1 : 1 + K])
        return {
            "z_tokens": z_tokens,
            "hand_hidden": torch.stack(hand_hidden, dim=1),
            "presence_logits": torch.stack(presence_logits, dim=1),
            "side_logits": torch.stack(side_logits, dim=1),
        }

    def rollout_thinking(
        self,
        memory: torch.Tensor,
        mem_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Unroll only the thinking chain for incremental sampling.

        Args:
            memory: Cross-attention memory, shape (B, L, mem_dim).
            mem_key_padding_mask: Optional bool mask over memory, shape
                (B, L); True = pad.

        Returns:
            z_tokens: (B, K, dim) thinking hidden states (final-normed).
            seq_state: Dict with "inputs" (B, 1+K, dim) input prefix.
        """
        inputs = self._run_thinking(memory, mem_key_padding_mask=mem_key_padding_mask)
        h_raw = self._forward_prefix(
            inputs, memory, mem_key_padding_mask=mem_key_padding_mask
        )
        K = self.num_thinking_tokens
        z_tokens = self.final_norm(h_raw[:, 1 : 1 + K])
        return z_tokens, {"inputs": inputs}

    def step_hand(
        self,
        seq_state: dict[str, torch.Tensor],
        memory: torch.Tensor,
        step_idx: int,
        prev_side: torch.Tensor | None = None,
        prev_pose: torch.Tensor | None = None,
        mem_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Run one incremental hand step during sampling.

        Args:
            seq_state: Dict with "inputs" (B, T, dim) accumulated prefix.
            memory: Cross-attention memory, shape (B, L, mem_dim).
            step_idx: Hand step index in [0, max_hands).
            prev_side: Previous hand side ids, shape (B,); required for step_idx >= 1.
            prev_pose: Previous hand sampled pose, shape (B, pose_dim);
                required for step_idx >= 1.
            mem_key_padding_mask: Optional bool mask over memory, shape
                (B, L); True = pad.

        Returns:
            h_s: (B, dim) hand hidden state (final-normed).
            presence_logit: (B,).
            side_logits: (B, 2).
            new_seq_state: Updated seq_state including this step's token.
        """
        inputs = seq_state["inputs"]
        B = inputs.shape[0]
        tok = self._hand_step_token(step_idx, B, prev_side, prev_pose)
        inputs = torch.cat([inputs, tok], dim=1)
        h_raw = self._forward_prefix(
            inputs, memory, mem_key_padding_mask=mem_key_padding_mask
        )
        h_s = self.final_norm(h_raw[:, -1])
        presence_logit = self.presence_head(h_s).squeeze(-1)
        side_logits = self.side_head(h_s)
        return h_s, presence_logit, side_logits, {"inputs": inputs}

    def probe_forward(self, z_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Auxiliary probe over pooled thinking tokens.

        Args:
            z_tokens: Thinking hidden states, shape (B, K, dim).

        Returns:
            Dict with task_logits (B, num_task_types) and log_l_obj (B,).
        """
        if not self.probe_enabled:
            raise RuntimeError("aux_probe is disabled; enable it in the config")
        pooled = self.probe_trunk(z_tokens.mean(dim=1))
        return {
            "task_logits": self.probe_task_head(pooled),
            "log_l_obj": self.probe_lobj_head(pooled).squeeze(-1),
        }
