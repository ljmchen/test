"""DexVLG: Language-aligned dexterous grasp pose generation.

Architecture (following the paper diagram):
    1. PointNet++ encodes colored point cloud into spatial tokens
    2. Projector maps point cloud tokens to language model dimension
    3. BERT encodes language instruction into text tokens
    4. Fusion transformer combines point cloud and language tokens
    5. Flow-Matching Transformer denoises pose queries via cross-attention
    6. Output is structured into T, R, theta per hand

This implementation replaces Uni3D with PointNet++ and Florence-2 with BERT.
Supports bimanual dexterous grasping with 2 query tokens (left/right hand).
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2 import PointNet2Encoder
from .flow_matching import FlowMatchingTransformer
from .latent_reasoner import LatentReasoner

logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """Fallback tokenizer when BERT tokenizer is unavailable.

    Uses a fixed vocabulary with character-level fallback. Only intended
    for local testing; real training should use the BERT tokenizer.
    """

    PAD, UNK, CLS, SEP = 0, 1, 2, 3
    VOCAB_SIZE = 30522

    def __call__(
        self,
        texts: list[str],
        padding: bool | str = True,
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: str = "pt",
    ) -> dict:
        all_ids = []
        for text in texts:
            words = text.lower().split()
            ids = [self.CLS]
            for w in words:
                ids.append(hash(w) % (self.VOCAB_SIZE - 4) + 4)
            ids.append(self.SEP)
            if truncation:
                ids = ids[:max_length]
            all_ids.append(ids)

        # padding semantics follow HF tokenizers: True/"longest" pads to the
        # longest sequence in the batch; "max_length" pads to max_length.
        if padding == "max_length" or not padding:
            max_len = max_length
        else:
            max_len = max(len(ids) for ids in all_ids)
        padded, masks = [], []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            padded.append(ids + [self.PAD] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)

        result = {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }
        return result


class FusionTransformer(nn.Module):
    """Lightweight transformer that fuses point cloud and language tokens.

    Replaces Florence-2 LLM as the multimodal fusion backbone.
    """

    def __init__(self, dim: int = 768, depth: int = 4, num_heads: int = 8):
        super().__init__()
        self.pc_type_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.lang_type_embed = nn.Parameter(torch.zeros(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        pc_tokens: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pc_tokens: Point cloud tokens, shape (B, N_pc, dim).
            lang_tokens: Language tokens, shape (B, N_lang, dim).
            lang_mask: Attention mask for language, shape (B, N_lang).

        Returns:
            Fused tokens, shape (B, N_pc + N_lang, dim).
        """
        pc = pc_tokens + self.pc_type_embed
        lang = lang_tokens + self.lang_type_embed

        combined = torch.cat([pc, lang], dim=1)

        pc_mask = torch.ones(
            pc.shape[0], pc.shape[1], dtype=torch.bool, device=pc.device
        )
        attn_mask = torch.cat([pc_mask, lang_mask.bool()], dim=1)
        key_padding_mask = ~attn_mask

        fused = self.encoder(combined, src_key_padding_mask=key_padding_mask)
        return self.norm(fused)


def _resolve_pretrained_path(bert_name: str) -> tuple[str, bool]:
    """Resolve repo-relative pretrained model paths before falling back to HF ids."""
    path = Path(bert_name).expanduser()
    if path.is_dir():
        return str(path), True
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        repo_path = repo_root / path
        if repo_path.is_dir():
            return str(repo_path), True
    return bert_name, False


def _random_bert(bert_dim: int):
    """Random-init BERT + SimpleTokenizer fallback (offline / weights unavailable)."""
    from transformers import BertConfig, BertModel

    config = BertConfig(
        vocab_size=30522,
        hidden_size=bert_dim,
        num_attention_heads=12,
        num_hidden_layers=6,
        intermediate_size=bert_dim * 4,
    )
    return SimpleTokenizer(), BertModel(config)


def _load_bert(bert_name: str, bert_dim: int):
    """Load BERT model and tokenizer, falling back to random init if needed."""
    pretrained_path, local_only = _resolve_pretrained_path(bert_name)
    try:
        from transformers import BertModel, BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(
            pretrained_path, local_files_only=local_only
        )
        bert = BertModel.from_pretrained(pretrained_path, local_files_only=local_only)
        logger.info("Loaded pretrained BERT from '%s'", pretrained_path)
        return tokenizer, bert
    except Exception as e:
        # A configured local path that fails to load is a misconfiguration —
        # fail loudly instead of silently training on random weights. Only fall
        # back when the name is a (non-local) HF id that can't be fetched offline.
        if local_only:
            raise RuntimeError(
                f"Failed to load BERT from configured local path '{pretrained_path}': {e}"
            ) from e
        logger.warning(
            "Could not load '%s' (%s); it is not a local path, using random init",
            pretrained_path, e,
        )
        return _random_bert(bert_dim)


def _load_modernbert(model_name: str, hidden_dim: int):
    """Load ModernBERT (a modern bidirectional encoder) via Auto* classes.

    Interface matches BERT (tokenizer -> input_ids/attention_mask, model ->
    last_hidden_state), so it drops straight into ``encode_language``. Falls back
    to a random BERT when the weights are unavailable (offline / not downloaded).
    """
    pretrained_path, local_only = _resolve_pretrained_path(model_name)
    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(pretrained_path, local_files_only=local_only)
        model = AutoModel.from_pretrained(pretrained_path, local_files_only=local_only)
        got = int(getattr(model.config, "hidden_size", hidden_dim))
        if got != hidden_dim:
            logger.warning(
                "ModernBERT hidden_size=%d != configured model.bert_dim=%d; "
                "set bert_dim=%d to match (projector/fusion use bert_dim).",
                got, hidden_dim, got,
            )
        logger.info("Loaded ModernBERT from '%s' (hidden_size=%d)", pretrained_path, got)
        return tokenizer, model
    except Exception as e:
        if local_only:
            raise RuntimeError(
                f"Failed to load ModernBERT from configured local path '{pretrained_path}': {e}"
            ) from e
        logger.warning(
            "Could not load ModernBERT '%s' (%s); it is not a local path, using random BERT",
            pretrained_path, e,
        )
        return _random_bert(hidden_dim)


def _load_language_model(backbone: str, model_name: str, hidden_dim: int):
    """Dispatch to the configured language backbone (BERT default, or ModernBERT)."""
    # A name that is clearly meant as a local path (e.g. a typo'd pretrained dir)
    # must not be silently treated as an HF id and fall back to random weights.
    looks_local = str(model_name).startswith(("pretrained_models", "models/", "./", "../", "/", "~"))
    if looks_local:
        _, local_only = _resolve_pretrained_path(model_name)
        if not local_only:
            raise FileNotFoundError(
                f"Configured local language-model path '{model_name}' does not exist. "
                "Fix model.bert_model or download the weights; refusing to fall back to random."
            )
    backbone = (backbone or "bert").lower().replace("-", "").replace("_", "")
    if backbone in ("modernbert", "modern"):
        return _load_modernbert(model_name, hidden_dim)
    return _load_bert(model_name, hidden_dim)


class DexVLG(nn.Module):
    """DexVLG model for bimanual dexterous grasp generation.

    Generates language-aligned dexterous grasp poses for both hands
    given a colored point cloud and a natural language instruction.

    Each hand's pose is parameterized as (T, R, theta):
        - T in R^3: wrist translation
        - R in R^6: 6D continuous rotation representation
        - theta in R^J: joint angles (J=22 for Shadow Hand)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        pc_dim = cfg.get("pc_feature_dim", 256)
        bert_dim = cfg.get("bert_dim", 768)
        flow_dim = cfg.get("flow_hidden_dim", 256)
        flow_depth = cfg.get("flow_depth", 6)
        flow_heads = cfg.get("flow_heads", 8)
        fusion_depth = cfg.get("fusion_depth", 4)
        num_pc_tokens = cfg.get("num_pc_tokens", 64)
        joint_dim = cfg.get("joint_dim", 22)
        language_backbone = cfg.get("language_backbone", "bert")
        bert_name = cfg.get("bert_model", "bert-base-uncased")
        freeze_bert = cfg.get("freeze_bert_embeddings", False)

        # Tokenizer padding policy: "longest" (HF padding=True, legacy default,
        # pad count varies with batch composition) or "max_length" (fixed 128
        # tokens, batch-invariant inference).
        self.lang_padding = str(cfg.get("lang_padding", "longest"))
        if self.lang_padding not in ("longest", "max_length"):
            raise ValueError(
                f"Unknown model.lang_padding '{self.lang_padding}'; "
                "expected 'longest' or 'max_length'"
            )
        # Mask language pad tokens in all downstream cross-attention (flow +
        # reasoner). Default False = legacy behavior (pad visible); flipping it
        # requires retraining — old checkpoints saw pad tokens during training.
        self.mask_pad_tokens = bool(cfg.get("mask_pad_tokens", False))

        self.trans_dim = 3
        self.rot_dim = 6
        self.joint_dim = joint_dim
        self.pose_dim = self.trans_dim + self.rot_dim + joint_dim

        # Per-element flow-loss weights. Joints occupy 22/31 dims and otherwise
        # dominate the plain MSE, leaving translation (3) / rotation (6)
        # under-trained. Up-weight them via model.flow_loss_component_weights
        # (all 1.0 -> identical to the old unweighted MSE).
        fw = cfg.get("flow_loss_component_weights", {}) or {}
        flow_weight = torch.cat(
            [
                torch.full((self.trans_dim,), float(fw.get("translation", 1.0))),
                torch.full((self.rot_dim,), float(fw.get("rotation", 1.0))),
                torch.full((joint_dim,), float(fw.get("joint", 1.0))),
            ]
        ).view(1, 1, self.pose_dim)
        self.register_buffer("flow_loss_weight", flow_weight, persistent=False)

        self.pc_encoder = PointNet2Encoder(
            in_channels=cfg.get("pc_in_channels", 6),
            num_output_tokens=num_pc_tokens,
            output_dim=pc_dim,
        )

        self.projector = nn.Sequential(
            nn.Linear(pc_dim, bert_dim),
            nn.LayerNorm(bert_dim),
            nn.GELU(),
            nn.Linear(bert_dim, bert_dim),
        )

        self.tokenizer, self.bert = _load_language_model(language_backbone, bert_name, bert_dim)

        if freeze_bert:
            embeddings = getattr(self.bert, "embeddings", None)
            if embeddings is not None:
                for p in embeddings.parameters():
                    p.requires_grad = False

        self.fusion = FusionTransformer(
            dim=bert_dim, depth=fusion_depth, num_heads=flow_heads,
        )

        self.architecture = cfg.get("architecture", "legacy_bimanual")
        if self.architecture not in ("legacy_bimanual", "latent_ar"):
            raise ValueError(
                f"Unknown model.architecture '{self.architecture}'; "
                "expected 'legacy_bimanual' or 'latent_ar'"
            )

        if self.architecture == "latent_ar":
            reasoner_cfg = cfg.get("reasoner", None)
            if reasoner_cfg is None:
                raise ValueError(
                    "architecture 'latent_ar' requires a model.reasoner config block"
                )
            max_hands = int(reasoner_cfg.get("max_hands", 2))
            if max_hands != 2:
                raise ValueError(
                    f"latent_ar batch contract is fixed to 2 hand slots; "
                    f"got reasoner.max_hands={max_hands}"
                )
            reasoner_dim = int(reasoner_cfg.get("dim", 512))
            self.reasoner = LatentReasoner(
                dim=reasoner_dim,
                depth=int(reasoner_cfg.get("depth", 4)),
                num_heads=int(reasoner_cfg.get("num_heads", 8)),
                mem_dim=bert_dim,
                pose_dim=self.pose_dim,
                num_thinking_tokens=int(reasoner_cfg.get("num_thinking_tokens", 6)),
                max_hands=max_hands,
                feedback_norm=bool(reasoner_cfg.get("feedback_norm", True)),
                mlp_ratio=float(reasoner_cfg.get("mlp_ratio", 4.0)),
                aux_probe=reasoner_cfg.get("aux_probe", None),
            )
            self.z_proj = nn.Linear(reasoner_dim, bert_dim)
            self.h_proj = nn.Linear(reasoner_dim, bert_dim)
            self.prev_proj = nn.Linear(reasoner_dim, bert_dim)
            self.hand_cond_noise_std = float(
                reasoner_cfg.get("hand_cond_noise_std", 0.1)
            )
            lw = cfg.get("loss_weights", {}) or {}
            self.loss_weights = {
                "flow": float(lw.get("flow", 1.0)),
                "presence": float(lw.get("presence", 0.5)),
                "side": float(lw.get("side", 0.5)),
                "probe": float(lw.get("probe", 0.1)),
            }
            self.flow_transformer = FlowMatchingTransformer(
                pose_dim=self.pose_dim,
                num_queries=1,
                dim=flow_dim,
                depth=flow_depth,
                num_heads=flow_heads,
                cond_dim=bert_dim,
                hand_interaction={"enabled": True, "cross_hand_attn": False},
            )
        else:
            hand_interaction = cfg.get("hand_interaction", None)
            self.flow_transformer = FlowMatchingTransformer(
                pose_dim=self.pose_dim,
                num_queries=2,
                dim=flow_dim,
                depth=flow_depth,
                num_heads=flow_heads,
                cond_dim=bert_dim,
                hand_interaction=hand_interaction,
            )

        cfg_settings = cfg.get("cfg", {}) or {}
        self.cfg_drop_prob = float(cfg_settings.get("drop_prob", 0.0))
        self.cfg_guidance_scale = float(cfg_settings.get("guidance_scale", 1.0))
        if self.cfg_drop_prob > 0.0:
            self.null_cond = nn.Parameter(torch.randn(1, 1, bert_dim) * 0.02)
        else:
            self.null_cond = None

        aff_cfg = cfg.get("affordance", {}) or {}
        self.use_affordance = bool(aff_cfg.get("enabled", False))
        if self.use_affordance:
            aff_hidden = int(aff_cfg.get("hidden_dim", 192))
            self.affordance_head = nn.Sequential(
                nn.Linear(bert_dim, aff_hidden),
                nn.GELU(),
                nn.Linear(aff_hidden, 1),
            )

    def encode_language(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode language instructions through BERT."""
        encoded = self.tokenizer(
            texts,
            padding="max_length" if self.lang_padding == "max_length" else True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        output = self.bert(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        return output.last_hidden_state, encoded["attention_mask"]

    def encode_condition(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode point cloud + language into fused condition tokens.

        Args:
            xyz: Point positions, shape (B, N, 3).
            rgb: Point colors, shape (B, N, 3).
            texts: Language instructions, length B.

        Returns:
            fused: Fused condition tokens, shape (B, L_total, bert_dim).
            key_padding_mask: Bool mask over fused, shape (B, L_total);
                True = language pad token. PC tokens and the affordance
                summary token are always valid (False).
        """
        pc_tokens = self.pc_encoder(xyz, rgb)
        pc_tokens = self.projector(pc_tokens)

        lang_features, lang_mask = self.encode_language(texts, xyz.device)

        fused = self.fusion(pc_tokens, lang_features, lang_mask)

        B = fused.shape[0]
        pc_valid = torch.zeros(
            B, pc_tokens.shape[1], dtype=torch.bool, device=fused.device
        )
        key_padding_mask = torch.cat([pc_valid, ~lang_mask.bool()], dim=1)

        if self.use_affordance:
            num_pc = pc_tokens.shape[1]
            pc_fused = fused[:, :num_pc]
            scores = self.affordance_head(pc_fused).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            summary = torch.einsum("bn,bnd->bd", weights, pc_fused)
            fused = torch.cat([fused, summary.unsqueeze(1)], dim=1)
            key_padding_mask = torch.cat(
                [key_padding_mask, key_padding_mask.new_zeros(B, 1)], dim=1
            )

        return fused, key_padding_mask

    def compute_loss(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        texts: list[str],
        gt_poses: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute conditional flow matching training loss.

        Samples a random timestep t ~ U(0,1), interpolates between noise
        and ground truth, and trains the model to predict the velocity
        field v(x_t, t) = x_1 - x_0.

        Args:
            xyz: Point positions, shape (B, N, 3).
            rgb: Point colors, shape (B, N, 3).
            texts: Language instructions, length B.
            gt_poses: Ground truth bimanual poses, shape (B, 2, pose_dim).

        Returns:
            Dictionary with 'loss' and 'loss_flow_matching'.
        """
        B = xyz.shape[0]
        device = xyz.device

        cond_tokens, cond_mask = self.encode_condition(xyz, rgb, texts)
        mem_mask = cond_mask if self.mask_pad_tokens else None

        if self.training and self.null_cond is not None and self.cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=device) < self.cfg_drop_prob
            if drop_mask.any():
                null = self.null_cond.expand(B, cond_tokens.shape[1], -1)
                cond_tokens = torch.where(drop_mask[:, None, None], null, cond_tokens)
                if mem_mask is not None:
                    # dropped rows are all null tokens -> fully valid
                    mem_mask = torch.where(
                        drop_mask[:, None], torch.zeros_like(mem_mask), mem_mask
                    )

        t = torch.rand(B, device=device)
        noise = torch.randn_like(gt_poses)

        t_expand = t[:, None, None].expand_as(gt_poses)
        x_t = (1.0 - t_expand) * noise + t_expand * gt_poses
        target_v = gt_poses - noise

        pred_v = self.flow_transformer(
            x_t, t, cond_tokens, memory_key_padding_mask=mem_mask
        )

        # per-element weighted MSE (flow_loss_weight is all-ones by default,
        # reducing to the plain MSE; up-weight translation/rotation via config).
        loss_fm = (self.flow_loss_weight * (pred_v - target_v) ** 2).mean()

        return {"loss": loss_fm, "loss_flow_matching": loss_fm}

    @torch.no_grad()
    def sample(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        texts: list[str],
        num_steps: int = 50,
    ) -> dict[str, torch.Tensor]:
        """Generate grasp poses via Euler ODE integration.

        Solves dx/dt = v(x, t, c) from t=0 to t=1 starting from Gaussian noise.

        Args:
            xyz: Point positions, shape (B, N, 3).
            rgb: Point colors, shape (B, N, 3).
            texts: Language instructions, length B.
            num_steps: Number of Euler integration steps.

        Returns:
            Dictionary with per-hand 'translation', 'rotation_6d', 'joints'.
        """
        B = xyz.shape[0]
        device = xyz.device

        cond_tokens, cond_mask = self.encode_condition(xyz, rgb, texts)
        mem_mask = cond_mask if self.mask_pad_tokens else None

        use_cfg = self.null_cond is not None and self.cfg_guidance_scale != 1.0
        if use_cfg:
            null_tokens = self.null_cond.expand(B, cond_tokens.shape[1], -1)

        x = torch.randn(B, 2, self.pose_dim, device=device)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((B,), t_val, device=device)
            v = self.flow_transformer(
                x, t, cond_tokens, memory_key_padding_mask=mem_mask
            )
            if use_cfg:
                # null tokens are all valid -> no padding mask
                v_uncond = self.flow_transformer(x, t, null_tokens)
                v = v_uncond + self.cfg_guidance_scale * (v - v_uncond)
            x = x + v * dt

        results = {}
        for i, hand in enumerate(["left", "right"]):
            pose = x[:, i]
            results[f"{hand}_translation"] = pose[:, : self.trans_dim]
            results[f"{hand}_rotation_6d"] = pose[
                :, self.trans_dim : self.trans_dim + self.rot_dim
            ]
            results[f"{hand}_joints"] = pose[:, self.trans_dim + self.rot_dim :]

        return results

    def compute_loss_latent_ar(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        texts: list[str],
        gt_poses: torch.Tensor,
        hand_mask: torch.Tensor,
        hand_side_ids: torch.Tensor,
        task_type_ids: torch.Tensor | None = None,
        log_l_obj: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training loss for the latent_ar architecture.

        Runs the reasoner teacher-forced over the canonical hand sequence and
        trains one flow-matching call per hand slot (independent timesteps),
        plus presence (BCE) and side (CE) decision losses.

        Args:
            xyz: Point positions, shape (B, N, 3).
            rgb: Point colors, shape (B, N, 3).
            texts: Language instructions, length B.
            gt_poses: GT poses in canonical hand order, shape (B, 2, pose_dim);
                padded slots are zero.
            hand_mask: Slot validity, shape (B, 2) bool.
            hand_side_ids: Hand side per slot (0=left, 1=right, -1=pad),
                shape (B, 2) long.
            task_type_ids: Optional task type labels, shape (B,); required
                when the aux probe is enabled.
            log_l_obj: Optional log object-scale targets, shape (B,); required
                when the aux probe is enabled.

        Returns:
            Dictionary with 'loss', 'loss_flow', 'loss_presence', 'loss_side'
            (and probe losses when enabled).
        """
        if hand_mask is None or hand_side_ids is None:
            raise ValueError(
                "compute_loss_latent_ar requires hand_mask and hand_side_ids"
            )
        B = xyz.shape[0]
        device = xyz.device
        hand_mask = hand_mask.bool()

        memory, memory_mask = self.encode_condition(xyz, rgb, texts)
        mem_mask = memory_mask if self.mask_pad_tokens else None

        gt_prev_pose = gt_poses[:, 0] + self.hand_cond_noise_std * torch.randn_like(
            gt_poses[:, 0]
        )
        gt_prev_side = hand_side_ids[:, 0].clamp(min=0)

        ro = self.reasoner.rollout_teacher_forced(
            memory, gt_prev_pose, gt_prev_side, mem_key_padding_mask=mem_mask
        )
        z_cond = self.z_proj(ro["z_tokens"])

        drop_mask = None
        if self.training and self.null_cond is not None and self.cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=device) < self.cfg_drop_prob

        prev_tok = self.prev_proj(
            self.reasoner.embed_prev_hand(gt_prev_side, gt_prev_pose)
        ).unsqueeze(1)

        weight = self.flow_loss_weight.view(1, self.pose_dim)
        mask_f = hand_mask.float()
        flow_sum = gt_poses.new_zeros(())
        for s in range(2):
            gt_s = gt_poses[:, s]
            t = torch.rand(B, device=device)
            noise = torch.randn_like(gt_s)
            x_t = (1.0 - t[:, None]) * noise + t[:, None] * gt_s
            target_v = gt_s - noise

            tokens = [memory, z_cond, self.h_proj(ro["hand_hidden"][:, s]).unsqueeze(1)]
            if s == 1:
                tokens.append(prev_tok)
            cond_s = torch.cat(tokens, dim=1)
            cond_mask_s = None
            if mem_mask is not None:
                # z/h/prev tokens appended after memory are all valid (False)
                cond_mask_s = torch.cat(
                    [
                        mem_mask,
                        mem_mask.new_zeros(B, cond_s.shape[1] - mem_mask.shape[1]),
                    ],
                    dim=1,
                )
            if drop_mask is not None and drop_mask.any():
                null = self.null_cond.expand(B, cond_s.shape[1], -1)
                cond_s = torch.where(drop_mask[:, None, None], null, cond_s)
                if cond_mask_s is not None:
                    # dropped rows are all null tokens -> fully valid
                    cond_mask_s = torch.where(
                        drop_mask[:, None], torch.zeros_like(cond_mask_s), cond_mask_s
                    )

            pred_v = self.flow_transformer(
                x_t,
                t,
                cond_s,
                hand_ids=hand_side_ids[:, s].clamp(min=0),
                memory_key_padding_mask=cond_mask_s,
            )
            per_sample = (weight * (pred_v - target_v) ** 2).mean(dim=-1)
            flow_sum = flow_sum + (per_sample * mask_f[:, s]).sum()

        loss_flow = flow_sum / hand_mask.sum().clamp(min=1)
        loss_presence = F.binary_cross_entropy_with_logits(
            ro["presence_logits"].float(), mask_f
        )
        loss_side = F.cross_entropy(
            ro["side_logits"][hand_mask].float(),
            hand_side_ids[hand_mask].clamp(min=0),
        )

        total = (
            self.loss_weights["flow"] * loss_flow
            + self.loss_weights["presence"] * loss_presence
            + self.loss_weights["side"] * loss_side
        )
        losses = {
            "loss_flow": loss_flow,
            "loss_presence": loss_presence,
            "loss_side": loss_side,
        }
        if self.reasoner.probe_enabled:
            if task_type_ids is None or log_l_obj is None:
                raise ValueError(
                    "aux_probe is enabled: task_type_ids and log_l_obj are required"
                )
            probe = self.reasoner.probe_forward(ro["z_tokens"])
            loss_probe_task = F.cross_entropy(
                probe["task_logits"].float(), task_type_ids
            )
            loss_probe_lobj = F.mse_loss(probe["log_l_obj"].float(), log_l_obj.float())
            total = total + self.loss_weights["probe"] * (
                loss_probe_task + loss_probe_lobj
            )
            losses["loss_probe_task"] = loss_probe_task
            losses["loss_probe_lobj"] = loss_probe_lobj
        losses["loss"] = total
        return losses

    @torch.no_grad()
    def sample_latent_ar(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        texts: list[str],
        num_steps: int = 10,
        presence_threshold: float = 0.5,
        force_sides: torch.Tensor | None = None,
        force_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Autoregressive sampling: thinking rollout, then per-hand emission.

        For each hand step: presence decides whether to emit (slot 0 is always
        kept), side is argmax with mutual-exclusion masking at step 1, the
        pose is sampled by per-hand Euler flow integration, and the sampled
        hand is embedded back into the sequence for the next step. Inactive
        samples still run every step (static shapes, no branching).

        Args:
            xyz: Point positions, shape (B, N, 3).
            rgb: Point colors, shape (B, N, 3).
            texts: Language instructions, length B.
            num_steps: Euler integration steps per hand.
            presence_threshold: Sigmoid threshold for emitting hand slot 1.
            force_sides: Optional forced side ids, shape (B, 2); used for
                forced-decision evaluation. Must be given together with
                force_mask (pad slots carry -1, which is only interpretable
                against the forced validity mask).
            force_mask: Optional forced slot validity, shape (B, 2) bool.

        Returns:
            Dictionary with 'poses' (B, 2, pose_dim), 'hand_mask' (B, 2) bool,
            'hand_sides' (B, 2) long (-1 = pad), plus diagnostics
            'presence_probs' (B, 2) and 'side_logits' (B, 2, 2).
        """
        if (force_sides is None) != (force_mask is None):
            raise ValueError(
                "force_sides and force_mask must be provided together: pad slots "
                "in force_sides hold -1, which is meaningless without the mask"
            )
        B = xyz.shape[0]
        device = xyz.device

        memory, memory_mask = self.encode_condition(xyz, rgb, texts)
        mem_mask = memory_mask if self.mask_pad_tokens else None
        z_tokens, seq_state = self.reasoner.rollout_thinking(
            memory, mem_key_padding_mask=mem_mask
        )
        z_cond = self.z_proj(z_tokens)

        use_cfg = self.null_cond is not None and self.cfg_guidance_scale != 1.0

        poses = torch.zeros(B, 2, self.pose_dim, device=device)
        out_mask = torch.zeros(B, 2, dtype=torch.bool, device=device)
        out_sides = torch.full((B, 2), -1, dtype=torch.long, device=device)
        presence_probs = torch.zeros(B, 2, device=device)
        all_side_logits = torch.zeros(B, 2, 2, device=device)

        prev_side: torch.Tensor | None = None
        prev_pose: torch.Tensor | None = None
        for s in range(2):
            h_s, presence_logit, side_logits, seq_state = self.reasoner.step_hand(
                seq_state, memory, s, prev_side, prev_pose,
                mem_key_padding_mask=mem_mask,
            )
            prob = torch.sigmoid(presence_logit.float())
            presence_probs[:, s] = prob
            all_side_logits[:, s] = side_logits.float()

            if force_mask is not None:
                mask_s = force_mask[:, s].bool()
            elif s == 0:
                mask_s = torch.ones(B, dtype=torch.bool, device=device)
            else:
                mask_s = prob > presence_threshold

            if force_sides is not None:
                side_s = force_sides[:, s].long().clamp(min=0)
            else:
                logits_s = side_logits.float()
                if s == 1:
                    logits_s = logits_s.scatter(
                        1, prev_side.unsqueeze(1), float("-inf")
                    )
                side_s = logits_s.argmax(dim=-1)

            tokens = [memory, z_cond, self.h_proj(h_s).unsqueeze(1)]
            if s == 1:
                tokens.append(
                    self.prev_proj(
                        self.reasoner.embed_prev_hand(prev_side, prev_pose)
                    ).unsqueeze(1)
                )
            cond_s = torch.cat(tokens, dim=1)
            cond_mask_s = None
            if mem_mask is not None:
                # z/h/prev tokens appended after memory are all valid (False)
                cond_mask_s = torch.cat(
                    [
                        mem_mask,
                        mem_mask.new_zeros(B, cond_s.shape[1] - mem_mask.shape[1]),
                    ],
                    dim=1,
                )
            if use_cfg:
                null_s = self.null_cond.expand(B, cond_s.shape[1], -1)

            x = torch.randn(B, self.pose_dim, device=device)
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t = torch.full((B,), step / num_steps, device=device)
                v = self.flow_transformer(
                    x, t, cond_s, hand_ids=side_s,
                    memory_key_padding_mask=cond_mask_s,
                )
                if use_cfg:
                    # null tokens are all valid -> no padding mask
                    v_uncond = self.flow_transformer(x, t, null_s, hand_ids=side_s)
                    v = v_uncond + self.cfg_guidance_scale * (v - v_uncond)
                x = x + v * dt

            poses[:, s] = torch.where(mask_s.unsqueeze(-1), x, torch.zeros_like(x))
            out_mask[:, s] = mask_s
            out_sides[:, s] = torch.where(mask_s, side_s, torch.full_like(side_s, -1))
            prev_side = side_s
            prev_pose = x

        return {
            "poses": poses,
            "hand_mask": out_mask,
            "hand_sides": out_sides,
            "presence_probs": presence_probs,
            "side_logits": all_side_logits,
        }

    def forward(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        texts: list[str],
        gt_poses: torch.Tensor | None = None,
        num_steps: int = 50,
        hand_mask: torch.Tensor | None = None,
        hand_side_ids: torch.Tensor | None = None,
        task_type_ids: torch.Tensor | None = None,
        log_l_obj: torch.Tensor | None = None,
        presence_threshold: float = 0.5,
        force_sides: torch.Tensor | None = None,
        force_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training mode (gt_poses provided): compute flow matching loss.
        Eval mode (gt_poses=None): sample grasp poses.
        Dispatches by cfg['architecture']: legacy_bimanual (default) | latent_ar.
        """
        if self.architecture == "latent_ar":
            if gt_poses is not None:
                if hand_mask is None or hand_side_ids is None:
                    raise ValueError(
                        "latent_ar training requires hand_mask and hand_side_ids"
                    )
                return self.compute_loss_latent_ar(
                    xyz,
                    rgb,
                    texts,
                    gt_poses,
                    hand_mask,
                    hand_side_ids,
                    task_type_ids=task_type_ids,
                    log_l_obj=log_l_obj,
                )
            return self.sample_latent_ar(
                xyz,
                rgb,
                texts,
                num_steps=num_steps,
                presence_threshold=presence_threshold,
                force_sides=force_sides,
                force_mask=force_mask,
            )
        if gt_poses is not None:
            return self.compute_loss(xyz, rgb, texts, gt_poses)
        return self.sample(xyz, rgb, texts, num_steps)
