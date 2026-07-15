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
from utils.metrics import TASK_TYPES
from utils.rotation import rotation_6d_to_matrix

logger = logging.getLogger(__name__)


def _fibonacci_sphere(num_points: int) -> torch.Tensor:
    """Deterministic, near-uniform unit vectors on S^2 (Fibonacci lattice).

    A data-free codebook of candidate approach directions: index ``k`` is a
    fixed unit vector, identical on every run (no clustering, no data pass).
    Used by GRACE as the anchor set the reasoner classifies over.

    Args:
        num_points: Number of anchor directions M.

    Returns:
        Unit vectors, shape (M, 3).
    """
    i = torch.arange(num_points, dtype=torch.float32)
    z = 1.0 - 2.0 * (i + 0.5) / num_points
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = torch.pi * (3.0 - 5.0 ** 0.5) * i
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)


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
        # Number of PointNet++ tokens; the fused memory is [pc(num_pc), lang(, aff)]
        # so memory[:, :num_pc] are the PC tokens (used for GRACE contact pooling).
        self.num_pc = num_pc_tokens

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

        # Per-task scaling ON TOP of flow_loss_weight (T2). Config:
        #   model.flow_loss_task_scales: {lgbidex: {rotation: 2.0}, ...}
        # Component names map to pose dims: translation [0:3], rotation [3:9],
        # joint [9:9+joint_dim]. Rows follow utils/metrics.TASK_TYPES order
        # (left, right, lgbidex, bidex). Key absent -> all-ones table = no-op.
        self._component_slices = {
            "translation": slice(0, self.trans_dim),
            "rotation": slice(self.trans_dim, self.trans_dim + self.rot_dim),
            "joint": slice(self.trans_dim + self.rot_dim, self.pose_dim),
        }
        task_scales_cfg = cfg.get("flow_loss_task_scales", {}) or {}
        task_scale_table = torch.ones(len(TASK_TYPES), self.pose_dim)
        for task_name, comp_scales in task_scales_cfg.items():
            if task_name not in TASK_TYPES:
                raise ValueError(
                    f"flow_loss_task_scales: unknown task_type '{task_name}'; "
                    f"expected one of {TASK_TYPES}"
                )
            row = TASK_TYPES.index(task_name)
            for comp_name, scale in (comp_scales or {}).items():
                if comp_name not in self._component_slices:
                    raise ValueError(
                        f"flow_loss_task_scales[{task_name}]: unknown component "
                        f"'{comp_name}'; expected one of "
                        f"{tuple(self._component_slices)}"
                    )
                task_scale_table[row, self._component_slices[comp_name]] = float(scale)
        self.register_buffer("task_scale_table", task_scale_table, persistent=False)

        # Geodesic rotation auxiliary loss (T3). Config:
        #   model.rot_geodesic: {weight: 0.5, task_types: [lgbidex], t_min: 0.3}
        # Key absent or weight == 0 -> complete no-op. Empty task_types list
        # applies the loss to every task type.
        rg_cfg = cfg.get("rot_geodesic", {}) or {}
        self.rot_geo_weight = float(rg_cfg.get("weight", 0.0))
        self.rot_geo_t_min = float(rg_cfg.get("t_min", 0.3))
        rg_tasks = list(rg_cfg.get("task_types", []) or [])
        unknown_rg = sorted(set(rg_tasks) - set(TASK_TYPES))
        if unknown_rg:
            raise ValueError(
                f"rot_geodesic.task_types: unknown task_type(s) {unknown_rg}; "
                f"expected members of {TASK_TYPES}"
            )
        rot_geo_task_mask = (
            torch.tensor([t in rg_tasks for t in TASK_TYPES], dtype=torch.bool)
            if rg_tasks
            else torch.ones(len(TASK_TYPES), dtype=torch.bool)
        )
        self.register_buffer("rot_geo_task_mask", rot_geo_task_mask, persistent=False)
        self.rot_geo_needs_task_ids = bool(rg_tasks)

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

        # GRACE (Grounded Reasoning over Approach & Contact). The flag is read
        # for EVERY architecture so a mis-configured legacy model raises loudly
        # and `self.use_grace` is always defined. The heads/buffers/losses only
        # materialize inside the latent_ar block below (they need the reasoner
        # hidden state); default (no `model.grace`) -> use_grace False -> zero
        # new params, tokens, losses, or required batch fields (bit-identical).
        grace_cfg = cfg.get("grace", {}) or {}
        self.use_grace = bool(grace_cfg.get("enabled", False))
        if self.use_grace and self.architecture != "latent_ar":
            raise ValueError("model.grace.enabled requires architecture 'latent_ar'")

        # Joint bimanual flow denoising (A1). When enabled (latent_ar only),
        # all present hands of a sample are denoised together as multiple flow
        # queries sharing one timestep; queries interact through the flow
        # transformer's (self-/cross-hand) attention, which replaces the
        # prev-hand memory token. Default False = the sequential per-hand
        # behavior, bit-identical to before this switch existed.
        self.joint_hand_denoise = bool(cfg.get("joint_hand_denoise", False))
        if self.joint_hand_denoise and self.architecture != "latent_ar":
            raise ValueError(
                "model.joint_hand_denoise requires architecture 'latent_ar'"
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
            # cross_hand_attn is derived from joint_hand_denoise: the joint
            # path feeds both hands as queries that must attend to each other
            # (single-query calls degenerate to attending only to themselves);
            # the sequential path conditions hand 2 via the prev-hand memory
            # token instead and keeps it off.
            self.flow_transformer = FlowMatchingTransformer(
                pose_dim=self.pose_dim,
                num_queries=1,
                dim=flow_dim,
                depth=flow_depth,
                num_heads=flow_heads,
                cond_dim=bert_dim,
                hand_interaction={
                    "enabled": True,
                    "cross_hand_attn": self.joint_hand_denoise,
                },
            )

            # ── GRACE heads (gated by model.grace.enabled) ──────────────────
            # Three supervised, visualizable "thoughts" produced from the
            # reasoner's per-hand hidden state h_s and fed back as flow
            # condition tokens: WHERE (contact heatmap over PC tokens), HOW
            # (categorical approach direction on an S^2 codebook), RELATE
            # (inter-hand relative rotation). The flow OUTPUT space is unchanged.
            if self.use_grace:
                if self.joint_hand_denoise:
                    raise NotImplementedError(
                        "GRACE is wired for the sequential per-hand flow path; "
                        "combining it with joint_hand_denoise is not supported "
                        "yet (the joint flow-term/sampler token wiring is left "
                        "unverified on purpose). Set joint_hand_denoise: false."
                    )
                self.grace_num_anchors = int(grace_cfg.get("num_approach_anchors", 64))
                grace_hidden = int(grace_cfg.get("hidden_dim", 256))
                # WHERE: contact query pooling over the fused PC tokens.
                self.grace_q = nn.Linear(reasoner_dim, bert_dim)
                # HOW: categorical posterior over the S^2 approach codebook.
                self.grace_approach = nn.Sequential(
                    nn.Linear(reasoner_dim, grace_hidden),
                    nn.GELU(),
                    nn.Linear(grace_hidden, self.grace_num_anchors),
                )
                self.grace_approach_embed = nn.Linear(3, bert_dim)
                self.register_buffer(
                    "grace_anchors",
                    _fibonacci_sphere(self.grace_num_anchors),
                    persistent=False,
                )
                # RELATE: inter-hand relative wrist rotation R0^T R1 (rot6d).
                self.grace_rel_head = nn.Linear(2 * reasoner_dim, self.rot_dim)
                self.grace_rel_embed = nn.Linear(self.rot_dim, bert_dim)

                self.grace_condition = bool(grace_cfg.get("condition", True))
                self.grace_detach = bool(grace_cfg.get("detach_condition", False))
                self.grace_sigma_scale = float(grace_cfg.get("contact_sigma_scale", 0.3))
                self.grace_sigma_min = float(grace_cfg.get("contact_sigma_min", 0.01))
                self.grace_sigma_max = float(grace_cfg.get("contact_sigma_max", 0.05))
                self.grace_w = {
                    "contact": float(grace_cfg.get("w_contact", 0.5)),
                    "anchor": float(grace_cfg.get("w_anchor", 0.5)),
                    "approach": float(grace_cfg.get("w_approach", 0.25)),
                    "rel_rot": float(grace_cfg.get("w_rel_rot", 0.5)),
                }
                # Optional per-task gating (mirrors rot_geo_task_mask): [] = all
                # task types; only then is task_type_ids required (see below).
                g_tasks = list(grace_cfg.get("task_types", []) or [])
                unknown_g = sorted(set(g_tasks) - set(TASK_TYPES))
                if unknown_g:
                    raise ValueError(
                        f"grace.task_types: unknown task_type(s) {unknown_g}; "
                        f"expected members of {TASK_TYPES}"
                    )
                grace_task_mask = (
                    torch.tensor([t in g_tasks for t in TASK_TYPES], dtype=torch.bool)
                    if g_tasks
                    else torch.ones(len(TASK_TYPES), dtype=torch.bool)
                )
                self.register_buffer(
                    "grace_task_mask", grace_task_mask, persistent=False
                )
                self.grace_needs_task_ids = bool(g_tasks)
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
        return_centers: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode point cloud + language into fused condition tokens.

        Args:
            xyz: Point positions, shape (B, N, 3).
            rgb: Point colors, shape (B, N, 3).
            texts: Language instructions, length B.
            return_centers: When True, also return the PointNet++ token centers
                ``xyz3`` (B, num_pc, 3) in the raw object frame — the substrate
                GRACE grounds its contact target on. Default False keeps the
                legacy 2-tuple return bit-identical for all other callers.

        Returns:
            fused: Fused condition tokens, shape (B, L_total, bert_dim). The
                first ``num_pc`` tokens are the PC tokens (position-aligned with
                ``xyz3`` when requested).
            key_padding_mask: Bool mask over fused, shape (B, L_total);
                True = language pad token. PC tokens and the affordance
                summary token are always valid (False).
            xyz3 (only when return_centers): PC token centers, shape (B, num_pc, 3).
        """
        if return_centers:
            pc_tokens, xyz3 = self.pc_encoder(xyz, rgb, return_centers=True)
        else:
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

        if return_centers:
            return fused, key_padding_mask, xyz3
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

    def _grace_heads(
        self, hand_hidden: torch.Tensor, pc_tokens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """GRACE WHERE + HOW heads from the per-hand reasoner hidden states.

        Args:
            hand_hidden: Per-hand reasoner hiddens, shape (B, H, reasoner_dim).
            pc_tokens: Fused PC condition tokens, shape (B, num_pc, bert_dim).

        Returns:
            Dict with contact_logits (B, H, num_pc), contact_tok (B, H, bert_dim),
            approach_logits (B, H, M), a_hat (B, H, 3), approach_tok
            (B, H, bert_dim).
        """
        q = self.grace_q(hand_hidden)  # (B, H, bert_dim)
        scale = pc_tokens.shape[-1] ** 0.5
        contact_logits = torch.einsum("bhd,bnd->bhn", q, pc_tokens) / scale
        contact_tok = torch.einsum(
            "bhn,bnd->bhd", torch.softmax(contact_logits, dim=-1), pc_tokens
        )
        approach_logits = self.grace_approach(hand_hidden)  # (B, H, M)
        a_hat = F.normalize(
            torch.softmax(approach_logits, dim=-1) @ self.grace_anchors, dim=-1
        )  # (B, H, 3)
        approach_tok = self.grace_approach_embed(a_hat)  # (B, H, bert_dim)
        return {
            "contact_logits": contact_logits,
            "contact_tok": contact_tok,
            "approach_logits": approach_logits,
            "a_hat": a_hat,
            "approach_tok": approach_tok,
        }

    def _grace_rel(self, h_pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GRACE RELATE head: predicted inter-hand relative rotation (rot6d).

        Args:
            h_pair: Both hands' reasoner hiddens, shape (B, 2, reasoner_dim).

        Returns:
            rel_rot6d: (B, 6) predicted relative rotation R0^T R1 as 6D.
            rel_tok: (B, bert_dim) relative-rotation condition token.
        """
        r6 = self.grace_rel_head(torch.cat([h_pair[:, 0], h_pair[:, 1]], dim=-1))
        return r6, self.grace_rel_embed(r6)

    def _grace_losses(
        self,
        g: dict[str, torch.Tensor],
        rel_rot6d: torch.Tensor,
        xyz3: torch.Tensor,
        gt_poses: torch.Tensor,
        grasp_center: torch.Tensor,
        approach_dir: torch.Tensor,
        hand_mask: torch.Tensor,
        log_l_obj: torch.Tensor,
        task_type_ids: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """The four self-supervised GRACE losses (all masked to present hands).

        Targets are derived from GT geometry — no external labels. Legal under
        relquantile11 (rot6d is normalization pass-through) and frame-correct
        (grasp_center / approach_dir and xyz3 are both in raw object meters).

        Args:
            g: Output of :meth:`_grace_heads`, per-hand shape (B, 2, *).
            rel_rot6d: Predicted relative rotation, shape (B, 6).
            xyz3: PC token centers (raw object frame), shape (B, num_pc, 3).
            gt_poses: GT poses (rot6d pass-through), shape (B, 2, pose_dim).
            grasp_center: Self-derived contact centers (raw frame), shape (B, 2, 3).
            approach_dir: Self-derived unit approach dirs (raw frame), shape (B, 2, 3).
            hand_mask: Slot validity, shape (B, 2) bool.
            log_l_obj: Log object characteristic length, shape (B,).
            task_type_ids: Optional task ids, shape (B,); required iff
                grace_needs_task_ids.

        Returns:
            Dict with loss_contact / loss_anchor / loss_approach / loss_rel_rot.
        """
        B = xyz3.shape[0]
        if self.grace_needs_task_ids and task_type_ids is None:
            raise ValueError(
                "grace.task_types is set: compute_loss_latent_ar requires "
                "task_type_ids to gate the GRACE losses"
            )
        # per-hand weight = present AND (task allowed); pad slots -> 0.
        per_hand_w = hand_mask.float()  # (B, 2)
        if task_type_ids is not None:
            per_hand_w = per_hand_w * self.grace_task_mask[task_type_ids].float()[:, None]
        denom = per_hand_w.sum().clamp(min=1.0)

        # ── WHERE: soft-CE to a scale-aware Gaussian over the num_pc centers ──
        L = log_l_obj.exp()  # (B,)
        sigma = (self.grace_sigma_scale * L).clamp(
            self.grace_sigma_min, self.grace_sigma_max
        )  # (B,)
        d2 = ((xyz3[:, None, :, :] - grasp_center[:, :, None, :]) ** 2).sum(-1)  # (B,2,N)
        q = torch.softmax(
            -d2 / (2.0 * (sigma[:, None, None] ** 2) + 1e-12), dim=-1
        ).detach()  # stop-grad target (xyz3 carries encoder gradient) — audit FIX 3
        log_p = torch.log_softmax(g["contact_logits"], dim=-1)  # (B, 2, N)
        contact_ce = -(q * log_p).sum(-1)  # (B, 2)
        loss_contact = (contact_ce * per_hand_w).sum() / denom

        # ── HOW anchor: CE to the nearest S^2 codebook direction (fp32 argmax) ──
        anchor_label = (
            approach_dir.detach().float() @ self.grace_anchors.float().t()
        ).argmax(dim=-1)  # (B, 2)
        anchor_ce = F.cross_entropy(
            g["approach_logits"].reshape(B * 2, -1).float(),
            anchor_label.reshape(B * 2),
            reduction="none",
        ).reshape(B, 2)
        loss_anchor = (anchor_ce * per_hand_w).sum() / denom

        # ── HOW approach: 1 - cos to the GT direction. a_hat and approach_dir
        # are unit for PRESENT hands; padded slots carry approach_dir == 0 so the
        # dot is 0 -> loss 1 (FINITE — never a 0/0 NaN), then zeroed by
        # per_hand_w. Do NOT divide by ‖approach_dir‖ (audit FIX 1).
        approach_cos = (g["a_hat"] * approach_dir).sum(-1)  # (B, 2)
        loss_approach = ((1.0 - approach_cos) * per_hand_w).sum() / denom

        # ── RELATE: geodesic of predicted vs GT relative rotation R0^T R1 ──
        both = hand_mask[:, 0] & hand_mask[:, 1]
        if task_type_ids is not None:
            both = both & self.grace_task_mask[task_type_ids]
        both_f = both.float()
        r_pred = rotation_6d_to_matrix(rel_rot6d.float())  # (B, 3, 3)
        r0 = rotation_6d_to_matrix(gt_poses[:, 0, 3:9].float())
        r1 = rotation_6d_to_matrix(gt_poses[:, 1, 3:9].float())
        rel_gt = torch.matmul(r0.transpose(-1, -2), r1)
        trace = (
            torch.matmul(r_pred.transpose(-1, -2), rel_gt)
            .diagonal(dim1=-2, dim2=-1)
            .sum(-1)
        )
        geo = torch.acos(((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6))  # (B,)
        loss_rel_rot = (geo * both_f).sum() / both_f.sum().clamp(min=1.0)

        return {
            "loss_contact": loss_contact,
            "loss_anchor": loss_anchor,
            "loss_approach": loss_approach,
            "loss_rel_rot": loss_rel_rot,
        }

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
        grasp_center: torch.Tensor | None = None,
        approach_dir: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training loss for the latent_ar architecture.

        Runs the reasoner teacher-forced over the canonical hand sequence and
        trains one flow-matching call per hand slot (independent timesteps),
        plus presence (BCE) and side (CE) decision losses. With
        joint_hand_denoise the per-slot calls are replaced by ONE joint call
        per sample over all present hands (shared timestep, queries attend to
        each other; see _joint_flow_terms) — the reasoner losses are unchanged.

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

        xyz3 = None
        if self.use_grace:
            memory, memory_mask, xyz3 = self.encode_condition(
                xyz, rgb, texts, return_centers=True
            )
            if grasp_center is None or approach_dir is None or log_l_obj is None:
                raise ValueError(
                    "model.grace.enabled: compute_loss_latent_ar requires "
                    "grasp_center, approach_dir, and log_l_obj batch inputs"
                )
        else:
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

        # GRACE thoughts (WHERE/HOW/RELATE) from the reasoner hiddens: compute
        # the supervised losses, and build the flow condition bundle (detached
        # if configured; None when grace.condition is false = supervise-but-
        # don't-condition ablation).
        grace_cond = None
        grace_losses = None
        if self.use_grace:
            g = self._grace_heads(ro["hand_hidden"], memory[:, : self.num_pc])
            rel_rot6d, rel_tok = self._grace_rel(ro["hand_hidden"])
            grace_losses = self._grace_losses(
                g, rel_rot6d, xyz3, gt_poses, grasp_center, approach_dir,
                hand_mask, log_l_obj, task_type_ids,
            )
            if self.grace_condition:
                ct, at = g["contact_tok"], g["approach_tok"]
                if self.grace_detach:
                    ct, at, rel_tok = ct.detach(), at.detach(), rel_tok.detach()
                grace_cond = {"contact_tok": ct, "approach_tok": at, "rel_tok": rel_tok}

        drop_mask = None
        if self.training and self.null_cond is not None and self.cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=device) < self.cfg_drop_prob

        weight = self.flow_loss_weight.view(1, self.pose_dim)
        if task_type_ids is not None:
            # (B, pose_dim) per-task component scaling; all-ones without the
            # flow_loss_task_scales config key (no-op).
            weight = weight * self.task_scale_table[task_type_ids]

        use_rot_geo = self.rot_geo_weight > 0.0
        if use_rot_geo and self.rot_geo_needs_task_ids and task_type_ids is None:
            raise ValueError(
                "rot_geodesic.task_types is set: compute_loss_latent_ar "
                "requires task_type_ids to filter the geodesic loss"
            )

        mask_f = hand_mask.float()
        if self.joint_hand_denoise:
            flow_sum, rot_geo_sum, rot_geo_cnt = self._joint_flow_terms(
                gt_poses=gt_poses,
                hand_mask=hand_mask,
                hand_side_ids=hand_side_ids,
                memory=memory,
                mem_mask=mem_mask,
                z_cond=z_cond,
                hand_hidden=ro["hand_hidden"],
                weight=weight,
                drop_mask=drop_mask,
                use_rot_geo=use_rot_geo,
                task_type_ids=task_type_ids,
            )
        else:
            flow_sum, rot_geo_sum, rot_geo_cnt = self._sequential_flow_terms(
                gt_poses=gt_poses,
                mask_f=mask_f,
                hand_side_ids=hand_side_ids,
                memory=memory,
                mem_mask=mem_mask,
                z_cond=z_cond,
                hand_hidden=ro["hand_hidden"],
                gt_prev_side=gt_prev_side,
                gt_prev_pose=gt_prev_pose,
                weight=weight,
                drop_mask=drop_mask,
                use_rot_geo=use_rot_geo,
                task_type_ids=task_type_ids,
                grace_cond=grace_cond,
            )

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
        if use_rot_geo:
            # 0 when every slot is masked out (e.g. all t <= t_min this step)
            loss_rot_geo = rot_geo_sum / rot_geo_cnt.clamp(min=1.0)
            total = total + self.rot_geo_weight * loss_rot_geo
            losses["loss_rot_geo"] = loss_rot_geo
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
        if grace_losses is not None:
            total = (
                total
                + self.grace_w["contact"] * grace_losses["loss_contact"]
                + self.grace_w["anchor"] * grace_losses["loss_anchor"]
                + self.grace_w["approach"] * grace_losses["loss_approach"]
                + self.grace_w["rel_rot"] * grace_losses["loss_rel_rot"]
            )
            losses.update(grace_losses)
        losses["loss"] = total
        return losses

    def _sequential_flow_terms(
        self,
        gt_poses: torch.Tensor,
        mask_f: torch.Tensor,
        hand_side_ids: torch.Tensor,
        memory: torch.Tensor,
        mem_mask: torch.Tensor | None,
        z_cond: torch.Tensor,
        hand_hidden: torch.Tensor,
        gt_prev_side: torch.Tensor,
        gt_prev_pose: torch.Tensor,
        weight: torch.Tensor,
        drop_mask: torch.Tensor | None,
        use_rot_geo: bool,
        task_type_ids: torch.Tensor | None,
        grace_cond: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy per-slot flow terms: one flow call per hand slot with its own
        timestep; hand slot 1 sees hand slot 0 through the prev-hand token.

        When ``grace_cond`` is given, each slot's flow additionally cross-attends
        to the GRACE condition tokens in a FIXED order — ``[contact_s,
        approach_s]`` and, at slot 1, ``[prev, rel]`` — identical to the order
        the sampler rebuilds from the model's own predicted hiddens.

        Returns:
            (flow_sum, rot_geo_sum, rot_geo_cnt) accumulated over both slots.
        """
        B = gt_poses.shape[0]
        device = gt_poses.device

        prev_tok = self.prev_proj(
            self.reasoner.embed_prev_hand(gt_prev_side, gt_prev_pose)
        ).unsqueeze(1)

        rot_geo_sum = gt_poses.new_zeros(())
        rot_geo_cnt = gt_poses.new_zeros(())
        flow_sum = gt_poses.new_zeros(())
        for s in range(2):
            gt_s = gt_poses[:, s]
            t = torch.rand(B, device=device)
            noise = torch.randn_like(gt_s)
            x_t = (1.0 - t[:, None]) * noise + t[:, None] * gt_s
            target_v = gt_s - noise

            tokens = [memory, z_cond, self.h_proj(hand_hidden[:, s]).unsqueeze(1)]
            if grace_cond is not None:
                tokens += [
                    grace_cond["contact_tok"][:, s : s + 1],
                    grace_cond["approach_tok"][:, s : s + 1],
                ]
            if s == 1:
                tokens.append(prev_tok)
                if grace_cond is not None:
                    tokens.append(grace_cond["rel_tok"].unsqueeze(1))
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

            if use_rot_geo:
                # Geodesic angle between the flow-endpoint estimate
                # x1_hat = x_t + (1-t) * pred_v and the GT rotation. rot6d is
                # normalization pass-through (relquantile11), so the [3:9]
                # slice is the true rotation. float() guards acos under bf16.
                x1_hat = x_t + (1.0 - t[:, None]) * pred_v
                r_pred = rotation_6d_to_matrix(x1_hat[:, 3:9].float())
                r_gt = rotation_6d_to_matrix(gt_s[:, 3:9].float())
                trace = (
                    (r_pred.transpose(-1, -2) @ r_gt)
                    .diagonal(dim1=-2, dim2=-1)
                    .sum(-1)
                )
                geo = torch.acos(((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6))
                # mask = hand present x task-type filter x t-gate (x1_hat is
                # too noisy near t=0 for a useful rotation signal)
                geo_mask = mask_f[:, s] * (t > self.rot_geo_t_min).float()
                if task_type_ids is not None:
                    geo_mask = geo_mask * self.rot_geo_task_mask[task_type_ids].float()
                rot_geo_sum = rot_geo_sum + (geo * geo_mask).sum()
                rot_geo_cnt = rot_geo_cnt + geo_mask.sum()

        return flow_sum, rot_geo_sum, rot_geo_cnt

    def _joint_flow_terms(
        self,
        gt_poses: torch.Tensor,
        hand_mask: torch.Tensor,
        hand_side_ids: torch.Tensor,
        memory: torch.Tensor,
        mem_mask: torch.Tensor | None,
        z_cond: torch.Tensor,
        hand_hidden: torch.Tensor,
        weight: torch.Tensor,
        drop_mask: torch.Tensor | None,
        use_rot_geo: bool,
        task_type_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint flow terms (joint_hand_denoise): every present hand of a
        sample becomes one flow query, all queries share the sample's single
        timestep, and one flow forward denoises them together (queries
        interact via the flow transformer's attention). The prev-hand token
        is not used; instead BOTH present hand-step hiddens h_s join the
        cross-attention memory.

        The batch is split by hand count (1 vs 2 present hands) and each
        subgroup runs one forward with static query count — absent slots
        never become queries, so they receive no gradient by construction.

        Returns:
            (flow_sum, rot_geo_sum, rot_geo_cnt) accumulated over all present
            hands; normalization matches the sequential path (sum over
            present hands of the per-hand weighted MSE mean).
        """
        B = gt_poses.shape[0]
        device = gt_poses.device

        # one t per sample, shared by all of that sample's hands
        t = torch.rand(B, device=device)
        h_all = self.h_proj(hand_hidden)  # (B, 2, bert_dim)
        n_hands = hand_mask.sum(dim=1)
        weight_full = weight.expand(B, -1)  # (B, pose_dim)

        flow_sum = gt_poses.new_zeros(())
        rot_geo_sum = gt_poses.new_zeros(())
        rot_geo_cnt = gt_poses.new_zeros(())

        for n in (1, 2):
            sel = n_hands == n
            if not sel.any():
                continue
            idx = sel.nonzero(as_tuple=True)[0]
            b = idx.numel()
            if n == 2:
                gt_q = gt_poses[idx]  # (b, 2, pose_dim)
                sides_q = hand_side_ids[idx].clamp(min=0)  # (b, 2)
                h_q = h_all[idx]  # (b, 2, bert_dim)
            else:
                # single present slot (canonical packing puts it at slot 0;
                # gather by mask to stay contract-agnostic)
                slots = hand_mask[idx].float().argmax(dim=1, keepdim=True)  # (b, 1)
                gt_q = gt_poses[idx].gather(
                    1, slots[..., None].expand(-1, -1, self.pose_dim)
                )
                sides_q = hand_side_ids[idx].gather(1, slots).clamp(min=0)
                h_q = h_all[idx].gather(
                    1, slots[..., None].expand(-1, -1, h_all.shape[-1])
                )

            t_q = t[idx]
            noise = torch.randn_like(gt_q)
            x_t = (1.0 - t_q[:, None, None]) * noise + t_q[:, None, None] * gt_q
            target_v = gt_q - noise

            # memory = [fused pc+lang(+aff), z_cond, h_s of present hands]
            cond = torch.cat([memory[idx], z_cond[idx], h_q], dim=1)
            cond_mask = None
            if mem_mask is not None:
                # z/h tokens appended after memory are all valid (False)
                mm = mem_mask[idx]
                cond_mask = torch.cat(
                    [mm, mm.new_zeros(b, cond.shape[1] - mm.shape[1])], dim=1
                )
            if drop_mask is not None and drop_mask.any():
                dm = drop_mask[idx]
                null = self.null_cond.expand(b, cond.shape[1], -1)
                cond = torch.where(dm[:, None, None], null, cond)
                if cond_mask is not None:
                    # dropped rows are all null tokens -> fully valid
                    cond_mask = torch.where(
                        dm[:, None], torch.zeros_like(cond_mask), cond_mask
                    )

            pred_v = self.flow_transformer(
                x_t,
                t_q,
                cond,
                hand_ids=sides_q,
                memory_key_padding_mask=cond_mask,
            )
            per_hand = (weight_full[idx].unsqueeze(1) * (pred_v - target_v) ** 2).mean(
                dim=-1
            )  # (b, n)
            flow_sum = flow_sum + per_hand.sum()

            if use_rot_geo:
                # per-query geodesic on the flow-endpoint estimate; every
                # query is a present hand, so the mask is only the t-gate x
                # task-type filter (broadcast over the sample's queries).
                x1_hat = x_t + (1.0 - t_q)[:, None, None] * pred_v
                r_pred = rotation_6d_to_matrix(x1_hat[..., 3:9].float())
                r_gt = rotation_6d_to_matrix(gt_q[..., 3:9].float())
                trace = (
                    (r_pred.transpose(-1, -2) @ r_gt)
                    .diagonal(dim1=-2, dim2=-1)
                    .sum(-1)
                )  # (b, n)
                geo = torch.acos(((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6))
                geo_mask = (t_q > self.rot_geo_t_min).float()
                if task_type_ids is not None:
                    geo_mask = geo_mask * self.rot_geo_task_mask[
                        task_type_ids[idx]
                    ].float()
                geo_mask = geo_mask[:, None].expand_as(geo)
                rot_geo_sum = rot_geo_sum + (geo * geo_mask).sum()
                rot_geo_cnt = rot_geo_cnt + geo_mask.sum()

        return flow_sum, rot_geo_sum, rot_geo_cnt

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
        if self.joint_hand_denoise:
            return self._sample_latent_ar_joint(
                xyz,
                rgb,
                texts,
                num_steps=num_steps,
                presence_threshold=presence_threshold,
                force_sides=force_sides,
                force_mask=force_mask,
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
        prev_h: torch.Tensor | None = None
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
            # GRACE tokens rebuilt from the model's OWN predicted h_s (never GT)
            # in the SAME fixed order as _sequential_flow_terms: [contact,
            # approach] then, at slot 1, [prev, rel]. This is the train/sample
            # consistency guarantee.
            if self.use_grace:
                gs = self._grace_heads(h_s.unsqueeze(1), memory[:, : self.num_pc])
                tokens += [gs["contact_tok"][:, 0:1], gs["approach_tok"][:, 0:1]]
            if s == 1:
                tokens.append(
                    self.prev_proj(
                        self.reasoner.embed_prev_hand(prev_side, prev_pose)
                    ).unsqueeze(1)
                )
                if self.use_grace:
                    _, rel_tok = self._grace_rel(torch.stack([prev_h, h_s], dim=1))
                    tokens.append(rel_tok.unsqueeze(1))
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
            prev_h = h_s

        return {
            "poses": poses,
            "hand_mask": out_mask,
            "hand_sides": out_sides,
            "presence_probs": presence_probs,
            "side_logits": all_side_logits,
        }

    def _flow_euler(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cond_mask: torch.Tensor | None,
        hand_ids: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Euler-integrate the flow ODE from t=0 to t=1 (joint sampling path).

        Args:
            x: Initial noise, shape (B, pose_dim) or (B, n, pose_dim).
            cond: Condition tokens, shape (B, L, bert_dim).
            cond_mask: Optional bool pad mask over cond, shape (B, L).
            hand_ids: Hand side ids, shape (B,) or (B, n).
            num_steps: Euler integration steps.

        Returns:
            Integrated pose(s), same shape as x.
        """
        B = x.shape[0]
        use_cfg = self.null_cond is not None and self.cfg_guidance_scale != 1.0
        if use_cfg:
            null_tokens = self.null_cond.expand(B, cond.shape[1], -1)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((B,), step / num_steps, device=x.device)
            v = self.flow_transformer(
                x, t, cond, hand_ids=hand_ids, memory_key_padding_mask=cond_mask
            )
            if use_cfg:
                # null tokens are all valid -> no padding mask
                v_uncond = self.flow_transformer(x, t, null_tokens, hand_ids=hand_ids)
                v = v_uncond + self.cfg_guidance_scale * (v - v_uncond)
            x = x + v * dt
        return x

    def _extend_cond_mask(
        self, mem_mask: torch.Tensor | None, cond: torch.Tensor
    ) -> torch.Tensor | None:
        """Pad-mask over [memory, extra tokens]: extra tokens are all valid."""
        if mem_mask is None:
            return None
        return torch.cat(
            [
                mem_mask,
                mem_mask.new_zeros(cond.shape[0], cond.shape[1] - mem_mask.shape[1]),
            ],
            dim=1,
        )

    @torch.no_grad()
    def _sample_latent_ar_joint(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        texts: list[str],
        num_steps: int = 10,
        presence_threshold: float = 0.5,
        force_sides: torch.Tensor | None = None,
        force_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Joint sampling (joint_hand_denoise): decisions as usual, poses jointly.

        The reasoner decision chain is computed exactly as in the sequential
        path: thinking rollout, hand step 0, a single-query Euler pass for
        slot 0 conditioned on [memory, z, h_0] (this preliminary pose feeds
        the reasoner's step-1 prev-hand conditioning, mirroring the sequential
        sampler bit-exactly), then hand step 1. Presence/side outputs are
        therefore identical to the sequential path under the same weights and
        seed.

        Final poses: samples with only slot 0 present keep the preliminary
        pose (that IS the trained single-hand configuration); samples with
        both slots present are re-sampled from fresh noise in ONE joint Euler
        integration where both hands are flow queries attending to each other,
        conditioned on [memory, z, h_0, h_1].

        Args / Returns: same contract as sample_latent_ar.
        """
        B = xyz.shape[0]
        device = xyz.device

        memory, memory_mask = self.encode_condition(xyz, rgb, texts)
        mem_mask = memory_mask if self.mask_pad_tokens else None
        z_tokens, seq_state = self.reasoner.rollout_thinking(
            memory, mem_key_padding_mask=mem_mask
        )
        z_cond = self.z_proj(z_tokens)

        presence_probs = torch.zeros(B, 2, device=device)
        all_side_logits = torch.zeros(B, 2, 2, device=device)

        # ── decision chain (identical computation to the sequential path) ──
        h_0, presence_logit, side_logits, seq_state = self.reasoner.step_hand(
            seq_state, memory, 0, None, None, mem_key_padding_mask=mem_mask
        )
        presence_probs[:, 0] = torch.sigmoid(presence_logit.float())
        all_side_logits[:, 0] = side_logits.float()
        if force_mask is not None:
            mask_0 = force_mask[:, 0].bool()
        else:
            mask_0 = torch.ones(B, dtype=torch.bool, device=device)
        if force_sides is not None:
            side_0 = force_sides[:, 0].long().clamp(min=0)
        else:
            side_0 = side_logits.float().argmax(dim=-1)

        h0_tok = self.h_proj(h_0).unsqueeze(1)
        cond_0 = torch.cat([memory, z_cond, h0_tok], dim=1)
        x_0 = self._flow_euler(
            torch.randn(B, self.pose_dim, device=device),
            cond_0,
            self._extend_cond_mask(mem_mask, cond_0),
            side_0,
            num_steps,
        )

        h_1, presence_logit, side_logits, seq_state = self.reasoner.step_hand(
            seq_state, memory, 1, side_0, x_0, mem_key_padding_mask=mem_mask
        )
        prob_1 = torch.sigmoid(presence_logit.float())
        presence_probs[:, 1] = prob_1
        all_side_logits[:, 1] = side_logits.float()
        if force_mask is not None:
            mask_1 = force_mask[:, 1].bool()
        else:
            mask_1 = prob_1 > presence_threshold
        if force_sides is not None:
            side_1 = force_sides[:, 1].long().clamp(min=0)
        else:
            logits_1 = side_logits.float().scatter(
                1, side_0.unsqueeze(1), float("-inf")
            )
            side_1 = logits_1.argmax(dim=-1)

        poses = torch.zeros(B, 2, self.pose_dim, device=device)
        poses[:, 0] = torch.where(mask_0.unsqueeze(-1), x_0, torch.zeros_like(x_0))
        h1_tok = self.h_proj(h_1).unsqueeze(1)

        # ── joint denoising: both slots present -> 2 queries, one Euler pass ──
        both = mask_0 & mask_1
        if both.any():
            idx = both.nonzero(as_tuple=True)[0]
            cond_j = torch.cat(
                [memory[idx], z_cond[idx], h0_tok[idx], h1_tok[idx]], dim=1
            )
            sides_j = torch.stack([side_0[idx], side_1[idx]], dim=1)
            x_j = self._flow_euler(
                torch.randn(idx.numel(), 2, self.pose_dim, device=device),
                cond_j,
                self._extend_cond_mask(
                    mem_mask[idx] if mem_mask is not None else None, cond_j
                ),
                sides_j,
                num_steps,
            )
            poses[idx] = x_j

        # degenerate forced case {slot 1 only}: single-query pass on [mem, z, h_1]
        only_1 = mask_1 & ~mask_0
        if only_1.any():
            idx = only_1.nonzero(as_tuple=True)[0]
            cond_1 = torch.cat([memory[idx], z_cond[idx], h1_tok[idx]], dim=1)
            x_1 = self._flow_euler(
                torch.randn(idx.numel(), self.pose_dim, device=device),
                cond_1,
                self._extend_cond_mask(
                    mem_mask[idx] if mem_mask is not None else None, cond_1
                ),
                side_1[idx],
                num_steps,
            )
            poses[idx, 1] = x_1

        out_mask = torch.stack([mask_0, mask_1], dim=1)
        out_sides = torch.stack(
            [
                torch.where(mask_0, side_0, torch.full_like(side_0, -1)),
                torch.where(mask_1, side_1, torch.full_like(side_1, -1)),
            ],
            dim=1,
        )
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
        grasp_center: torch.Tensor | None = None,
        approach_dir: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training mode (gt_poses provided): compute flow matching loss.
        Eval mode (gt_poses=None): sample grasp poses.
        Dispatches by cfg['architecture']: legacy_bimanual (default) | latent_ar.
        ``grasp_center`` / ``approach_dir`` (B, 2, 3) are the self-supervised
        GRACE targets; required only when ``model.grace.enabled``.
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
                    grasp_center=grasp_center,
                    approach_dir=approach_dir,
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
