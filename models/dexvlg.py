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

from .pointnet2 import PointNet2Encoder
from .flow_matching import FlowMatchingTransformer

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
        padding: bool = True,
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

        max_len = max(len(ids) for ids in all_ids) if padding else max_length
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
        logger.warning(
            "Could not load pretrained '%s' (%s), using random init",
            pretrained_path,
            e,
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
        logger.warning(
            "Could not load ModernBERT '%s' (%s), using random BERT", pretrained_path, e
        )
        return _random_bert(hidden_dim)


def _load_language_model(backbone: str, model_name: str, hidden_dim: int):
    """Dispatch to the configured language backbone (BERT default, or ModernBERT)."""
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

        self.trans_dim = 3
        self.rot_dim = 6
        self.joint_dim = joint_dim
        self.pose_dim = self.trans_dim + self.rot_dim + joint_dim

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

        self.flow_transformer = FlowMatchingTransformer(
            pose_dim=self.pose_dim,
            num_queries=2,
            dim=flow_dim,
            depth=flow_depth,
            num_heads=flow_heads,
            cond_dim=bert_dim,
        )

    def encode_language(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode language instructions through BERT."""
        encoded = self.tokenizer(
            texts,
            padding=True,
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
    ) -> torch.Tensor:
        """Encode point cloud + language into fused condition tokens.

        Args:
            xyz: Point positions, shape (B, N, 3).
            rgb: Point colors, shape (B, N, 3).
            texts: Language instructions, length B.

        Returns:
            Fused condition tokens, shape (B, L_total, bert_dim).
        """
        pc_tokens = self.pc_encoder(xyz, rgb)
        pc_tokens = self.projector(pc_tokens)

        lang_features, lang_mask = self.encode_language(texts, xyz.device)

        return self.fusion(pc_tokens, lang_features, lang_mask)

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

        cond_tokens = self.encode_condition(xyz, rgb, texts)

        t = torch.rand(B, device=device)
        noise = torch.randn_like(gt_poses)

        t_expand = t[:, None, None].expand_as(gt_poses)
        x_t = (1.0 - t_expand) * noise + t_expand * gt_poses
        target_v = gt_poses - noise

        pred_v = self.flow_transformer(x_t, t, cond_tokens)

        loss_fm = nn.functional.mse_loss(pred_v, target_v)

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

        cond_tokens = self.encode_condition(xyz, rgb, texts)

        x = torch.randn(B, 2, self.pose_dim, device=device)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((B,), t_val, device=device)
            v = self.flow_transformer(x, t, cond_tokens)
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

    def forward(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        texts: list[str],
        gt_poses: torch.Tensor | None = None,
        num_steps: int = 50,
    ) -> dict[str, torch.Tensor]:
        """
        Training mode (gt_poses provided): compute flow matching loss.
        Eval mode (gt_poses=None): sample grasp poses.
        """
        if gt_poses is not None:
            return self.compute_loss(xyz, rgb, texts, gt_poses)
        return self.sample(xyz, rgb, texts, num_steps)
