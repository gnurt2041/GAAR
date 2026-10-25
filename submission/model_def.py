"""Attribute classifier: a timm ViT backbone with a linear head or a region head.

This file is shared by inference (it is copied into the submission archive) and by
training (`train.py` imports it), so the checkpoint keys always match.

Linear head:  logits = W · avgpool(tokens) + b
Region head:  every attribute reads a fixed horizontal band of the token grid
              (head / upper body / lower body / full body), then applies its own
              linear weight vector.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

ATTRIBUTE_NAMES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short",
    "UpperBody-Color-Black", "UpperBody-Color-Blue", "UpperBody-Color-Brown",
    "UpperBody-Color-Green", "UpperBody-Color-Grey", "UpperBody-Color-Orange",
    "UpperBody-Color-Pink", "UpperBody-Color-Purple", "UpperBody-Color-Red",
    "UpperBody-Color-White", "UpperBody-Color-Yellow", "UpperBody-Color-Other",
    "LowerBody-Length-Short",
    "LowerBody-Color-Black", "LowerBody-Color-Blue", "LowerBody-Color-Brown",
    "LowerBody-Color-Green", "LowerBody-Color-Grey", "LowerBody-Color-Orange",
    "LowerBody-Color-Pink", "LowerBody-Color-Purple", "LowerBody-Color-Red",
    "LowerBody-Color-White", "LowerBody-Color-Yellow", "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag",
    "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat",
]


def get_backbone_normalization(backbone_name: str) -> tuple[tuple, tuple]:
    """Mean/std of the backbone's pre-training (read from timm metadata, no download)."""
    try:
        cfg = timm.get_pretrained_cfg(backbone_name)
        if cfg is not None and cfg.mean is not None and cfg.std is not None:
            return tuple(cfg.mean), tuple(cfg.std)
    except Exception:
        pass
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _region_of(name: str) -> int:
    if name.startswith(("Hair-", "Accessory-Hat", "Accessory-Glasses")):
        return 0
    if name.startswith("UpperBody-"):
        return 1
    if name.startswith("LowerBody-"):
        return 2
    return 3


class PARModel(nn.Module):
    def __init__(self, backbone_name: str, n_attr: int = 40, pretrained: bool = True,
                 drop_rate: float = 0.2, img_size: tuple | None = None, region_head: bool = False,
                 grad_checkpointing: bool = False, pretrained_path: str | None = None, **_unused):
        super().__init__()
        kw = {"img_size": tuple(img_size)} if img_size else {}
        # num_classes=0: timm returns the average-pooled feature vector
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, **kw)
        if pretrained_path:
            self._load_local_pretrained(pretrained_path)
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(feat_dim, n_attr)

        self.region_mode = bool(region_head)
        if self.region_mode:
            gh, gw = self.backbone.patch_embed.grid_size
            y = (torch.arange(gh).float() + 0.5) / gh
            sg = lambda v: torch.sigmoid(v / 0.04)
            # soft bands over the rows of the token grid: head, upper body, lower body, full body
            rows = torch.stack([sg(0.28 - y), sg(y - 0.12) * sg(0.62 - y), sg(y - 0.42), torch.ones(gh)])
            W = rows[:, :, None].expand(4, gh, gw).reshape(4, gh * gw)
            self.register_buffer("region_w", W / W.sum(1, keepdim=True), persistent=False)
            self.register_buffer("region_idx", torch.tensor([_region_of(n) for n in ATTRIBUTE_NAMES]),
                                 persistent=False)
            self.region_weight = nn.Parameter(torch.randn(n_attr, feat_dim) * 0.01)
            self.region_bias = nn.Parameter(torch.zeros(n_attr))

        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=True)

    def _load_local_pretrained(self, path: str) -> None:
        """Load backbone weights from a local file (for machines without internet)."""
        from timm.layers import resample_abs_pos_embed
        sd = {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
        pe, tgt = sd.get("pos_embed"), self.backbone.pos_embed
        if pe is not None and pe.shape != tgt.shape:
            n_pre = int(getattr(self.backbone, "num_prefix_tokens", 1))
            old = int(round((pe.shape[1] - n_pre) ** 0.5))
            sd["pos_embed"] = resample_abs_pos_embed(pe, new_size=tuple(self.backbone.patch_embed.grid_size),
                                                     old_size=(old, old), num_prefix_tokens=n_pre)
        res = self.backbone.load_state_dict(sd, strict=False)
        print(f"[pretrained_path] {path}: missing={list(res.missing_keys)} unexpected={list(res.unexpected_keys)}",
              flush=True)

    def pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if self.region_mode:
            tok = self.backbone.forward_features(x)[:, self.backbone.num_prefix_tokens:]       # (B, N, C)
            reg = torch.einsum("rn,bnc->brc", self.region_w.to(tok.dtype), tok)              # (B, 4, C)
            reg = self.dropout(reg)[:, self.region_idx]                                       # (B, A, C)
            logits = (reg * self.region_weight.to(reg.dtype)).sum(-1) + self.region_bias
            return (logits, reg.mean(1)) if return_features else logits
        feat = self.pooled_features(x)
        logits = self.head(self.dropout(feat))
        return (logits, feat) if return_features else logits


def build_from_config(mc: dict) -> PARModel:
    """Build an inference model from a model_config.json dict."""
    return PARModel(backbone_name=mc["backbone"], n_attr=mc["n_attr"], pretrained=False,
                    img_size=tuple(mc["image_size"]), region_head=bool(mc.get("region_head", False)))
