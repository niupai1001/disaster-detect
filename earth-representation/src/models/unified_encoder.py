"""Unified encoder: any band count in, one spatial feature pyramid out.

The model never sees "S2" or "Landsat" as a special case; it sees a stack of
1-band images plus the descriptor of each band.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .band_aggregator import make_aggregator
from .band_stem import BandStem, descriptor_features
from .resnet_encoder import ResNetEncoder


class UnifiedEncoder(nn.Module):
    def __init__(self, width=64, aggregate="mean", desc_dim=32):
        super().__init__()
        self.stem = BandStem(width=width)
        # FiLM, not an additive bias: an additive descriptor term was ~17x larger
        # than the content features and made the encoder blind to the image.
        self.desc_proj = nn.Sequential(nn.Linear(2, desc_dim), nn.SiLU(),
                                       nn.Linear(desc_dim, 2 * width))
        nn.init.zeros_(self.desc_proj[-1].weight)
        nn.init.zeros_(self.desc_proj[-1].bias)
        self.aggregate_kind = aggregate
        self.agg = make_aggregator(aggregate, width)
        self.backbone = ResNetEncoder(in_ch=width)

    def forward(self, image, centre_nm, gsd_m, band_mask=None):
        """image: [B,C,H,W] float; centre_nm/gsd_m: [C] or [B,C]."""
        if image.dim() != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(image.shape)}")
        b, c, h, w = image.shape
        if torch.is_tensor(centre_nm) and centre_nm.dim() == 1:
            centre_nm = centre_nm.unsqueeze(0).expand(b, -1)
        if torch.is_tensor(gsd_m) and gsd_m.dim() == 1:
            gsd_m = gsd_m.unsqueeze(0).expand(b, -1)

        feats = self.stem(image)                       # [B,C,width,h/4,w/4]
        desc = torch.stack([descriptor_features(centre_nm[i], gsd_m[i], image.device)
                            for i in range(b)])        # [B,C,2]
        film = self.desc_proj(desc)                    # [B,C,2*width]
        gamma, beta = film.chunk(2, dim=-1)
        feats = feats * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) \
            + beta.unsqueeze(-1).unsqueeze(-1)

        if self.aggregate_kind == "attention":
            u = self.agg(feats, desc, band_mask)
        else:
            if band_mask is not None:
                band_mask = band_mask.to(image.device)
            u = self.agg(feats, band_mask)
        levels = self.backbone(u)
        levels["global"] = F.adaptive_avg_pool2d(levels["s32"], 1).flatten(1)
        return levels
