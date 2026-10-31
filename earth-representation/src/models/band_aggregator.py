"""Band aggregation: masked mean baseline, optional light attention."""

import torch
import torch.nn as nn


class MaskedMean(nn.Module):
    """Average the per-band features, ignoring bands that are masked out."""

    def forward(self, features, band_mask=None):
        # features: [B, C, D, h, w]
        if band_mask is None:
            return features.mean(dim=1)
        w = band_mask.to(features.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1e-6)
        return (features * w).sum(dim=1) / denom


class DescriptorAttention(nn.Module):
    """Score each band from its descriptor, softmax over the band axis."""

    def __init__(self, dim, hidden=32):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, features, descriptors, band_mask=None):
        s = self.score(descriptors).squeeze(-1)          # [B, C]
        if band_mask is not None:
            s = s.masked_fill(~band_mask, float("-inf"))
        w = torch.softmax(s, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return (features * w).sum(dim=1)


def make_aggregator(kind, dim=64):
    if kind == "mean":
        return MaskedMean()
    if kind == "attention":
        return DescriptorAttention(dim)
    raise KeyError(f"unknown aggregator {kind}")
