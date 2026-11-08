"""Shared per-module, per-channel query from static catalog and native GSD metadata.

The Fourier frequency set is a required argument on purpose: I2 has not frozen it,
so no default can silently become part of a checkpoint identity.
"""
import math
import torch
from torch import nn
from .types import ChannelMeta

KINDS = ('conv', 'channel', 'decoder')


def fourier_features(value, frequencies):
    if value.ndim != 0:
        raise ValueError('One scalar metadata coordinate per call')
    phase = value * frequencies * (2 * math.pi)
    return torch.cat((phase.sin(), phase.cos()))


class ModuleQuery(nn.Module):
    def __init__(self, *, frequencies, d=32, hidden=64):
        super().__init__()
        frequencies = tuple(float(item) for item in frequencies)
        if not frequencies or any(not math.isfinite(item) or item <= 0 for item in frequencies):
            raise ValueError('Explicit positive Fourier frequencies required (I2 pending)')
        if min(d, hidden) < 1:
            raise ValueError('Positive query width required')
        self.register_buffer('frequencies', torch.tensor(frequencies))
        self.d = d
        self.seed = nn.ParameterDict({kind: nn.Parameter(torch.zeros(d)) for kind in KINDS})
        width = 2 * len(frequencies) + 2
        self.project = nn.ModuleDict(
            {kind: nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, d)) for kind in KINDS})

    def encode(self, meta, gsd_xy):
        if not isinstance(meta, ChannelMeta):
            raise ValueError('Typed ChannelMeta required; tensor indices are not channel identity')
        gx, gy = (float(item) for item in gsd_xy)
        if not (math.isfinite(gx) and math.isfinite(gy)) or gx <= 0 or gy <= 0:
            raise ValueError('Positive finite native GSD required')
        fraction = meta.canonical_ordinal + 0.5
        fraction = fraction / meta.catalog_total_channels
        position = self.frequencies.new_tensor(fraction)
        logs = self.frequencies.new_tensor([math.log(gx), math.log(gy)])
        return torch.cat((fourier_features(position, self.frequencies), logs))

    def forward(self, kind, meta, gsd_xy):
        if kind not in KINDS:
            raise ValueError(f'Unknown module kind {kind!r}; expected one of {KINDS}')
        return self.seed[kind] + self.project[kind](self.encode(meta, gsd_xy))
