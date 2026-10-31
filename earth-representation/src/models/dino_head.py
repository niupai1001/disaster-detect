"""DINO projection head: discarded at inference time."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=4096, hidden=1024, bottleneck=128, nlayers=3):
        super().__init__()
        # BatchNorm, not LayerNorm: the encoder features are dominated by one common
        # direction, and only a batch-level mean removal exposes the differences.
        layers = [nn.BatchNorm1d(in_dim), nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, bottleneck)]
        self.mlp = nn.Sequential(*layers)
        self.last = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck, out_dim, bias=False))

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1, p=2)
        return self.last(x)


