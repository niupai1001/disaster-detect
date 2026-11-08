"""Shared MemoryProjector: one P32 -> multi-slot K/V condition memory."""
import torch
from torch import nn


class SensorMemory(nn.Module):
    def __init__(self, *, d_p=32, hidden=64, slots=8, d_kv=32):
        super().__init__()
        if min(d_p, hidden, slots, d_kv) < 1:
            raise ValueError('Positive descriptor, hidden, slot and key widths required')
        self.d_p = d_p
        self.slots = slots
        self.d_kv = d_kv
        self.project = nn.Sequential(nn.Linear(d_p, hidden), nn.GELU(), nn.Linear(hidden, 2 * slots * d_kv))

    def forward(self, p):
        if p.ndim < 1 or p.shape[-1] != self.d_p:
            raise ValueError(f'Descriptor must end in {self.d_p} dimensions')
        if not p.is_floating_point() or not bool(torch.isfinite(p).all()):
            raise ValueError('Finite floating descriptor required')
        kv = self.project(p).reshape(*p.shape[:-1], 2, self.slots, self.d_kv)
        return kv[..., 0, :, :], kv[..., 1, :, :]
