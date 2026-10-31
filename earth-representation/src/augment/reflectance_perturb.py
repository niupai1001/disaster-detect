"""Bounded reflectance perturbation, estimated from product differences.

Ranges are deliberately small and fixed in the config: this is not an RGB colour
jitter, it stands in for the residual calibration/harmonisation differences
between sensors. QA masks are never modified by it.
"""

import torch


def apply_gain_offset(x, rng, gain=0.05, offset=0.02):
    """Per-band multiplicative and additive jitter within the declared bounds."""
    c = x.shape[0]
    g = 1.0 + (torch.rand(c, 1, 1, generator=rng) * 2 - 1) * gain
    o = (torch.rand(c, 1, 1, generator=rng) * 2 - 1) * offset
    return x * g + o, {"gain": float(g.mean()), "offset": float(o.mean())}
