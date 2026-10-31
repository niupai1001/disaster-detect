"""Spatial degradation: an approximate sensor PSF, not a generic resize.

The physical view is "the same ground seen by a coarser instrument": low-pass
first, then sample on the coarser grid. A plain bilinear resize (the generic
control) skips the low-pass, which is exactly the difference under test.

Accepts both [C,H,W] and [B,C,H,W].
"""

import torch
import torch.nn.functional as F


def _batched(x):
    return (x.unsqueeze(0), True) if x.dim() == 3 else (x, False)


def gaussian_kernel(sigma, channels, device, dtype=torch.float32):
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k = torch.outer(g, g)
    return k.expand(channels, 1, k.shape[0], k.shape[1]).contiguous()


def lowpass(x, sigma):
    """Separable Gaussian over the last two dims, reflection padding."""
    x4, squeeze = _batched(x)
    c = x4.shape[1]
    k = gaussian_kernel(sigma, c, x4.device, x4.dtype)
    pad = k.shape[-1] // 2
    y = F.conv2d(F.pad(x4, (pad, pad, pad, pad), mode="reflect"), k, groups=c)
    return y[0] if squeeze else y


def downsample_upsample(x, factor):
    """Sample at 1/factor of the resolution, then return to the original grid."""
    x4, squeeze = _batched(x)
    b, c, h, w = x4.shape
    small = F.interpolate(x4, scale_factor=1.0 / factor, mode="area")
    y = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    return y[0] if squeeze else y


def spatial_degrade(x, sigma, factor):
    """Physical order: blur, then resample."""
    return downsample_upsample(lowpass(x, sigma), factor)


def generic_resize(x, factor):
    """Control condition: resize without the low-pass."""
    x4, squeeze = _batched(x)
    b, c, h, w = x4.shape
    small = F.interpolate(x4, size=(max(1, int(h / factor)), max(1, int(w / factor))),
                          mode="bilinear", align_corners=False)
    y = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    return y[0] if squeeze else y
