"""Selective scale consistency (Phase 4).

The idea under test: do not force high-resolution features to equal
downsampled-view features everywhere; only compare them on the ground support
both views actually observe, and weight by how much valid, low-frequency
content the coarser view retained.
"""

import torch
import torch.nn.functional as F


def pool_to_support(x, r):
    """Average over an r x r ground block: the scale both views observe."""
    return F.avg_pool2d(x, r, ceil_mode=True)


def block_weights(valid, r, hf_weight=None):
    """Weight per block: valid fraction x low-frequency retention.

    valid: [B,1,H,W] in {0,1}. hf_weight: optional [B,1,h,w] (1 = smooth).
    """
    v = F.avg_pool2d(valid.float(), r, ceil_mode=True)
    if hf_weight is None:
        return v
    return v * hf_weight


def high_frequency_weight(hi, r):
    """1 where the block is smooth, ->0 where detail dominates.

    Detail that a coarser view cannot see must not be demanded of it, which is
    the whole point of being selective about where consistency is enforced.
    """
    mean = pool_to_support(hi, r)
    var = pool_to_support(hi ** 2, r) - mean ** 2
    var = var.clamp_min(0)
    return 1.0 / (1.0 + var)


def scale_loss(feats_hi, feats_lo, r_list=(2, 4), valid=None, mode="selective",
               random_frac=0.5, generator=None):
    """Masked cosine consistency between two views of the same ground.

    mode: uniform | fixed | random | selective
    """
    total, weight_sum = 0.0, 0.0
    for r in r_list:
        a = pool_to_support(feats_hi, r)
        b = pool_to_support(feats_lo, r)
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode="bilinear",
                              align_corners=False)
        cos = F.cosine_similarity(a, b, dim=1).unsqueeze(1)
        if mode == "uniform":
            w = torch.ones_like(cos)
        elif mode == "fixed":
            w = torch.ones_like(cos) if r == 2 else torch.zeros_like(cos)
        elif mode == "random":
            keep = (torch.rand(cos.shape, device=cos.device, generator=generator)
                    < random_frac).float()
            w = keep
        elif mode == "selective":
            v = valid if valid is not None else torch.ones_like(cos)
            v = F.adaptive_avg_pool2d(v.float(), cos.shape[-2:])
            hf = F.adaptive_avg_pool2d(high_frequency_weight(feats_hi, r),
                                       cos.shape[-2:])
            w = v * hf
        else:
            raise KeyError(f"unknown scale mode {mode}")
        total = total + (w * (1.0 - cos)).sum()
        weight_sum = weight_sum + w.sum()
    return total / weight_sum.clamp_min(1e-6)

