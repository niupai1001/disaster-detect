"""Band subsampling: simulate a sensor that does not carry these channels.

Dropped bands are zeroed and reported through the band mask, so the aggregator
ignores them instead of averaging an invented zero into the feature.
"""

import torch


def sample_band_mask(n_bands, rng, keep_min=3, drop_max=None):
    """Return a boolean [C] mask with at least keep_min bands kept."""
    drop_max = drop_max if drop_max is not None else max(1, n_bands // 2)
    drop = int(torch.randint(0, drop_max + 1, (1,), generator=rng))
    drop = min(drop, n_bands - keep_min)
    if drop <= 0:
        return torch.ones(n_bands, dtype=torch.bool)
    idx = torch.randperm(n_bands, generator=rng)[:drop]
    mask = torch.ones(n_bands, dtype=torch.bool)
    mask[idx] = False
    return mask


def apply_band_dropout(x, mask):
    """x: [C,H,W]; mask: [C] bool."""
    return x * mask.to(x.dtype).view(-1, 1, 1), mask
