"""DINO multi-crop: two global views plus k local views of one patch.

Works on CPU or CUDA tensors: the random generator must live on the same device
as the image, otherwise torch refuses to create the indices.
"""

import torch
import torch.nn.functional as F


def random_crop(x, size, rng):
    c, h, w = x.shape
    if h < size or w < size:
        x = F.interpolate(x.unsqueeze(0), size=(max(h, size), max(w, size)),
                          mode="bilinear", align_corners=False)[0]
        c, h, w = x.shape
    top = int(torch.randint(0, h - size + 1, (1,), generator=rng, device=x.device))
    left = int(torch.randint(0, w - size + 1, (1,), generator=rng, device=x.device))
    return x[:, top:top + size, left:left + size]


def flips(x, rng):
    if torch.rand(1, generator=rng, device=x.device) < 0.5:
        x = torch.flip(x, dims=[2])
    if torch.rand(1, generator=rng, device=x.device) < 0.5:
        x = torch.flip(x, dims=[1])
    k = int(torch.randint(0, 4, (1,), generator=rng, device=x.device))
    return torch.rot90(x, k, dims=[1, 2]) if k else x


def multi_crop(x, global_size=224, local_size=96, n_local=2, rng=None):
    """Return (global_views, local_views); every view keeps the band axis."""
    if rng is None:
        rng = torch.Generator(device=x.device).manual_seed(0)
    globals_ = [flips(random_crop(x, global_size, rng), rng) for _ in range(2)]
    locals_ = [flips(random_crop(x, local_size, rng), rng) for _ in range(n_local)]
    return globals_, locals_
