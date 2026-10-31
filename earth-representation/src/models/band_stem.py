"""One shared single-band stem: band count never enters a weight shape."""

import torch
import torch.nn as nn


class BandStem(nn.Module):
    """1 -> width, downsample x4, applied to every band with the same weights."""

    def __init__(self, width=64, in_norm=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, width // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.width = width

    def forward(self, x):
        # x: [B, C, H, W] -> [B, C, width, H/4, W/4]
        b, c, h, w = x.shape
        f = self.net(x.reshape(b * c, 1, h, w))
        return f.reshape(b, c, self.width, h // 4, w // 4)


def descriptor_features(centre_nm, gsd_m, device=None):
    """Raw numeric description of a band: [.., 2] = (wavelength, log GSD).

    Kept as plain numbers here; the learned projection lives in the encoder so
    this stays replaceable when a product has no reliable wavelength.
    """
    centre = torch.as_tensor(centre_nm, dtype=torch.float32, device=device)
    gsd = torch.as_tensor(gsd_m, dtype=torch.float32, device=device)
    if centre.dim() == 0:
        centre = centre.unsqueeze(0)
    if gsd.dim() == 0:
        gsd = gsd.unsqueeze(0)
    return torch.stack([centre / 1000.0, torch.log(gsd) / 4.0], dim=-1)
