from __future__ import annotations


def build_unet_plus_plus(*, input_channels: int, output_classes: int, base_channels: int = 32):
    import torch
    from torch import nn

    from .common import make_double_conv

    class UNetPlusPlus(nn.Module):
        def __init__(self):
            super().__init__()
            c = base_channels
            self.x00 = make_double_conv(input_channels, c)
            self.x10 = make_double_conv(c, c * 2)
            self.x20 = make_double_conv(c * 2, c * 4)
            self.x01 = make_double_conv(c + c * 2, c)
            self.x11 = make_double_conv(c * 2 + c * 4, c * 2)
            self.x02 = make_double_conv(c * 2 + c * 2, c)
            self.pool = nn.MaxPool2d(2)
            self.head = nn.Conv2d(c, output_classes, 1)

        def upsample(self, x, like):
            return torch.nn.functional.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)

        def forward(self, x):
            x00 = self.x00(x)
            x10 = self.x10(self.pool(x00))
            x20 = self.x20(self.pool(x10))
            x01 = self.x01(torch.cat([x00, self.upsample(x10, x00)], dim=1))
            x11 = self.x11(torch.cat([x10, self.upsample(x20, x10)], dim=1))
            x02 = self.x02(torch.cat([x00, x01, self.upsample(x11, x00)], dim=1))
            return self.head(x02)

    return UNetPlusPlus()
