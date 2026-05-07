from __future__ import annotations


def build_deeplabv3_plus(*, input_channels: int, output_classes: int, base_channels: int = 32):
    import torch
    from torch import nn

    from .common import make_depthwise_separable_conv, make_double_conv

    class ASPP(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)),
                    make_depthwise_separable_conv(in_channels, out_channels, dilation=2),
                    make_depthwise_separable_conv(in_channels, out_channels, dilation=4),
                ]
            )
            self.project = nn.Sequential(
                nn.Conv2d(out_channels * len(self.branches), out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))

    class DeepLabV3Plus(nn.Module):
        def __init__(self):
            super().__init__()
            c = base_channels
            self.low = make_double_conv(input_channels, c)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), make_depthwise_separable_conv(c, c * 2))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), make_depthwise_separable_conv(c * 2, c * 4))
            self.aspp = ASPP(c * 4, c * 2)
            self.low_project = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
            self.decoder = make_double_conv(c * 3, c)
            self.head = nn.Conv2d(c, output_classes, 1)

        def forward(self, x):
            low = self.low(x)
            features = self.down2(self.down1(low))
            context = self.aspp(features)
            context = torch.nn.functional.interpolate(context, size=low.shape[-2:], mode="bilinear", align_corners=False)
            decoded = self.decoder(torch.cat([context, self.low_project(low)], dim=1))
            return self.head(decoded)

    return DeepLabV3Plus()
