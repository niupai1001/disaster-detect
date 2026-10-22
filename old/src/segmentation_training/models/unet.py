from __future__ import annotations


def build_unet(*, input_channels: int, output_classes: int, base_channels: int = 32):
    import torch
    from torch import nn

    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = base_channels
            self.down1 = DoubleConv(input_channels, c)
            self.pool1 = nn.MaxPool2d(2)
            self.down2 = DoubleConv(c, c * 2)
            self.pool2 = nn.MaxPool2d(2)
            self.bridge = DoubleConv(c * 2, c * 4)
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2 = DoubleConv(c * 4, c * 2)
            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1 = DoubleConv(c * 2, c)
            self.head = nn.Conv2d(c, output_classes, 1)

        def forward(self, x):
            d1 = self.down1(x)
            d2 = self.down2(self.pool1(d1))
            bridge = self.bridge(self.pool2(d2))
            u2 = torch.cat([self.up2(bridge), d2], dim=1)
            u1 = torch.cat([self.up1(self.dec2(u2)), d1], dim=1)
            return self.head(self.dec1(u1))

    return UNet()

