from __future__ import annotations


def build_resunet(*, input_channels: int, output_classes: int, base_channels: int = 32):
    import torch
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
            )
            self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.body(x) + self.skip(x))

    class ResUNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = base_channels
            self.down1 = ResidualBlock(input_channels, c)
            self.pool1 = nn.MaxPool2d(2)
            self.down2 = ResidualBlock(c, c * 2)
            self.pool2 = nn.MaxPool2d(2)
            self.bridge = ResidualBlock(c * 2, c * 4)
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2 = ResidualBlock(c * 4, c * 2)
            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1 = ResidualBlock(c * 2, c)
            self.head = nn.Conv2d(c, output_classes, 1)

        def forward(self, x):
            d1 = self.down1(x)
            d2 = self.down2(self.pool1(d1))
            bridge = self.bridge(self.pool2(d2))
            u2 = torch.cat([self.up2(bridge), d2], dim=1)
            u1 = torch.cat([self.up1(self.dec2(u2)), d1], dim=1)
            return self.head(self.dec1(u1))

    return ResUNet()

