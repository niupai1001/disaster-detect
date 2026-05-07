from __future__ import annotations


def build_attention_unet(*, input_channels: int, output_classes: int, base_channels: int = 32):
    import torch
    from torch import nn

    from .common import center_crop_or_pad, make_double_conv

    class AttentionGate(nn.Module):
        def __init__(self, gate_channels, skip_channels, inter_channels):
            super().__init__()
            self.gate = nn.Conv2d(gate_channels, inter_channels, 1, bias=False)
            self.skip = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
            self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter_channels, 1, 1), nn.Sigmoid())

        def forward(self, gate, skip):
            gate = torch.nn.functional.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return skip * self.psi(self.gate(gate) + self.skip(skip))

    class AttentionUNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = base_channels
            self.down1 = make_double_conv(input_channels, c)
            self.pool1 = nn.MaxPool2d(2)
            self.down2 = make_double_conv(c, c * 2)
            self.pool2 = nn.MaxPool2d(2)
            self.bridge = make_double_conv(c * 2, c * 4)
            self.att2 = AttentionGate(c * 4, c * 2, c)
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2 = make_double_conv(c * 4, c * 2)
            self.att1 = AttentionGate(c * 2, c, max(c // 2, 1))
            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1 = make_double_conv(c * 2, c)
            self.head = nn.Conv2d(c, output_classes, 1)

        def forward(self, x):
            d1 = self.down1(x)
            d2 = self.down2(self.pool1(d1))
            bridge = self.bridge(self.pool2(d2))
            a2 = self.att2(bridge, d2)
            u2 = self.up2(bridge)
            u2 = torch.cat([u2, center_crop_or_pad(a2, u2)], dim=1)
            dec2 = self.dec2(u2)
            a1 = self.att1(dec2, d1)
            u1 = self.up1(dec2)
            u1 = torch.cat([u1, center_crop_or_pad(a1, u1)], dim=1)
            return self.head(self.dec1(u1))

    return AttentionUNet()
