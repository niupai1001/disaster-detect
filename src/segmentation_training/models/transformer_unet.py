from __future__ import annotations


def build_transformer_unet(
    *,
    input_channels: int,
    output_classes: int,
    base_channels: int = 32,
    transformer_layers: int = 1,
    num_heads: int = 2,
    token_grid: int = 16,
):
    import torch
    from torch import nn

    from .common import center_crop_or_pad, make_double_conv

    class TransformerBridge(nn.Module):
        def __init__(self, channels):
            super().__init__()
            heads = max(1, min(num_heads, channels))
            while channels % heads != 0:
                heads -= 1
            layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=heads,
                dim_feedforward=channels * 2,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, transformer_layers))
            self.token_grid = max(1, int(token_grid))

        def forward(self, x):
            batch, channels, height, width = x.shape
            grid_h = min(height, self.token_grid)
            grid_w = min(width, self.token_grid)
            pooled = torch.nn.functional.adaptive_avg_pool2d(x, (grid_h, grid_w))
            sequence = pooled.flatten(2).transpose(1, 2)
            encoded = self.encoder(sequence)
            encoded = encoded.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)
            encoded = torch.nn.functional.interpolate(encoded, size=(height, width), mode="bilinear", align_corners=False)
            return x + encoded

    class TransformerUNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = base_channels
            self.down1 = make_double_conv(input_channels, c)
            self.pool1 = nn.MaxPool2d(2)
            self.down2 = make_double_conv(c, c * 2)
            self.pool2 = nn.MaxPool2d(2)
            self.bridge_conv = make_double_conv(c * 2, c * 4)
            self.bridge_transformer = TransformerBridge(c * 4)
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2 = make_double_conv(c * 4, c * 2)
            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1 = make_double_conv(c * 2, c)
            self.head = nn.Conv2d(c, output_classes, 1)

        def forward(self, x):
            d1 = self.down1(x)
            d2 = self.down2(self.pool1(d1))
            bridge = self.bridge_transformer(self.bridge_conv(self.pool2(d2)))
            u2 = self.up2(bridge)
            dec2 = self.dec2(torch.cat([u2, center_crop_or_pad(d2, u2)], dim=1))
            u1 = self.up1(dec2)
            dec1 = self.dec1(torch.cat([u1, center_crop_or_pad(d1, u1)], dim=1))
            return self.head(dec1)

    return TransformerUNet()
