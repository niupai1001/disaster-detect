"""ResNet-18 style spatial backbone with configurable input width.

Outputs feature maps at strides 4/8/16/32 so a segmentation head can use the
fine levels while DINO only needs the global embedding.
"""

import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetEncoder(nn.Module):
    """in_ch -> [b1, /4], then /8, /16, /32 stages."""

    def __init__(self, in_ch=64, widths=(64, 128, 256, 512), blocks=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            BasicBlock(in_ch, widths[0], stride=1),
            *[BasicBlock(widths[0], widths[0]) for _ in range(blocks[0] - 1)])
        self.layer2 = nn.Sequential(*[BasicBlock(widths[0], widths[1], stride=2)]
                                    + [BasicBlock(widths[1], widths[1])
                                       for _ in range(blocks[1] - 1)])
        self.layer3 = nn.Sequential(*[BasicBlock(widths[1], widths[2], stride=2)]
                                    + [BasicBlock(widths[2], widths[2])
                                       for _ in range(blocks[2] - 1)])
        self.layer4 = nn.Sequential(*[BasicBlock(widths[2], widths[3], stride=2)]
                                    + [BasicBlock(widths[3], widths[3])
                                       for _ in range(blocks[3] - 1)])
        self.out_channels = dict(s4=widths[0], s8=widths[1], s16=widths[2], s32=widths[3])

    def forward(self, x):
        f4 = self.stem(x)
        f8 = self.layer2(f4)
        f16 = self.layer3(f8)
        f32 = self.layer4(f16)
        return {"s4": f4, "s8": f8, "s16": f16, "s32": f32}
