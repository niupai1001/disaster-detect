from __future__ import annotations


def center_crop_or_pad(skip, target):
    import torch
    from torch.nn import functional as F

    _, _, height, width = skip.shape
    target_height, target_width = target.shape[-2:]

    crop_top = max((height - target_height) // 2, 0)
    crop_left = max((width - target_width) // 2, 0)
    cropped = skip[:, :, crop_top : crop_top + min(height, target_height), crop_left : crop_left + min(width, target_width)]

    pad_height = target_height - cropped.shape[-2]
    pad_width = target_width - cropped.shape[-1]
    if pad_height > 0 or pad_width > 0:
        cropped = F.pad(cropped, [0, max(pad_width, 0), 0, max(pad_height, 0)])

    return cropped


def make_double_conv(in_channels: int, out_channels: int):
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def make_depthwise_separable_conv(in_channels: int, out_channels: int, *, dilation: int = 1):
    from torch import nn

    padding = dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
