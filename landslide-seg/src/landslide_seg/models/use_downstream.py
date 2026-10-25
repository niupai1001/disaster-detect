"""Task-side adapters and probes; none of these weights belong to USE."""
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('bridge10', 'rgb3', 'patch32', 'fpn', 'linear', 'shallow', 'unet')


class FeatureDecoder(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ('fpn', 'linear', 'shallow'):
            raise ValueError('Unknown feature decoder')
        self.mode = mode
        self.load_info = {'initialization': 'random; no task or visual pretrained weights'}
        self.adapter_report = {'interface': f'USE32 to task-side {mode}', 'use_multiscale': False}
        if mode == 'linear':
            self.head = nn.Conv2d(32, 1, 1)
        elif mode == 'shallow':
            self.head = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))
        else:
            channels = (32, 64, 128, 128)
            self.stages = nn.ModuleList()
            previous = 32
            for index, width in enumerate(channels):
                self.stages.append(nn.Sequential(nn.Conv2d(previous, width, 3, stride=1 if index == 0 else 2, padding=1),
                                                 nn.GroupNorm(8, width), nn.GELU()))
                previous = width
            self.laterals = nn.ModuleList(nn.Conv2d(width, 64, 1) for width in channels)
            self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 1, 1))

    def forward(self, features, **kwargs):
        if self.mode != 'fpn':
            return {'logits': self.head(features)}
        pyramid = []
        x = features
        for stage, lateral in zip(self.stages, self.laterals):
            x = stage(x)
            pyramid.append(lateral(x))
        merged = pyramid[-1]
        for feature in reversed(pyramid[:-1]):
            merged = feature + F.interpolate(merged, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        return {'logits': self.head(merged)}

    def param_groups(self):
        return {'encoder': [], 'decoder': list(self.parameters()), 'new': []}


def build_visual(mode, pretrained_dir):
    """Return the task network and the number of latent adapter outputs it consumes."""
    if mode not in MODES:
        raise ValueError('Unknown frozen USE downstream mode')
    if mode == 'unet':
        from .use_unet import UNetFeatureDecoder
        return UNetFeatureDecoder(), 32
    if mode in ('fpn', 'linear', 'shallow'):
        return FeatureDecoder(mode), 32
    from .segformer import SegFormerSeg
    if mode == 'bridge10':
        return SegFormerSeg(adapter='direct10', backbone='b0', stem=True, pretrained_dir=pretrained_dir), 10
    channels = 3 if mode == 'rgb3' else 32
    visual = SegFormerSeg(adapter='direct10', backbone='b0', stem=True, pretrained_dir=pretrained_dir,
                          in_chans=channels, band_order=[f'latent_{i}' for i in range(channels)])
    embedding = visual.model.segformer.encoder.patch_embeddings[0]
    old = embedding.proj
    if mode == 'rgb3':
        # direct10's B,G,R column arrangement is restored to the original RGB order.
        with torch.no_grad():
            old.weight.copy_(old.weight[:, [2, 1, 0]].clone())
        description = 'learned 32-to-3 adapter, original RGB patch weights; latent channels are not physical RGB'
    else:
        embedding.proj = nn.Conv2d(channels, old.out_channels, old.kernel_size, old.stride, old.padding,
                                  bias=old.bias is not None)
        description = 'new randomly initialized 32-input patch embedding; later MiT stages pretrained'
    visual.adapter_report = {'interface': description, 'first_layer_initialization': 'pretrained RGB' if mode == 'rgb3' else 'random'}
    return visual, channels
