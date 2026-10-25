"""USE features supply both MiT and the original-grid detail branch."""
from torch import nn
from .universal_spectral_encoder import UniversalSpectralEncoder
from .frozen_use import FrozenUSE


class SegFormerUSE(nn.Module):
    def __init__(self, visual, use_config=None):
        super().__init__()
        self.visual = visual
        self.use = UniversalSpectralEncoder(**(use_config or {}))
        self.bridge = nn.Conv2d(self.use.width, 10, 1)
        self.adapter_report = {'interface': 'USE latent32 to latent10', 'visual': visual.adapter_report}
        self.load_info = visual.load_info

    def freeze_use(self):
        if not isinstance(self.use, FrozenUSE):
            self.use = FrozenUSE(self.use)

    def probe_integrity(self):
        if not isinstance(self.use, FrozenUSE):
            return None
        digest = self.use.verify()
        return dict(encoder_unchanged=True, encoder_has_grad=False,
                    encoder_training=False, use_sha256=digest)

    def forward(self, observation, **kwargs):
        encoded = self.use(observation)
        out = self.visual(self.bridge(encoded['features']), **kwargs)
        out['input_valid'] = encoded['valid']
        return out

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def param_groups(self):
        groups = self.visual.param_groups()
        groups['new'] = list(groups['new']) + list(self.use.parameters()) + list(self.bridge.parameters())
        return {key: [p for p in values if p.requires_grad] for key, values in groups.items()}

    def count_params(self):
        return {'total': sum(p.numel() for p in self.parameters()),
                'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)}
