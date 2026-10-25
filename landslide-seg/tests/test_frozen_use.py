import pytest
import torch
from torch import nn
from landslide_seg.models.frozen_use import FrozenUSE


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.register_buffer('runtime_cache', torch.zeros(1), persistent=False)

    def forward(self, observation):
        return {'features': self.bn(self.conv(observation['x']))}


def test_downstream_updates_preserve_all_use_state():
    frozen = FrozenUSE(Encoder())
    head = nn.Conv2d(2, 1, 1)
    model = nn.Sequential(frozen, head).train()
    optimizer = torch.optim.AdamW(head.parameters())
    before = frozen.verify(optimizer)
    for _ in range(3):
        optimizer.zero_grad()
        features = frozen({'x': torch.randn(2, 1, 4, 4, requires_grad=True)})['features']
        assert not features.requires_grad
        head(features).square().mean().backward()
        optimizer.step()
        assert frozen.verify(optimizer) == before
    assert not model[0].encoder.training


def test_parameter_buffer_and_optimizer_contamination_abort():
    for target in ('parameter', 'buffer', 'nonpersistent_buffer'):
        frozen = FrozenUSE(Encoder())
        with torch.no_grad():
            if target == 'parameter':
                frozen.encoder.conv.weight.add_(1)
            elif target == 'buffer':
                frozen.encoder.bn.num_batches_tracked.add_(1)
            else:
                frozen.encoder.runtime_cache.add_(1)
        with pytest.raises(RuntimeError, match='SHA256'):
            frozen.verify()
    frozen = FrozenUSE(Encoder())
    with pytest.raises(RuntimeError, match='Optimizer'):
        frozen.verify(torch.optim.SGD(frozen.parameters(), lr=.01))
    with pytest.raises(ValueError, match='labels'):
        frozen({'x': torch.randn(1, 1, 4, 4), 'task_mask': torch.zeros(4, 4)})


def test_production_use_freeze_groups_forward_and_checkpoint():
    from landslide_seg.models.segformer_use import SegFormerUSE

    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Conv2d(10, 1, 1)
            self.adapter_report, self.load_info = {}, {}

        def forward(self, x):
            return {'logits': self.head(x)}

        def param_groups(self):
            return {'new': list(self.head.parameters())}

    model = SegFormerUSE(Visual(), {'width': 8, 'hidden': 8})
    model.freeze_use()
    model.train()
    grouped = [p for values in model.param_groups().values() for p in values]
    optimizer = torch.optim.AdamW(grouped)
    reference = model.use.verify(optimizer)
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters() if p.requires_grad}
    for channels in (3, 4, 6, 10):
        observation = dict(x=torch.randn(1, channels, 4, 4),
                           band_valid=torch.ones(1, channels, dtype=torch.bool),
                           pixel_valid=torch.ones(1, channels, 4, 4, dtype=torch.bool),
                           band_metadata=torch.ones(1, channels, 3),
                           known_mask=torch.ones(1, channels, 3, dtype=torch.bool))
        output = model(observation)
        assert output['logits'].shape == (1, 1, 4, 4)
        reverse = {k: v.flip(1) for k, v in observation.items()}
        torch.testing.assert_close(model(reverse)['logits'], output['logits'], atol=1e-5, rtol=1e-4)
        assert model.use.verify(optimizer) == reference
    restored = SegFormerUSE(Visual(), {'width': 8, 'hidden': 8})
    restored.use.load_state_dict(model.use.encoder.state_dict(), strict=True)
    restored.freeze_use()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.probe_integrity()['use_sha256'] == reference
    # Reference comes from the independently selected SSL encoder, not the downstream file.
    with torch.no_grad():
        restored.use.encoder.fuse.weight.add_(1)
    with pytest.raises(RuntimeError, match='SHA256'):
        restored.probe_integrity()
