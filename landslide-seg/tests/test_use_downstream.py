import pytest
import torch
from landslide_seg.models.factory import build_model
from landslide_seg.models.use_downstream import FeatureDecoder


@pytest.mark.parametrize('mode', ['linear', 'shallow', 'fpn'])
def test_feature_decoders_and_frozen_factory(mode):
    model = build_model({'model':dict(type='use', use_downstream=mode, freeze_use=True)})
    model.train()
    assert not model.use.encoder.training
    groups = model.param_groups()
    optimizer = torch.optim.AdamW([p for values in groups.values() for p in values])
    before = model.use.verify(optimizer)
    observation = dict(x=torch.randn(1,3,17,19),band_valid=torch.ones(1,3,dtype=torch.bool),
                       pixel_valid=torch.ones(1,3,17,19,dtype=torch.bool),band_metadata=torch.ones(1,3,3),
                       known_mask=torch.ones(1,3,3,dtype=torch.bool))
    assert model(observation)['logits'].shape == (1,1,17,19)
    assert model.use.verify(optimizer) == before
    assert all(id(p) not in {id(q) for q in model.use.parameters()} for g in groups.values() for p in g)
    with pytest.raises(ValueError,match='require frozen'):
        build_model({'model':dict(type='use',use_downstream=mode,freeze_use=False)})


def test_new_decoder_receives_gradients():
    model = FeatureDecoder('fpn')
    model(torch.randn(2,32,16,16))['logits'].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
