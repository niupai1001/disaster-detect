import torch
import pytest
from torch import nn
from landslide_seg.models.use_unet import UNetFeatureDecoder
from landslide_seg.models.segformer_use import SegFormerUSE
from landslide_seg.models.frozen_use import state_sha256


def test_unet_updates_all_task_stages_without_changing_frozen_use():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    model=SegFormerUSE(UNetFeatureDecoder())
    model.bridge=nn.Identity()
    model.freeze_use();model.train()
    optimizer=torch.optim.AdamW([p for group in model.param_groups().values() for p in group],lr=6e-4)
    frozen=model.use.verify(optimizer)
    before=state_sha256(model.visual)
    observation=dict(x=torch.randn(2,4,33,35),band_valid=torch.ones(2,4,dtype=torch.bool),
        pixel_valid=torch.ones(2,4,33,35,dtype=torch.bool),band_metadata=torch.ones(2,4,3),
        known_mask=torch.ones(2,4,3,dtype=torch.bool))
    output=model(observation)['logits']
    assert output.shape==(2,1,33,35)
    loss=nn.functional.binary_cross_entropy_with_logits(output,torch.rand_like(output))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.visual.parameters())
    optimizer.step()
    assert state_sha256(model.visual)!=before
    assert model.use.verify(optimizer)==frozen
    assert all(p.grad is None for p in model.use.parameters())
    assert not model.use.training
    assert not any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.visual.modules())


def test_unet_rejects_raw_spectral_input_and_too_small_grid():
    model=UNetFeatureDecoder()
    with pytest.raises(ValueError,match='requires'):
        model(torch.randn(1,10,32,32))
    with pytest.raises(ValueError,match='requires'):
        model(torch.randn(1,32,15,32))
