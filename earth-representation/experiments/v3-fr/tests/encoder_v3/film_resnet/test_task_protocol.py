from types import SimpleNamespace

import torch

from encoder_v3.hierarchical_revised.tasks import task_masks
from encoder_v3.latent_response.types import NativeGrid
from encoder_v3.patch_latent.model import Config


def test_t26_frozen_task_order_and_repeatability():
    grid = NativeGrid((10.,0.,500000.,0.,-10.,3000000.),(40,44),"EPSG:32630")
    physical = SimpleNamespace(modality="optical")
    valid = torch.ones(grid.shape,dtype=torch.bool)
    channels = [SimpleNamespace(grid=grid,values=torch.zeros(grid.shape),valid=valid.clone(),physical=physical,name=name)
                for name in ("B02","B03","B04","B08")]
    # A and B1 select by immutable identity and exposure rather than global RNG state.
    sample = SimpleNamespace(identifier="v3_fr_synthetic_mask_protocol",channels=channels)
    config = Config()
    tasks = {}
    for kind in ("A","B1","B1","A"):
        actual = task_masks(sample,config,kind,17,0)
        if kind in tasks:
            for a,b in zip(sum(actual,[]),sum(tasks[kind],[]),strict=True):
                assert torch.equal(a,b)
        tasks[kind] = actual
        assert all(not (keep & target).any() and not (target & ~c.valid).any()
                   for keep,target,c in zip(*actual,channels,strict=True))
        torch.randn(73)
