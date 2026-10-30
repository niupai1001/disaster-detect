import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/use_next_p2')]
from test_scale import asset
from landslide_seg.models.use_next_p2 import tasks
from landslide_seg.models.use_next_p2.encoder import ScaleEncoder
from landslide_seg.models.use_next_v2.masking import update_ema
from landslide_seg.models.use_next_p3.tasks import LatentTask


class LocalOnly(torch.nn.Module):
    def __init__(self,*args):
        super().__init__();torch.manual_seed(1234);self.local=ScaleEncoder('physical')
        self.teacher_local=copy.deepcopy(self.local).requires_grad_(False).eval()
    def forward(self,a,g,h):return self.local(a,hidden_cells=h,grid=g)
    def ema(self,m):update_ema(self.teacher_local,self.local,m)


@pytest.mark.parametrize('objective',['scale','latent','joint'])
def test_objective_and_ema_boundaries(monkeypatch,objective):
    torch.set_num_threads(4);monkeypatch.setattr(tasks,'ScaleUSE',LocalOnly)
    model=LatentTask(None,objective=objective)
    before=copy.deepcopy(model.use.teacher_local.state_dict())
    a=asset();cal={'s2/red,nir':dict(mean=[0.,0.],std=[1.,1.])}
    loss,detail=model.loss(a,1234,5678,cal)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in model.use.teacher_local.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.use.local.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())==(objective!='latent')
    assert any(p.grad is not None for p in model.latent_predictor.parameters())==(objective!='scale')
    opt=torch.optim.AdamW(list(model.use.local.parameters())+model.task_parameters(),lr=.001)
    opt.step();model.after_optimizer(0,10000)
    assert any(not torch.equal(v,model.use.teacher_local.state_dict()[k]) for k,v in before.items())


def test_identical_shapes_and_initialization(monkeypatch):
    monkeypatch.setattr(tasks,'ScaleUSE',LocalOnly)
    first=LatentTask(None,objective='scale').state_dict()
    for objective in ('latent','joint'):
        other=LatentTask(None,objective=objective).state_dict()
        assert set(other)==set(first) and all(torch.equal(v,other[k]) for k,v in first.items())


def test_sar_latent_available_scale_absent(monkeypatch):
    monkeypatch.setattr(tasks,'ScaleUSE',LocalOnly)
    a=asset();a.modality='s1';a.channel_names=('vv','vh');a.radiometry='gamma0_linear'
    model=LatentTask(None,objective='joint')
    loss,detail=model.loss(a,1234,5678,{})
    assert torch.isfinite(loss) and not detail['scale_applied'] and detail['latent_applied']


def test_scale_numerics_match_p2_with_unused_native_paths_skipped(monkeypatch):
    monkeypatch.setattr(tasks,'ScaleUSE',LocalOnly)
    base=tasks.ScaleTask(None,'physical');current=LatentTask(None,objective='scale')
    cal={'s2/red,nir':dict(mean=[0.,0.],std=[1.,1.])}
    first,_=base.loss(asset(),1234,5678,cal);second,_=current.loss(asset(),1234,5678,cal)
    assert torch.equal(first,second)
    first.backward();second.backward()
    for (name,p),(other,q) in zip(base.use.local.named_parameters(),current.use.local.named_parameters()):
        assert name==other
        assert (p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad)


@pytest.mark.parametrize('count',[3,4,6,7,10,12])
def test_band_drop_integer_bounds(count):
    from landslide_seg.models.use_next_p3.tasks import band_view
    a=asset();a.image=a.image[:1].repeat(count,1,1);a.pixel_valid=a.pixel_valid[:1].repeat(count,1,1)
    a.channel_metadata=a.channel_metadata[:1].repeat(count,1);a.metadata_known=a.metadata_known[:1].repeat(count,1)
    a.channel_names=tuple(str(i) for i in range(count));a.native_resolution=(10.,)*count;a.effective_resolution=(None,)*count
    observed=False
    for seed in range(100):
        result=band_view(a,seed);dropped=count-len(result.image)
        if dropped:
            observed=True;assert .2<=dropped/count<=.4
    assert observed
