"""Mask support regression checks for v3 objectives."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from pretrain_use_v3 import gradient_error,spatial_error
from pretrain_use_v2 import task_view


def test_gradient_pair_rule():
    p=torch.arange(16.).reshape(1,1,4,4).requires_grad_(); t=torch.zeros_like(p)
    valid=torch.ones_like(p,dtype=torch.bool); hidden=torch.zeros_like(valid); hidden[:,:,1,1]=True
    loss,n=gradient_error(p,t,hidden,valid)
    assert n==4
    valid[:,:,1,2]=False
    _,n=gradient_error(p,t,hidden,valid)
    assert n==3
    assert torch.isfinite(spatial_error(p,t,hidden,valid))


def test_summary_visible_only():
    o=dict(x=torch.randn(1,3,17,19),pixel_valid=torch.ones(1,3,17,19,dtype=torch.bool),
           band_valid=torch.ones(1,3,dtype=torch.bool),band_metadata=torch.tensor([.5,.1,10.]).expand(1,3,3).clone(),
           known_mask=torch.ones(1,3,3,dtype=torch.bool))
    for task in ('spectral','fine','context'):
        view,hidden,summary=task_view(o,task,22,block=16,metadata_mode='known')
        changed={k:v.clone() for k,v in o.items()}; changed['x'][hidden]=999
        other,_,s=task_view(changed,task,22,block=16,metadata_mode='known')
        torch.testing.assert_close(view['x'],other['x']); torch.testing.assert_close(summary,s)
