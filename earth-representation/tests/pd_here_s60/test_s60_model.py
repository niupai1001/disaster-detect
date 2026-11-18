from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from train_pd_here_s60 import NativeTargetAdapter
from pd_here_s60.model import DualAxisViT, inputs60, retile
from pd_here_geotoken_dualvit_r3.model import masked_reconstruction_loss
import torch
import pytest


def test_native_round_trip_and_mask_mapping():
    for side in (6,12):
        x=torch.arange(16*side**2).reshape(1,16,side,side)
        new=retile(x,4,8)
        assert len(new.flatten().unique())==x.numel()
        torch.testing.assert_close(retile(new,8,4),x)
        for i in range(16):
            mask=torch.zeros(1,16,dtype=torch.bool);mask[0,i]=True
            x60,q60,k60=inputs60((x,),(torch.ones_like(x,dtype=torch.bool),),mask)
            assert k60.sum()==4
            torch.testing.assert_close(x60[0][k60].flatten().sort().values,x[mask].flatten().sort().values)


@pytest.mark.parametrize('spatial_only',[False,True])
def test_attention_shapes_isolation_and_restore(spatial_only):
    torch.set_num_threads(4)
    torch.manual_seed(17)
    m=DualAxisViT(spatial_only=spatial_only).eval()
    names=('spatial',) if spatial_only else ('spatial','spectral','fusion')
    if spatial_only:
        assert not hasattr(m,'spectral_block') and not hasattr(m,'fusion_block')
    v=tuple(torch.rand(2,64,*s,requires_grad=True) for s in m.layout.native_shapes)
    q=tuple(torch.ones_like(x,dtype=torch.bool) for x in v)
    q[0][0,0,0,0]=False
    for task in ('A','B1'):
        keep=torch.ones(2,64,dtype=torch.bool)
        bands=torch.ones(2,10,dtype=torch.bool)
        if task=='A':keep[:,16:]=False
        else:bands[0,1]=False;bands[1,6]=False
        shapes={}
        hooks=[getattr(m,n+'_block').register_forward_pre_hook(
            lambda _,args,n=n:shapes.update({n:tuple(args[0].shape)})) for n in names]
        out=m(v,q,keep,bands)
        for hook in hooks:hook.remove()
        p,c=(16,10) if task=='A' else (64,9)
        expected={'spatial':(2,p,128)}
        if not spatial_only:expected.update(spectral=(2*p,c,128),fusion=(2*p,c+1,128))
        assert shapes==expected
        assert out['prediction'].shape==(2,64,198)
        active=[a & keep[:,:,None,None] & bands[:,j,None,None,None] for j,a in enumerate(q)]
        edited=tuple(torch.where(a,x,float('nan')) for x,a in zip(v,active,strict=True))
        torch.testing.assert_close(m(edited,q,keep,bands)['prediction'],out['prediction'],rtol=0,atol=0)
        for grad,a in zip(torch.autograd.grad(out['prediction'].square().mean(),v),active,strict=True):
            assert not grad[~a].any()
        if task=='A':
            torch.testing.assert_close(out['pre_position_decoder'][:,16:],m.mask_token.expand(2,48,-1))
        changed=tuple(torch.cat((x[:1],x[1:]+1),0) for x in v)
        torch.testing.assert_close(m(changed,q,keep,bands)['prediction'][0],out['prediction'][0],rtol=0,atol=0)


def test_qa_absent_source_and_zerohead_recovery():
    m=DualAxisViT([.2]*10)
    v=tuple(torch.rand(1,64,*s) for s in m.layout.native_shapes)
    q=tuple(torch.ones_like(x,dtype=torch.bool) for x in v)
    keep=torch.ones(1,64,dtype=torch.bool)
    bands=torch.ones(1,10,dtype=torch.bool);bands[:,0]=False
    for j in range(1,10):q[j][:,0]=False
    for x in q:x[:,1]=False
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    for step in range(2):
        out=m(v,q,keep,bands)
        assert out['valid'].sum()==62
        torch.testing.assert_close(out['pre_position_decoder'][:,:1],m.mask_token)
        assert not out['prediction'][:,1].any()
        opt.zero_grad()
        masked_reconstruction_loss(out['prediction'],v,q,keep,bands,'B1').backward()
        assert bool(m.spatial_values.weight.grad.abs().max()>0)==bool(step)
        opt.step()
    old=tuple(retile(x,8,4) for x in v)
    old_q=tuple(retile(x,8,4) for x in q)
    old_keep=torch.ones(1,16,dtype=torch.bool)
    new=m(v,q,keep,bands)['prediction']
    repacked=NativeTargetAdapter(m)(old,old_q,old_keep,bands)['prediction']
    torch.testing.assert_close(masked_reconstruction_loss(new,v,q,keep,bands,'B1'),
                               masked_reconstruction_loss(repacked,old,old_q,old_keep,bands,'B1'))
