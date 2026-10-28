import copy
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from landslide_seg.models.use_v4_mit import USEv4, staged, load_visual


def obs(c=3,h=128,w=None,b=1):
    w=h if w is None else w
    return dict(x=torch.randn(b,c,h,w),pixel_valid=torch.ones(b,c,h,w,dtype=torch.bool))


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(4); torch.manual_seed(1234)
    return USEv4().eval()


@pytest.mark.parametrize('c',[1,3,6,10,16])
@pytest.mark.parametrize('h',[127,128,129,256])
def test_interface(model,c,h):
    with torch.no_grad(): f=model(obs(c,h,h+2))
    assert f['fine'].shape==(1,32,h,h+2)
    for d,s,x in zip((32,64,160,256),(4,8,16,32),f['context']):
        assert x.shape==(1,d,(h+s-1)//s,(h+2+s-1)//s)
        assert torch.isfinite(x).all()


def test_native_staged(model):
    encoder=load_visual().segformer.encoder.eval()
    for h in (127,128,129):
        x=torch.randn(1,3,h,h+2)
        with torch.no_grad():
            expected=encoder(x,output_hidden_states=True).hidden_states
            actual,_=staged(encoder,x)
            # Also compare the completely native staged route with the original API.
            z=x
            for i,(p,blocks,norm) in enumerate(zip(encoder.patch_embeddings,encoder.block,encoder.layer_norm)):
                t,hh,ww=p(z)
                for block in blocks: t=block(t,hh,ww)[0]
                z=norm(t).transpose(1,2).reshape(1,-1,hh,ww).contiguous()
                torch.testing.assert_close(z,expected[i],rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(actual[i],expected[i],rtol=1e-5,atol=1e-6)


def test_injection_and_gradients(model):
    counts={}; handles=[]
    for name in ('mit.patch_embeddings.0.proj','mit.patch_embeddings.0.layer_norm','adapter'):
        counts[name]=0
        def hook(m,a,o,key=name): counts[key]+=1
        handles.append(model.get_submodule(name).register_forward_hook(hook))
    model.zero_grad(); f=model(obs(),diagnostics=True)
    f['mit_input'].retain_grad()
    sum(x.square().mean() for x in f['context']).backward()
    for handle in handles: handle.remove()
    assert counts=={'mit.patch_embeddings.0.proj':0,'mit.patch_embeddings.0.layer_norm':1,'adapter':1}
    assert f['mit_input'].grad.abs().sum()>0
    assert model.adapter.weight.grad.abs().sum()>0
    assert model.readout.in_proj_weight.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in model.mit.parameters())
    assert not any(hasattr(model,n) for n in ('inject','stages','merges','projections','local'))
    model.train(); assert not any(m.training for m in model.mit.modules()); model.eval()


def test_seed_restore_and_accumulation(tmp_path):
    import pretrain_use_v4 as pt
    from landslide_seg.models.frozen_use import state_sha256
    a=pt.initialize(); b=pt.initialize()
    assert state_sha256(a)==state_sha256(b)
    oa=pt.optimizer(a); ob=pt.optimizer(b)
    torch.manual_seed(5); o=obs(b=2); meta=torch.zeros(2,3,3); known=meta.bool(); summary=torch.zeros(2,3,2)
    def step(m,opt):
        opt.zero_grad(); m(o,meta,known,summary).square().mean().backward(); opt.step()
    step(a,oa)
    pt.atomic_save(pt.checkpoint(a,oa,1,None,{'test':1},[]),tmp_path/'state.pt')
    step(a,oa); expected=torch.rand(3)
    pt.restore(b,ob,tmp_path/'state.pt',{'test':1}); step(b,ob)
    assert state_sha256(a)==state_sha256(b)
    assert torch.equal(expected,torch.rand(3))
    a.zero_grad(); b.zero_grad()
    a(o,meta,known,summary).square().mean().backward()
    for i in range(2):
        b({k:v[i:i+1] for k,v in o.items()},meta[i:i+1],known[i:i+1],summary[i:i+1]).square().mean().div(2).backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.grad is not None: torch.testing.assert_close(pa.grad,pb.grad,atol=2e-5,rtol=2e-4)
