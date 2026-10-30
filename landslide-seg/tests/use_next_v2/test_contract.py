import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from landslide_seg.models.use_next_v2 import EOAsset, EOObservation
from landslide_seg.models.use_next_v2.physical import PhysicalLocal, area_grid
from landslide_seg.models.use_next_v2.masking import mask_asset,update_ema


def asset(sensor='s2', size=12, rotation=False):
    t = (10.,2. if rotation else 0.,500000.,1. if rotation else 0.,-10.,3000000.)
    return EOAsset(sensor,('red','nir'),torch.arange(2*size*size).reshape(2,size,size).float()/1000,
                   torch.tensor([[.665,.03,10.],[.842,.1,10.]]),torch.ones(2,3,dtype=torch.bool),
                   {'crs':'EPSG:32648','transform':t},None,None,torch.ones(2,size,size,dtype=torch.bool),
                   '2024-01-01','surface_reflectance',{})


@pytest.mark.parametrize('rotation',[False,True])
def test_area_conservation(rotation):
    a=asset(rotation=rotation)
    matrix,shape,transform,crs=area_grid(a)
    assert shape[0]%4==shape[1]%4==0
    determinant=abs(a.sample_grid['transform'][0]*a.sample_grid['transform'][4]-a.sample_grid['transform'][1]*a.sample_grid['transform'][3])
    # Local metric projection differs from UTM scale, but does not discard border area.
    assert abs(matrix.values().sum().item()/(12*12*determinant)-1)<.002


def test_missing_and_invalid():
    with pytest.raises(ValueError): EOObservation('x',None,(),'TRAIN',{}).validate()
    with pytest.raises(ValueError): PhysicalLocal()(asset('hr'))
    a=asset(); a.image[0,0,0]=float('nan'); a.pixel_valid[0,0,0]=False
    assert torch.isfinite(PhysicalLocal()(a).global_embedding).all()


def test_permutation_gradient_and_exact_resume():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    m=PhysicalLocal(); a=asset(); opt=torch.optim.AdamW(m.parameters(),lr=.0003)
    def step(model,optimizer):
        optimizer.zero_grad(); out=model(a).global_embedding
        loss=out.square().mean(); loss.backward(); optimizer.step()
        return loss.detach()
    step(m,opt)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
    state=copy.deepcopy(m.state_dict()); optim=copy.deepcopy(opt.state_dict()); rng=torch.get_rng_state()
    expected=step(m,opt)
    clone=PhysicalLocal(); clone.load_state_dict(state,strict=True)
    co=torch.optim.AdamW(clone.parameters(),lr=.0003); co.load_state_dict(optim); torch.set_rng_state(rng)
    assert torch.equal(expected,step(clone,co))
    assert all(torch.equal(v,clone.state_dict()[k]) for k,v in m.state_dict().items())
    reordered=copy.deepcopy(a)
    reordered.image=a.image.flip(0); reordered.pixel_valid=a.pixel_valid.flip(0)
    reordered.channel_metadata=a.channel_metadata.flip(0); reordered.metadata_known=a.metadata_known.flip(0)
    assert torch.allclose(m(a).global_embedding,m(reordered).global_embedding,atol=1e-6)


def test_pair_does_not_inherit_qualified():
    path=Path(__file__).resolve().parents[2]/'scripts/use_next_v2/audit_data.py'
    spec=importlib.util.spec_from_file_location('audit',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rows={k:{'role':'TRAIN','date':'2024-01-01'} for k in ('a','b')}
    p={'id':'p','role':'TRAIN','views':{'s2':'a','landsat':'b'},'qualified':True,'joint_valid_ratio':1}
    result=mod.pair_audit(p,rows)
    assert not result['eligible'] and 'change_screen_unverified' in result['reasons']


def test_unknown_metadata_cannot_affect_features():
    torch.set_num_threads(2)
    m=PhysicalLocal().eval(); a=asset()
    a.metadata_known.zero_()
    a.channel_metadata.zero_()
    expected=m(a).global_embedding
    a.channel_metadata.fill_(100)
    assert torch.equal(expected,m(a).global_embedding)


def test_common_grid_mask_and_ema():
    a=asset(); matrix,shape,t,crs=area_grid(a)
    grid={'crs':crs,'origin':(t[2],t[5]),'shape':shape}
    hidden=torch.zeros(shape,dtype=torch.bool); hidden[shape[0]//2,shape[1]//2]=True
    masked,pixels=mask_asset(a,hidden,grid)
    assert pixels.any() and not masked.pixel_valid[:,pixels].any()
    changed=copy.deepcopy(a); changed.image[:,pixels]=999
    changed,_=mask_asset(changed,hidden,grid)
    assert torch.equal(masked.image,changed.image)
    m=PhysicalLocal(); teacher=copy.deepcopy(m)
    with torch.no_grad():
        for p in m.parameters():p.add_(1)
    initial=copy.deepcopy(teacher.state_dict())
    update_ema(teacher,m,.996)
    for k,v in teacher.state_dict().items():
        assert torch.allclose(v,initial[k]*.996+m.state_dict()[k]*.004,atol=1e-6)
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())


def test_disk_optimizer_rng_cursor_resume(tmp_path):
    torch.set_num_threads(2); torch.manual_seed(1234)
    m=PhysicalLocal(); o=torch.optim.AdamW(m.parameters(),lr=.0003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=2)
    cursor=0; a=asset()
    def step():
        nonlocal cursor
        o.zero_grad(); target=torch.randn(192)
        loss=(m(a).global_embedding-target).square().mean()
        loss.backward(); o.step(); scheduler.step(); cursor+=1
        return loss.detach()
    step()
    checkpoint=tmp_path/'resume.pt'
    torch.save(dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=scheduler.state_dict(),
                    rng=torch.get_rng_state(),cursor=cursor),checkpoint)
    expected=step(); final=copy.deepcopy(m.state_dict()); final_opt=copy.deepcopy(o.state_dict())
    saved=torch.load(checkpoint,weights_only=True)
    m.load_state_dict(saved['model'],strict=True); o.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler']); torch.set_rng_state(saved['rng']); cursor=saved['cursor']
    assert torch.equal(expected,step()) and cursor==2
    assert all(torch.equal(v,final[k]) for k,v in m.state_dict().items())
    for k,v in final_opt['state'].items():
        for field,value in v.items():assert torch.equal(value,o.state_dict()['state'][k][field])


def test_label_grid_identity_and_geometry_roundtrip():
    from landslide_seg.models.use_next_v2 import FeatureLevel
    from landslide_seg.models.use_next_v2.geometry import to_label_grid
    from pyproj import Transformer
    a=asset(rotation=True)
    x=torch.arange(144).reshape(1,12,12).float()
    level=FeatureLevel(x,torch.ones(12,12,dtype=torch.bool),a.sample_grid['transform'],10.,10.,'s2',None,False,'EPSG:32648')
    mapped,valid=to_label_grid(level,level.transform,level.crs,(12,12))
    assert torch.allclose(mapped,x,atol=1e-4) and valid.all()
    _,_,t,local=area_grid(a)
    forward=Transformer.from_crs(level.crs,local,always_xy=True)
    reverse=Transformer.from_crs(local,level.crs,always_xy=True)
    for X,Y in ((500000,3000000),(500144,2999880)):
        xx,yy=reverse.transform(*forward.transform(X,Y))
        assert abs(xx-X)<.1 and abs(yy-Y)<.1
