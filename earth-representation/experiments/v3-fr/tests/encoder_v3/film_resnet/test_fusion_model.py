from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from encoder_v3.film_resnet.frontend import VisibleBand, pack
from encoder_v3.film_resnet.fusion import PhaseAwareFusion, add_attention_state
from encoder_v3.film_resnet.model import Options, split_field
from encoder_v3.film_resnet.position import PhysicalPosition, positioned_block
from encoder_v3.latent_response.types import NativeGrid
from encoder_v3.patch_latent.model import SpatialBlock
from encoder_v3.shared_mae.geometry import metric_geometry


TOL = dict(atol=1e-6,rtol=1e-5)


def grid(gsd,shape=(5,7),x=-43,y=18):
    return NativeGrid((gsd,0,x,0,-gsd,y),shape,"EPSG:32630")


@pytest.mark.parametrize("gsd",[10.,20.,30.])
def test_t09_independent_overlap(gsd):
    g = grid(gsd)
    other = grid(10.,(7,9),x=-49,y=23)
    geometry = metric_geometry([SimpleNamespace(grid=a) for a in (g,other)],40.)
    fusion = PhaseAwareFusion(0,False)
    mapping = fusion.area.get(g,geometry,2,"cpu")
    expected = torch.zeros(mapping.pixels,mapping.slots,dtype=torch.float64)
    for i in range(mapping.pixels):
        row,col = divmod(i,g.shape[1])
        x,y = g.affine[2]+col*gsd,g.affine[5]-(row+1)*gsd
        for j in range(mapping.slots):
            parent,slot = divmod(j,4)
            dy,dx = divmod(parent,geometry.shape[1])
            a = geometry.origin[0]+40*dx+20*(slot%2)
            b = geometry.origin[1]+40*dy+20*(slot//2)
            expected[i,j] = max(0,min(x+gsd,a+20)-max(x,a))*max(0,min(y+gsd,b+20)-max(y,b))
    actual = torch.zeros_like(expected)
    actual[mapping.source,mapping.destination] = mapping.area
    torch.testing.assert_close(actual,expected,atol=1e-8,rtol=1e-8)
    torch.testing.assert_close(actual.sum(1),torch.full_like(actual[:,0],gsd**2))
    y,x = torch.meshgrid(torch.arange(g.shape[0]),torch.arange(g.shape[1]),indexing="ij")
    signals = [torch.ones_like(x),x,y,(x>2).long(),(x<=2).long(),
               ((x==1)&(y==2)).long(),(x+y)%2,(x==2).long()]
    values = torch.stack([a.flatten() for a in signals]).double()
    keep = torch.ones_like(values,dtype=torch.bool)
    keep[:,2::4] = False
    area = expected[None]*keep[...,None]
    support = area.sum(1)
    truth = (values[...,None]*area).sum(1)/support.clamp_min(1e-30)
    result,rho = mapping.aggregate(values[...,None],keep)
    torch.testing.assert_close(result[...,0],truth,atol=1e-8,rtol=1e-8)
    torch.testing.assert_close(rho,support/400,atol=1e-8,rtol=1e-8)


def test_t10_phase_and_slot_reshape():
    g = grid(10.,(4,4),x=0,y=40)
    m = torch.ones(g.shape,dtype=torch.bool)
    x = torch.arange(16,dtype=torch.float32).reshape(4,4)
    model = PhaseAwareFusion(0,False)
    values = []
    for content in (x,x.flip(1)):
        p = pack(VisibleBand.from_native(content,m,m,g))
        features = model.frontend([p],torch.zeros(1,32))
        values.append(model.phase_features(p,features)[0,:,0,0,0])
        assert torch.equal(values[-1],content.flatten())
    assert not torch.equal(*values)
    g = metric_geometry([SimpleNamespace(grid=g)],40.)
    p = model([VisibleBand.from_native(x,m,m,g.grids[0])],torch.zeros(1,32),g)
    assert torch.equal(p.backbone_input().permute(0,2,3,1).reshape(-1,4,8),p.values)
    assert p.observed.all()


def test_t11_t12_channel_order_and_chunks():
    torch.manual_seed(17)
    g = grid(20.)
    geo = metric_geometry([SimpleNamespace(grid=g)],40.)
    values = torch.randn(9,*g.shape,requires_grad=True)
    keep = torch.rand_like(values)>.4
    keep[0].fill_(False)
    desc = torch.randn(9,32,requires_grad=True)
    fusion = PhaseAwareFusion()
    def forward(order,chunk):
        return fusion([VisibleBand.from_native(values[i],keep[i],keep[i],g) for i in order],
                      desc[order],geo,channel_chunk=chunk,edge_chunk=23)
    order = list(range(len(keep)))
    baseline = forward(order,99)
    p = (values,desc,*fusion.parameters())
    weight = torch.randn_like(baseline.values)
    gradient = torch.autograd.grad((baseline.values*weight).mean(),p,allow_unused=True)
    for chunk in (1,2,8):
        result = forward(order[::-1],chunk)
        torch.testing.assert_close(result.values,baseline.values,**TOL)
        assert torch.equal(result.observed,baseline.observed)
        for a,b in zip(gradient,torch.autograd.grad((result.values*weight).mean(),p,allow_unused=True),strict=True):
            if a is None:
                assert b is None
            else:
                torch.testing.assert_close(a,b,**TOL)
    assert torch.count_nonzero(gradient[0][~keep]) == 0
    assert torch.count_nonzero(gradient[1][0]) == 0
    empty = torch.zeros_like(keep[0])
    result = fusion([VisibleBand.from_native(values[0],empty,empty,g)],desc[:1],geo)
    assert not result.observed.any() and not result.values.any() and result.structural.any()


def test_t12_stable_empty_slots():
    score = torch.tensor([[900.,0.,1.],[-1000.,0.,2.],[1000.,0.,3.]],requires_grad=True)
    values = torch.randn(3,3,8,requires_grad=True)
    valid = torch.tensor([[True,False,True],[True,False,False],[False,False,True]])
    state = None
    for i in range(3):
        state = add_attention_state(state,score[i:i+1],values[i:i+1],valid[i:i+1])
    _,mass,total,observed = state
    actual = total/torch.where(observed,mass,1.)[...,None]
    safe = torch.where(observed,score.masked_fill(~valid,-torch.inf),0.)
    expected = (safe.softmax(0)[...,None]*values*valid[...,None]).sum(0)
    torch.testing.assert_close(actual,expected,**TOL)
    for a,b in zip(torch.autograd.grad(actual.sum(),(score,values)),
                   torch.autograd.grad(expected.sum(),(score,values)),strict=True):
        torch.testing.assert_close(a,b,**TOL)


def test_t14_t15_positions_and_partial_pool():
    g = grid(10.,(20,21))
    geo = metric_geometry([SimpleNamespace(grid=g)],40.)
    block = SpatialBlock(32,2,8,2,"conv","mit")
    mask = torch.ones(1,1,*geo.shape,dtype=torch.bool)
    mask[...,0,0] = False
    pos = PhysicalPosition(32)
    q,k,mass = pos(geo,mask,0,2)
    assert not torch.equal(q[1,0],q[1,1])
    assert mass[0,0,0,0]==.75 and mass[0,0,-1,-1]==.5
    qc,kc,_ = pos.coordinates(geo,mask,0,2)
    torch.testing.assert_close(kc[0,0],(qc[0,1]+qc[1,0]+qc[1,1])/3)
    x = torch.randn(1,32,*geo.shape,requires_grad=True)
    plain = block(x,mask)
    zero = positioned_block(block,x,mask,torch.zeros_like(q),torch.zeros_like(k),mass)
    torch.testing.assert_close(zero,plain,**TOL)
    actual = positioned_block(block,x,mask,q,k,mass)
    assert not torch.allclose(actual,plain)


def test_t18_decoder_phase_order():
    r,h,w = 4,3,5
    channels = torch.arange(8*r*r*h*w).reshape(1,8*r*r,h,w).float()
    native = F.pixel_shuffle(channels,r)
    actual = split_field(native,torch.ones(1,1,h,w,dtype=torch.bool),r)
    expected = channels.reshape(1,8,r*r,h,w).permute(0,1,3,4,2).flatten(2)
    assert torch.equal(actual,expected)


def test_unsupported_geometry_and_config():
    g = grid(10.)
    with pytest.raises(ValueError,match="north-up"):
        replace(g,affine=(10,1,0,0,-10,100))
    with pytest.raises(ValueError,match="metric CRS"):
        metric_geometry([SimpleNamespace(grid=a) for a in (g,replace(g,crs="EPSG:32631"))],40.)
    with pytest.raises(TypeError):
        Options(frontend="ordered",blocks=2,film=True,position_mode="none",
                terminal_expander="legacy_continuous",legacy_b1_detail=True,unexpected=True)


def test_common_fusion_initialization_with_and_without_resnet():
    torch.manual_seed(13018)
    without = PhaseAwareFusion(0,False).state_dict()
    torch.manual_seed(13018)
    with_resnet = PhaseAwareFusion(2,False).state_dict()
    assert set(without) < set(with_resnet)
    assert all(torch.equal(value,with_resnet[name]) for name,value in without.items())
