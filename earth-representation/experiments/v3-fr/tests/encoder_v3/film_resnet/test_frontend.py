from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from encoder_v3.film_resnet.frontend import Frontend, VisibleBand, factor_for, pack, unpack
from encoder_v3.latent_response.types import NativeGrid


def band(gsd=10., shape=(33,47), pattern="holes"):
    grid = NativeGrid((gsd,0.,-100.,0.,-gsd,100.),shape,"EPSG:32630")
    values = torch.randn(shape[1],shape[0]).T.requires_grad_()
    qa = torch.ones_like(values,dtype=torch.bool)
    qa[1::4,2::5] = False
    visible = qa.clone()
    visible[3::7] = False
    if pattern == "empty":
        visible.fill_(False)
    if pattern == "one":
        visible.fill_(False)
        visible[0,0] = True
    return values,qa,visible,grid


@pytest.mark.parametrize("gsd,factor",[(10.,4),(20.,2),(30.,1),(60.,1)])
@pytest.mark.parametrize("pattern",["holes","empty","one"])
def test_t01_t02_roundtrip(gsd,factor,pattern):
    values,qa,visible,grid = band(gsd,pattern=pattern)
    assert not values.is_contiguous()
    assert factor_for(grid)==factor
    p = pack(VisibleBand.from_native(values,qa,visible,grid))
    assert torch.equal(unpack(p.values,p),torch.where(visible,values,0.))
    assert torch.equal(unpack(p.visible,p),visible)
    assert torch.all(unpack(p.structural,p))
    assert int(p.structural.sum()) == values.numel() and not (p.visible & ~p.structural).any()
    assert torch.equal(p.values[:factor**2,0,0],torch.where(visible,values,0.)[:factor,:factor].flatten())
    altered = torch.where(visible,values,float("nan"))
    other = pack(VisibleBand.from_native(altered,qa,visible,grid))
    assert torch.equal(p.values,other.values)
    unpack(p.values,p).sum().backward()
    assert torch.equal(values.grad,visible.to(values.dtype))


def test_t03_t04_content_and_empty():
    frontend = Frontend()
    p = pack(VisibleBand.from_native(*band()))
    empty = pack(VisibleBand.from_native(*band(pattern="empty")))
    x = frontend.lift(p.values[None],p.visible[None],p.structural[None])[0]
    assert torch.equal(x[:16],p.values)
    assert torch.count_nonzero(frontend([empty],torch.randn(1,32)))==0
    values,qa,visible,grid = band()
    with pytest.raises(ValueError,match="subset"):
        VisibleBand.from_native(values,qa,torch.ones_like(visible),grid)


def test_t05_t06_film_identity_and_gradients():
    torch.manual_seed(17)
    enabled,disabled = Frontend(),Frontend(film=False)
    disabled.load_state_dict(enabled.state_dict())
    p = pack(VisibleBand.from_native(*band()))
    descriptors = torch.randn(2,32)
    out = enabled([p,p],descriptors)
    assert torch.equal(out,disabled([p,p],descriptors))
    assert torch.equal(out[0],out[1])
    out.square().mean().backward()
    for block in enabled.blocks:
        assert all(param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm()>0
                   for param in (block.conv1.weight,block.conv2.weight,block.film.weight,block.scale))
        with torch.no_grad():
            block.film.weight.add_(block.film.weight.grad,alpha=-.01)
        assert not torch.equal(block.film(descriptors)[0],block.film(descriptors)[1])


def test_t07_cross_pack_dependency():
    torch.manual_seed(17)
    values,qa,visible,grid = band(shape=(45,51))
    visible = qa.clone()
    p = pack(VisibleBand.from_native(values,qa,visible,grid))
    front = Frontend()
    x = front([p],torch.zeros(1,32))
    g, = torch.autograd.grad(x[0,0,4,4], values)
    assert g[16:20,12:16].abs().max()>0
    assert torch.count_nonzero(g[:,36:])==0


@pytest.mark.parametrize("tile",[(3,5),(5,7),(7,2)])
def test_t08_halo_output_and_gradient(tile):
    torch.manual_seed(29)
    front = Frontend()
    p = pack(VisibleBand.from_native(*band()))
    other = replace(p, values=p.values*.7)
    desc = torch.randn(2,32,requires_grad=True)
    parameters = tuple(front.parameters())+(desc,)
    y = front([p,other],desc)
    weight = torch.randn_like(y)
    grad = torch.autograd.grad((y*weight).mean(), parameters)
    alternate = front([p,other],desc,tile=tile)
    other_grad = torch.autograd.grad((alternate*weight).mean(), parameters)
    torch.testing.assert_close(alternate,y,atol=1e-6,rtol=1e-5)
    for a,b in zip(grad,other_grad,strict=True):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
