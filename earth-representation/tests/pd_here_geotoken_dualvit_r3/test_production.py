import copy
import torch
from pd_here_geotoken_dualvit_r3.model import Layout,GeoTokenDualViT,masked_reconstruction_loss
from pd_here_geotoken_dualvit_r3.data import patchify,unpatchify,geometry


def test_native_roundtrip_and_alignment():
    for shape in ((12,12),(6,6)):
        x=torch.arange(16*shape[0]*shape[1]).reshape(4*shape[0],4*shape[1])
        assert torch.equal(unpatchify(patchify(x,shape)),x)
    row={'bands':[dict(band_key='B02',crs='EPSG:32657',transform=[10,0,10,0,-10,100]),dict(band_key='B05',crs='EPSG:32657',transform=[20,0,0,0,-20,100])]}
    layout,origin=geometry(row)
    assert origin==(20,100) and layout.native_shapes==((12,12),(6,6))


def test_ragged_padding_isolation():
    torch.manual_seed(17);layout=Layout(('a','b'),((4,4),(2,2)),(2,2))
    m=GeoTokenDualViT(layout,16,8);x=tuple(torch.rand(2,4,*s) for s in layout.native_shapes)
    q=tuple(torch.ones_like(v,dtype=torch.bool) for v in x)
    for valid in q:valid[1,3]=False
    keep=torch.tensor([[True,True,False,False],[True,False,False,False]])
    bands=torch.ones(2,2,dtype=torch.bool)
    out=m(x,q,keep,bands)
    assert out['valid'].tolist()==[[True,True],[True,False]]
    assert not out['prediction'][1,3].any()
    for i in range(2):
        solo=m(tuple(v[i:i+1] for v in x),tuple(v[i:i+1] for v in q),keep[i:i+1],bands[i:i+1])
        torch.testing.assert_close(out['prediction'][i:i+1],solo['prediction'],rtol=1e-5,atol=1e-6)


def test_global_parent_accumulation_update():
    torch.manual_seed(17);layout=Layout(('a','b'),((4,4),(2,2)),(2,2))
    m=GeoTokenDualViT(layout,16,8);other=copy.deepcopy(m)
    x=tuple(torch.rand(3,4,*s) for s in layout.native_shapes);q=tuple(torch.ones_like(v,dtype=torch.bool) for v in x)
    keep=torch.tensor([[True,False,False,False]]*3);bands=torch.ones(3,2,dtype=torch.bool)
    optimizers=[torch.optim.SGD(model.parameters(),lr=1e-3) for model in (m,other)]
    masked_reconstruction_loss(m(x,q,keep,bands)['prediction'],x,q,keep,bands,'A').backward()
    for sl in (slice(0,2),slice(2,3)):
        xx=tuple(v[sl] for v in x);qq=tuple(v[sl] for v in q);kk=keep[sl];bb=bands[sl]
        (masked_reconstruction_loss(other(xx,qq,kk,bb)['prediction'],xx,qq,kk,bb,'A')*len(kk)/3).backward()
    for p,r in zip(m.parameters(),other.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,r.grad,rtol=1e-4,atol=2e-6)
    for o in optimizers:o.step()
    # SGD measures reduction equivalence without amplifying near-zero roundoff.
    for p,r in zip(m.parameters(),other.parameters()):torch.testing.assert_close(p,r,rtol=1e-5,atol=1e-7)

def test_native_offsets_with_coordinate_ramps_and_impulses():
    from pd_here_geotoken_dualvit_r3.evaluation import metric_records, reduce_records
    layout = Layout(('a','b'),((12,12),(6,6)),(4,4))
    values = []
    for j,(h,w) in enumerate(layout.native_shapes):
        yy,xx = torch.meshgrid(torch.arange(4*h),torch.arange(4*w),indexing='ij')
        image = (yy*2+xx+j*100).float()
        image[h,w] += 43
        values.append(patchify(image,(h,w))[None])
    qa = tuple(torch.ones_like(v,dtype=torch.bool) for v in values)
    keep = torch.zeros(1,16,dtype=torch.bool)
    keep[:,::4] = True
    bands = torch.ones(1,2,dtype=torch.bool)
    prediction = torch.cat([v.flatten(2) for v in values],-1)
    records = metric_records(prediction,values,qa,keep,bands,'A')
    assert [r['gradient_mse'] for r in records] == [0.,0.]
    assert masked_reconstruction_loss(prediction,values,qa,keep,bands,'A') == 0
    altered = prediction.clone()
    altered[0,5,144] += 1.
    measured = metric_records(altered,values,qa,keep,bands,'A')
    assert measured[0]['mse'] == 0 and measured[1]['mse'] > 0
    assert measured[1]['gradient_mse'] > 0
    flat = tuple(torch.ones_like(v) for v in values)
    metrics,counts = reduce_records(metric_records(prediction,flat,qa,keep,bands,'A'), counts=True)
    assert 'gradient_amplitude' not in metrics and 'gradient_correlation' not in counts
    assert counts['gradient_mse'] == 1

def test_two_cross_patch_blocks_keep_hidden_values_isolated():
    layout = Layout(('a','b'),((4,4),(2,2)),(2,2))
    torch.manual_seed(17)
    one = GeoTokenDualViT(layout,16,8)
    torch.manual_seed(17)
    two = GeoTokenDualViT(layout,16,8,global_depth=2)
    for name,value in one.state_dict().items():
        torch.testing.assert_close(value,two.state_dict()[name],rtol=0,atol=0)
    values = tuple(torch.rand(2,4,*shape,requires_grad=True) for shape in layout.native_shapes)
    qa = tuple(torch.ones_like(x,dtype=torch.bool) for x in values)
    keep = torch.tensor([[True,False,False,False],[False,True,False,False]])
    visible_bands = torch.ones(2,2,dtype=torch.bool)
    output = two(values,qa,keep,visible_bands)
    assert output['encoded'].shape == (2,1,16)
    hidden_changed = tuple(torch.where(keep[:,:,None,None],x,99.) for x in values)
    torch.testing.assert_close(output['prediction'],two(hidden_changed,qa,keep,visible_bands)['prediction'],rtol=0,atol=0)
    output['prediction'].square().mean().backward()
    for x in values:
        assert not x.grad[~keep[:,:,None,None].expand_as(x)].any()
    assert two.extra_global_blocks[0].attention.in_proj_weight.grad.abs().sum()>0

def test_width_preserving_fusion_uses_global_patch_attention():
    layout = Layout(('a','b'),((4,4),(2,2)),(2,2))
    torch.manual_seed(17)
    model = GeoTokenDualViT(layout,16,8,global_depth=2,fusion_width=32).eval()
    assert isinstance(model.fuse,torch.nn.Linear)
    assert model.fuse.in_features == model.fuse.out_features == 32
    assert model.geographic_width == 32
    assert model.decoder_embed.in_features == 32
    values = tuple(torch.rand(2,4,*shape) for shape in layout.native_shapes)
    qa = tuple(torch.ones_like(x,dtype=torch.bool) for x in values)
    keep = torch.ones(2,4,dtype=torch.bool)
    bands = torch.ones(2,2,dtype=torch.bool)
    captured = {}
    hook = model.global_block.attention.register_forward_pre_hook(
        lambda module,args: captured.update(tokens=args[0].detach().clone()))
    output = model(values,qa,keep,bands)
    hook.remove()
    assert output['encoded'].shape == (2,4,32)
    tokens = captured['tokens']
    assert tokens.shape == (2,4,32)
    _,weights = model.global_block.attention(tokens,tokens,tokens,need_weights=True,average_attn_weights=False)
    assert weights.shape == (2,4,4,4)
    assert weights[0,:,1,0].sum()>0
    changed = [x.clone() for x in values]
    changed[0][0,0] += 1.
    trial = model(tuple(changed),qa,keep,bands)
    assert (output['encoded'][0,1]-trial['encoded'][0,1]).abs().max()>1e-6
    torch.testing.assert_close(output['encoded'][1],trial['encoded'][1],rtol=0,atol=0)

def test_projection_only_branches_preserve_qa_and_hidden_isolation():
    layout = Layout(('a','b'),((4,4),(2,2)),(2,2))
    model = GeoTokenDualViT(layout,16,8,global_depth=2,fusion_width=32,branch_transformer=False)
    assert isinstance(model.spectral_block,torch.nn.Identity)
    assert isinstance(model.spatial_block,torch.nn.Identity)
    values = tuple(torch.rand(2,4,*s,requires_grad=True) for s in layout.native_shapes)
    qa = tuple(torch.ones_like(x,dtype=torch.bool) for x in values)
    qa[0][0,0] = False
    keep = torch.tensor([[True,False,False,False],[False,True,False,False]])
    bands = torch.ones(2,2,dtype=torch.bool)
    original = model(values,qa,keep,bands)
    modified = tuple(torch.where(q & keep[:,:,None,None],v,torch.full_like(v,float('nan')))
                     for v,q in zip(values,qa,strict=True))
    torch.testing.assert_close(original['prediction'],model(modified,qa,keep,bands)['prediction'],rtol=0,atol=0)
    original['prediction'].square().mean().backward()
    for x,q in zip(values,qa,strict=True):
        assert not x.grad[~(q & keep[:,:,None,None])].any()
