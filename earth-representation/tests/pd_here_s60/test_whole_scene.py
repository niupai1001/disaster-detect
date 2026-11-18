from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from train_pd_here_s60_whole_scene import masks, objective
from pd_here_s60.whole_scene import WholeSceneViT, pack_scene, tile_bounds, tiles
import torch


def test_native_512_tiling_preserves_pixels_and_parent_loss():
    torch.set_num_threads(2)
    h,w=37,530
    a=torch.arange(h*w,dtype=torch.float32).reshape(h,w)/30000
    q=torch.ones_like(a,dtype=torch.bool);q[-1,-1]=False
    grid=SimpleNamespace(shape=(h,w),affine=(10.,0.,500000.,0.,-10.,3000000.),crs='EPSG:32632')
    bands=tuple(SimpleNamespace(key=k,values=a*c,valid=q,grid=grid) for k,c in [('red',1),('nir',2)])
    scene=pack_scene(SimpleNamespace(bands=bands),1.)
    assert len(tile_bounds(scene))==2
    view=masks(scene,'id',0)[1]
    items=list(tiles(scene,view[0]))
    for tile,_,_ in items:
        assert max(tile.grid)*6<=512
    assert sum(int(tile.qa.sum()) for tile,_,_ in items)==int(q.sum())*2
    all_indices=torch.cat([i for _,_,i in items])
    torch.testing.assert_close(all_indices.sort().values,torch.arange(len(scene.values)))
    model=WholeSceneViT({scene.layout:[0.,0.]})
    update=objective(model,scene,view,'B1',backward_scale=1.)
    hidden=int((~view[1]).nonzero().item())
    torch.testing.assert_close(update,bands[hidden].values[q].square().mean())
    assert model.projections['0'].output.weight.grad.abs().sum()>0


def test_old_dual_branch_equivalence_for_aligned_layout():
    from pd_here_geotoken_dualvit_r3.model import Layout,GeoTokenDualViT
    from pd_here_s60.whole_scene import Scene
    from pd_here_s60.whole_scene_dual import WholeSceneDualViT
    torch.manual_seed(17)
    geom=Layout(('red','nir'),((4,4),(2,2)),(2,2))
    old=GeoTokenDualViT(geom,decoder_width=128,global_depth=2,fusion_width=256).eval()
    key=(('red',4,4),('nir',2,2))
    new=WholeSceneDualViT({key:[0.,0.]}).eval()
    proj=new.projections['0']
    for name in ('spatial_values','spatial_qa','spatial_block','spectral_block','fuse','encoder_norm',
                 'decoder_embed','decoder_blocks','decoder_norm'):
        getattr(new,name).load_state_dict(getattr(old,name).state_dict())
    for name in ('spectral_reduce','spatial_reduce','availability','decoder_availability'):
        getattr(proj,name).load_state_dict(getattr(old,name).state_dict())
    proj.output.load_state_dict(old.prediction.state_dict())
    for i,(h,w) in enumerate(geom.native_shapes):
        proj.spectral_values[str(h*w)].load_state_dict(old.spectral_values[f'{h}x{w}'].state_dict())
        proj.spectral_qa[str(h*w)].load_state_dict(old.spectral_qa[f'{h}x{w}'].state_dict())
    new.global_blocks[0].load_state_dict(old.global_block.state_dict())
    new.global_blocks[1].load_state_dict(old.extra_global_blocks[0].state_dict())
    with torch.no_grad():
        new.mask_token.copy_(old.mask_token[0])
    values=tuple(torch.rand(1,4,h,w) for h,w in geom.native_shapes)
    qa=tuple(torch.ones_like(x,dtype=torch.bool) for x in values)
    qa[0][0,1,1,1]=False
    x=torch.cat([z.flatten(2)[0] for z in values],-1)
    q=torch.cat([z.flatten(2)[0] for z in qa],-1)
    scene=Scene((2,2),key,x,q,(0,0),int(q.sum()),120)
    for keep,bands in [(torch.tensor([True,False,True,False]),torch.tensor([True,True])),
                       (torch.ones(4,dtype=torch.bool),torch.tensor([True,False]))]:
        reference=old(values,qa,keep[None],bands[None])['prediction'][0]
        actual=new(scene,keep,bands)
        torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-5)
        active=q & keep[:,None] & bands[proj.band_slots]
        changed=Scene((2,2),key,torch.where(active,x,float('nan')),q,(0,0),int(q.sum()),120)
        torch.testing.assert_close(new(changed,keep,bands),actual,atol=0,rtol=0)


def test_scene_dual_axis_shared_init_local_fusion_and_hidden_values():
    from pd_here_s60.whole_scene import Scene
    from pd_here_s60.whole_scene_dual_axis import WholeSceneDualAxisViT
    layout=(('blue',2,2),('green',2,2),('nir',1,1))
    means={layout:[.1,.2,.3]}
    torch.manual_seed(17)
    spatial=WholeSceneViT(means)
    torch.manual_seed(17)
    dual=WholeSceneDualAxisViT(means)
    assert all(torch.equal(value,dual.state_dict()[key]) for key,value in spatial.state_dict().items())
    with torch.no_grad():
        dual.projections['0'].output.weight.normal_(std=.02)
    values=torch.rand(4,9,requires_grad=True)
    qa=torch.ones_like(values,dtype=torch.bool)
    qa[0,0]=False
    scene=Scene((2,2),layout,values,qa,(0,0),int(qa.sum()))
    for keep,bands,shape in [(torch.tensor([True,False,False,True]),torch.ones(3,dtype=torch.bool),(2,4,128)),
                             (torch.ones(4,dtype=torch.bool),torch.tensor([True,False,True]),(4,3,128))]:
        capture=[]
        with torch.no_grad():
            hook=dual.fusion_block.register_forward_pre_hook(lambda _,args:capture.append(tuple(args[0].shape)))
            dual(scene,keep,bands)
            hook.remove()
        assert capture==[shape]
        output=dual(scene,keep,bands)
        active=qa & keep[:,None] & bands[dual.projections['0'].band_slots]
        perturbed=Scene((2,2),layout,torch.where(active,values,float('nan')),qa,(0,0),int(qa.sum()))
        torch.testing.assert_close(dual(perturbed,keep,bands),output,atol=0,rtol=0)
        gradient,=torch.autograd.grad(output.square().mean(),values)
        assert not gradient[~active].any()


def test_legacy_ablation_preserves_shared_initial_tensors_and_paths():
    from pd_here_s60.whole_scene import Scene
    from pd_here_s60.whole_scene_dual import WholeSceneDualViT
    layout=(('red',4,4),('nir',2,2))
    means={layout:[.1,.2]}
    torch.manual_seed(17)
    old=WholeSceneDualViT(means).eval()
    torch.manual_seed(17)
    shallow=WholeSceneDualViT(means,global_depth=1).eval()
    torch.manual_seed(17)
    spectral=WholeSceneDualViT(means,spectral_only=True).eval()
    reference=old.state_dict()
    for model in (shallow,spectral):
        for name,value in model.state_dict().items():
            expected=reference[name][:,128:] if model is spectral and name=='fuse.weight' else reference[name]
            assert torch.equal(value,expected),name
    assert len(shallow.global_blocks)==1 and len(spectral.global_blocks)==2
    assert not hasattr(spectral,'spatial_block') and not hasattr(spectral.projections['0'],'spatial_reduce')
    with torch.no_grad():
        for model in (old,shallow,spectral):
            model.projections['0'].output.weight.fill_(.01)
    values=torch.rand(4,20,requires_grad=True)
    qa=torch.ones_like(values,dtype=torch.bool)
    scene=Scene((2,2),layout,values,qa,(0,0),80,120)
    keep=torch.ones(4,dtype=torch.bool)
    bands=torch.tensor([True,False])
    hook=old.global_blocks[1].register_forward_hook(lambda _,args,output:args[0])
    torch.testing.assert_close(shallow(scene,keep,bands),old(scene,keep,bands),atol=1e-6,rtol=1e-5)
    hook.remove()
    hook=old.fuse.register_forward_pre_hook(lambda _,args:(torch.cat((torch.zeros_like(args[0][:,:128]),args[0][:,128:]),-1),))
    result=spectral(scene,keep,bands)
    torch.testing.assert_close(result,old(scene,keep,bands),atol=1e-6,rtol=1e-5)
    hook.remove()
    gradient,=torch.autograd.grad(result[...,-4:].square().mean(),values)
    assert not gradient[:,16:].any()
