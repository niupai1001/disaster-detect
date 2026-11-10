from dataclasses import replace
import copy
import numpy as np
import pytest
import torch
from encoder_v3.shared_mae.model import NativeChannel,Sample
from encoder_v3.shared_mae.metadata import PhysicalChannel
from encoder_v3.latent_response.types import NativeGrid
from encoder_v3.hierarchical.geometry import partitions,area_map,neighborhood,sparse_edges
from encoder_v3.hierarchical.model import HierarchicalMAE,SparseAttention,GlobalBlock

torch.set_num_threads(2)


def sample(side=16,gsd=1.,channels=2,name='synthetic'):
    grid=NativeGrid((gsd,0.,100.3,0.,-gsd,200.7),(side,side),'EPSG:32631')
    return Sample(name,'TRAIN',[NativeChannel(torch.rand(side,side),torch.ones(side,side,dtype=torch.bool),grid,
        PhysicalChannel('optical','surface_reflectance',gsd,gsd,.45+i*.15,.04,source='synthetic:test'),str(i)) for i in range(channels)])


@pytest.mark.parametrize('gsd,counts',[(10.,(49,16,1)),(20.,(16,4,1)),(30.,(9,4,1))])
def test_geometry_counts_and_exact_area(gsd,counts):
    s=sample(side=1,gsd=128.)
    # Explicit scene with the same 128m extent and a reference finest native support.
    g=NativeGrid((gsd,0.,100.,0.,-gsd,228.),(int(128//gsd)+1,)*2,'EPSG:32631')
    geo,parts,_=partitions((g,),128.)
    # First full geographic patch has these counts; partial boundary patches are separate.
    assert tuple(int((p.patch==0).sum()) for p in parts)==counts
    for p in parts:
        m=area_map(g,geo,p,'cpu')
        torch.testing.assert_close(m.native(torch.ones(len(p))),torch.ones(g.shape[0]*g.shape[1]),atol=1e-6,rtol=1e-6)
        value,support=m.aggregate(torch.ones(g.shape[0]*g.shape[1],32),torch.ones(g.shape,dtype=torch.bool))
        torch.testing.assert_close(value,torch.ones_like(value))
        torch.testing.assert_close(support,torch.ones_like(support),atol=1e-6,rtol=1e-6)
    for c,p in zip(parts,parts[1:]):
        assert np.all(c.area>0) and c.parent.min()>=0 and c.parent.max()<len(p)


def test_anisotropic_offset_boundary_and_mask_independent_geometry():
    g=NativeGrid((1.3,0.,500000.2,0.,-2.1,4000000.7),(9,11),'EPSG:32631')
    geo,parts,_=partitions((g,),5.)
    assert np.isclose(sum(parts[0].area),9*11*1.3*2.1)
    for p in parts:
        mapping=area_map(g,geo,p,'cpu')
        torch.testing.assert_close(mapping.native(torch.ones(len(p))),torch.ones(99),atol=1e-6,rtol=1e-6)
        _,support=mapping.aggregate(torch.ones(99,2),torch.zeros(9,11,dtype=torch.bool))
        assert not support.any()
    with pytest.raises(ValueError):NativeGrid((1.,.1,0.,0.,-1.,0.),(4,4),'EPSG:32631')
    with pytest.raises(ValueError):NativeGrid((1.,0.,0.,0.,-1.,0.),(4,4),'EPSG:4326')


@pytest.mark.parametrize('kind',['A','B1'])
@pytest.mark.parametrize('poison',[float('nan'),1e6])
def test_hidden_poison_predictions_and_zero_input_gradient(kind,poison):
    torch.manual_seed(4);s=sample();m=HierarchicalMAE(patch_metres=4.)
    loss,details=m([s],kind=kind,return_predictions=True)
    changed=copy.deepcopy(s)
    for c,mask in zip(changed.channels,details[0]['target_masks']):c.values[mask]=poison
    _,other=m([changed],kind=kind,return_predictions=True)
    for x,y in zip(details[0]['predictions'],other[0]['predictions']):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    for x,y in zip(details[0]['online_features'],other[0]['online_features']):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    for c in s.channels:c.values.requires_grad_()
    _,d=m([s],kind=kind,return_predictions=True)
    sum(x.sum() for x in d[0]['predictions']).backward()
    for c,mask in zip(s.channels,d[0]['target_masks']):
        assert c.values.grad is None or not c.values.grad[mask].any()


def test_parallel_frontend_three_branches_and_four_cross_paths():
    torch.manual_seed(3);m=HierarchicalMAE(patch_metres=8.);s=sample(side=24)
    seen=[]
    hook=m.online.convolution.register_forward_hook(lambda *a:seen.append(1))
    loss,_=m([s]);loss.mean().backward();hook.remove()
    assert len(seen)==1
    for module in [*m.online.projection,*m.online.branches,m.online.coarse,*m.online.cross.values(),*m.decoder,*m.topdown]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    for level in range(3):
        base=[x.clone() for x in m.online(s,m.layout(s),[c.valid for c in s.channels])[0]]
        with torch.no_grad():m.online.projection[level].weight.add_(.5)
        changed=m.online(s,m.layout(s),[c.valid for c in s.channels])[0]
        assert not torch.equal(base[level],changed[level])


def test_sparse_neighbor_positive_and_remote_negative_control():
    s=sample(side=32);_,parts,_=partitions(tuple(c.grid for c in s.channels),8.)
    edges=neighborhood(parts[0],(3.,3.),'cpu')
    edge=torch.cat([x[2]+torch.tensor([[x[0]],[0]]) for x in edges],1)
    assert np.any(parts[0].patch[edge[0]]!=parts[0].patch[edge[1]])
    assert not ((edge[0]==0)&(edge[1]==len(parts[0])-1)).any()
    x=torch.randn(len(parts[0]),64,requires_grad=True)
    block=SparseAttention(64);valid=torch.ones(len(x),dtype=torch.bool)
    block(x,x,edges,valid,valid)[0,0].backward()
    assert x.grad[1].abs().sum()>0 and not x.grad[-1].any()


def test_global_attention_matches_unchunked_and_reaches_far_token():
    torch.manual_seed(12);block=GlobalBlock();x=torch.randn(530,128,requires_grad=True)
    valid=torch.ones(530,dtype=torch.bool)
    actual=block(x,valid)
    q,k,v=block.qkv(block.norm(x)).reshape(530,3,4,-1).permute(1,2,0,3).unbind(0)
    y=x+block.out(torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(0,1).flatten(1))
    expected=y+block.ff(y)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    actual[0,0].backward();assert x.grad[-1].abs().sum()>0


def test_scene_isolation_and_single_channel():
    torch.manual_seed(8);m=HierarchicalMAE(patch_metres=4.)
    a,b=sample(name='a'),sample(side=12,name='b',channels=1)
    batch,_=m([a,b]);single=torch.cat([m([s])[0] for s in (a,b)])
    torch.testing.assert_close(batch,single,rtol=0,atol=0)
    assert m.available_tasks(b)==['A']
    with pytest.raises(ValueError,match='two valid channels'):m([b],kind='B1')


def test_jepa_target_ema_and_checkpoint_next_step(tmp_path):
    from encoder_v3.hierarchical.train import save,resume
    from encoder_v3.latent_response.checkpoint import parameter_sha
    torch.manual_seed(9);m=HierarchicalMAE(patch_metres=4.,lambda_jepa=.2)
    for a,b in zip(m.online.parameters(),m.target.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    s=sample();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    assert not any(p.requires_grad for p in m.target.parameters())
    scheduler=torch.optim.lr_scheduler.StepLR(opt,2,.5);state={'cursor':0};binding={'test':'v3-h'}
    def step():
        opt.zero_grad(set_to_none=True);loss,details=m([s]);loss.mean().backward()
        assert all(p.grad is None for p in m.target.parameters())
        assert len(details[0]['jepa_counts'])==3 and details[0]['jepa']>0
        before=next(m.target.parameters()).clone();opt.step()
        expected=before*m.ema_momentum+next(m.online.parameters())*(1-m.ema_momentum)
        m.update_ema();torch.testing.assert_close(next(m.target.parameters()),expected)
        scheduler.step();state['cursor']+=1
    step();save(tmp_path/'last.pt',m,opt,scheduler,state,binding)
    step();expected=parameter_sha(m)
    resume(tmp_path/'last.pt',m,opt,scheduler,state,binding);step()
    assert parameter_sha(m)==expected and state['cursor']==2 and int(m.ema_updates)==2


def test_task_weighting_microbatch_and_heldout_guard(monkeypatch):
    from encoder_v3.hierarchical import train
    torch.manual_seed(7);m=HierarchicalMAE(patch_metres=4.)
    samples=[sample(name='a',channels=1),sample(name='b')]
    rows=[dict(id=s.identifier,sensor='S2',role='TRAIN',bands=[dict(shape=(16,16))]) for s in samples]
    ds=type('Dataset',(),{'rows':rows})()
    monkeypatch.setattr(train,'sample_from_dataset',lambda dataset,i,resolver,device:samples[i])
    config=dict(device='cpu',precision='float32',seed=17,micro_batch_observations=2,micro_batch_pixels=999999,task_weights={'A':.7,'B1':.3})
    value,_=train.batch_loss(m,ds,rows,None,config,backward=True)
    grads={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    other,_=train.batch_loss(m,ds,rows,None,dict(config,micro_batch_observations=1),backward=True)
    assert value==other
    for n,p in m.named_parameters():
        if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=1e-6,rtol=1e-5)
    direct=(m([samples[0]],kind='A')[0][0]+.7*m([samples[1]],kind='A')[0][0]+.3*m([samples[1]],kind='B1')[0][0])/2
    assert abs(float(direct.detach())-value)<1e-6
    with pytest.raises(ValueError,match='Non-TRAIN'):
        train.update(m,ds,[dict(rows[0],role='VAL')],None,None,None,{},config)


def test_qa_poison_and_empty_observation():
    torch.manual_seed(21);m=HierarchicalMAE(patch_metres=4.);s=sample()
    for c in s.channels:c.valid[:3,:]=False
    changed=copy.deepcopy(s)
    for c in changed.channels:c.values[~c.valid]=float('nan')
    a,_=m([s]);b,_=m([changed]);torch.testing.assert_close(a,b,rtol=0,atol=0)
    for c in s.channels:c.valid[:]=False
    with pytest.raises(ValueError,match='No valid'):m.available_tasks(s)


def test_chunked_fusion_matches_whole_value_and_gradient():
    from encoder_v3.hierarchical.model import Encoder
    torch.manual_seed(32);e=Encoder();summary=torch.randn(3,4101,32,requires_grad=True)
    support=torch.rand(3,4101);condition=torch.randn(3,32)
    whole=e.fuse(summary,support,condition,0)
    split=torch.cat([e.fuse(summary[:,b:b+4096],support[:,b:b+4096],condition,0) for b in range(0,4101,4096)])
    torch.testing.assert_close(whole,split,atol=1e-6,rtol=1e-6)
    first=torch.autograd.grad(whole.square().mean(),summary,retain_graph=True)[0]
    second=torch.autograd.grad(split.square().mean(),summary)[0]
    torch.testing.assert_close(first,second,atol=1e-7,rtol=1e-5)


@pytest.mark.parametrize('qwidth,kwidth',[(96,64),(64,96),(128,96),(96,128)])
def test_each_cross_direction_respects_source_validity(qwidth,kwidth):
    block=SparseAttention(qwidth,kwidth)
    q=torch.randn(3,qwidth);context=torch.randn(4,kwidth,requires_grad=True)
    index=np.array([[0,0,1,2],[0,1,2,3]])
    edge=sparse_edges(index,3,'cpu');valid=torch.ones(3,dtype=torch.bool)
    result=block(q,context,edge,valid,torch.tensor([True,False,True,False]))
    result[0,0].backward()
    assert context.grad[0].abs().sum()>0 and not context.grad[1:].any()
    poisoned=context.detach().clone();poisoned[1]+=1e4;poisoned[3]-=1e4
    other=block(q,poisoned,edge,valid,torch.tensor([True,False,True,False]))
    torch.testing.assert_close(result,other,rtol=0,atol=0)


def test_jepa_online_permission_and_target_response():
    m=HierarchicalMAE(patch_metres=4.,lambda_jepa=.2);s=sample()
    _,first=m([s],return_predictions=True)
    changed=copy.deepcopy(s)
    for c,mask in zip(changed.channels,first[0]['target_masks']):c.values[mask]+=100.
    _,second=m([changed],return_predictions=True)
    for a,b in zip(first[0]['online_features'],second[0]['online_features']):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    full=[c.valid for c in s.channels]
    a=m.target(s,m.layout(s),full)[0];b=m.target(changed,m.layout(changed),full)[0]
    assert any(not torch.equal(x,y) for x,y in zip(a,b))


def test_viewer_new_run_and_real_resume_identity(tmp_path):
    from encoder_v3.hierarchical.train import writer_for
    config={'viewer_root':str(tmp_path/'viewer'),'lambda_jepa':0.,'task_weights':{'A':.7,'B1':.3}}
    state={};binding={'test':'v3-h'}
    with writer_for(config,binding,state,tmp_path/'train',False) as writer:
        writer.log(step=0,epoch=0.,values={'val_loss':1.})
    identity=copy.deepcopy(state['viewer'])
    with pytest.raises(ValueError,match='checkpoint'):writer_for(config,binding,state,tmp_path/'train',False)
    with writer_for(config,binding,state,tmp_path/'train',True) as writer:
        writer.log(step=1,epoch=.01,values={'train_loss':.9})
    assert state['viewer']==identity
    with pytest.raises(ValueError,match='identity'):writer_for(config,{'test':'other'},state,tmp_path/'train',True)
