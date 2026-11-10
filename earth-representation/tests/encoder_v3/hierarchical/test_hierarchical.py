from dataclasses import replace
from pathlib import Path
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
torch.use_deterministic_algorithms(True)


def test_compact_encoder_matches_dense_features_and_parameter_gradients():
    from encoder_v3.hierarchical.model import Encoder,ParentAttention
    torch.manual_seed(83);m=HierarchicalMAE(patch_metres=4.);s=sample(side=12)
    layout=m.layout(s);keeps,_,_=m.masks(s,layout,'A',17,0)
    dense=m.online;compact=copy.deepcopy(dense)
    for network,threshold in ((dense,10**9),(compact,0)):
        for module in network.modules():
            if isinstance(module,(Encoder,SparseAttention,ParentAttention)):module.compact_threshold=threshold
    a=dense(s,layout,keeps);b=compact(s,layout,keeps)
    for x,y in zip(a.features,b.features):torch.testing.assert_close(x,y,atol=3e-6,rtol=1e-5)
    weights=[torch.randn_like(x) for x in a.features]
    sum((x*w).mean() for x,w in zip(a.features,weights)).backward()
    sum((x*w).mean() for x,w in zip(b.features,weights)).backward()
    for p,q in zip(dense.parameters(),compact.parameters()):
        if p.grad is None:assert q.grad is None
        else:torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=2e-4)


def test_compact_attention_matches_dense_values_and_all_gradients():
    torch.manual_seed(71);s=sample(side=12);m=HierarchicalMAE(patch_metres=4.)
    edges=m.layout(s)[3][0];x=torch.randn(len(m.layout(s)[1][0]),64);valid=torch.rand(len(x))>.55
    dense=SparseAttention(64);dense.compact_threshold=10**9
    compact=copy.deepcopy(dense);compact.compact_threshold=0
    a=x.clone().requires_grad_();b=x.clone().requires_grad_()
    ya=dense(a,a,edges,valid,valid);yb=compact(b,b,edges,valid,valid)
    torch.testing.assert_close(ya,yb,atol=2e-6,rtol=1e-5)
    ya.square().mean().backward();yb.square().mean().backward()
    torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=1e-4)
    for p,q in zip(dense.parameters(),compact.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=1e-4)


def test_batched_mask_counts_and_qa_union_match_per_channel_reference():
    from encoder_v3.hierarchical.model import mask_sums,occupied_regions
    torch.manual_seed(91);s=sample(side=12,channels=3)
    c=s.channels[2];c.grid=NativeGrid((1.7,0.,100.9,0.,-2.3,201.1),(7,9),'EPSG:32631')
    c.values=torch.randn(7,9);c.valid=torch.rand(7,9)>.3
    s.channels[1].valid.zero_();s.channels[0].valid=torch.rand(12,12)>.4
    masks=[c.valid for c in s.channels]
    torch.testing.assert_close(mask_sums(masks),torch.tensor([int(v.sum()) for v in masks]))
    layout=HierarchicalMAE(patch_metres=4.).layout(s);expected=torch.zeros(len(layout[1][2]),dtype=torch.bool)
    for c,maps in zip(s.channels,layout[2]):expected|=maps[2].support(c.valid)>0
    assert torch.equal(occupied_regions(s,layout),expected)


def test_visible_condition_constant_one_pixel_and_empty_support():
    from encoder_v3.hierarchical.simple_decoder import visible_condition
    for keep,empty in ((torch.ones(2,2,dtype=torch.bool),False),(torch.tensor([[True,False],[False,False]]),False),(torch.zeros(2,2,dtype=torch.bool),True)):
        truth=torch.full((1,2,2),.3,requires_grad=True)
        condition,scale=visible_condition(torch.zeros(1,32),torch.zeros(1,12),truth,keep,empty)
        assert scale.item()==(1. if empty else 0.) and bool(torch.isfinite(condition).all())
        condition.sum().backward();assert bool(torch.isfinite(truth.grad).all())


def test_native_decoder_exact_bilinear_footprint_and_gradients():
    from encoder_v3.hierarchical.simple_decoder import NativeResidualDecoder
    fine=NativeGrid((1.,0.,100.3,0.,-1.,220.7),(16,16),'EPSG:32631')
    coarse=NativeGrid((3.7,0.,100.6,0.,-3.7,220.4),(4,4),'EPSG:32631')
    geo,parts,_=partitions((fine,coarse),5.7);part=parts[0];mapping=area_map(coarse,geo,part,'cpu')
    cx,cy=part.centers.T;width=np.diff(part.x)[np.arange(len(part))%(len(part.x)-1)];height=np.diff(part.y)[np.arange(len(part))//(len(part.x)-1)]
    coeff=torch.tensor(np.column_stack((1+cx*.2+cy*.3+cx*cy*.02,(.2+.02*cy)*width,(.3+.02*cx)*height,.02*width*height)),dtype=torch.float32,requires_grad=True)
    target=torch.ones(coarse.shape,dtype=torch.bool)
    actual=NativeResidualDecoder.integrate(coeff,mapping,coarse,geo,part,target)
    yy,xx=np.meshgrid(np.arange(4),np.arange(4),indexing='ij');x=(coarse.affine[2]-geo.origin[0])+(xx+.5)*3.7;y=(coarse.affine[5]-geo.origin[1])-(yy+.5)*3.7
    expected=torch.tensor(1+x*.2+y*.3+x*y*.02,dtype=torch.float32)
    torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-6)
    actual.sum().backward();assert bool(torch.isfinite(coeff.grad).all()) and coeff.grad[:,0].sum()==pytest.approx(16.)


@pytest.mark.parametrize('kind',['A','B1'])
def test_native_decoder_baseline_is_legal_and_learns_band_conditions(kind):
    from encoder_v3.hierarchical.simple_decoder import native_stencil
    torch.manual_seed(17);m=HierarchicalMAE(patch_metres=4.,residual_readout=True,decoder_type='native_residual')
    s=sample(side=12,channels=3);s.channels[0].valid[0,0]=False
    loss,detail=m([s],kind=kind,return_predictions=True);d=detail[0]
    keeps,targets,_=m.masks(s,m.layout(s),kind,17,0)
    assert d['mae']==pytest.approx(d['baseline_mae'])
    for c,keep,target,pred in zip(s.channels,keeps,targets,d['predictions']):
        if not bool(target.any()):continue
        query,source,weight=native_stencil(c.grid,keep.numpy(),target.numpy(),'cpu')
        expected=torch.zeros(len(query)) if source is None else (c.values.flatten()[source]*weight).sum(1)
        torch.testing.assert_close(pred.flatten()[query],expected,rtol=0,atol=0)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3)
    loss.mean().backward();opt.step();opt.zero_grad(set_to_none=True)
    changed=copy.deepcopy(s)
    for c,mask in zip(changed.channels,targets):c.values[mask]+=100
    changed.channels[0].values[0,0]=float('nan')
    first=m([s],kind=kind,return_predictions=True)[1][0]['predictions']
    poisoned=m([changed],kind=kind,return_predictions=True)[1][0]['predictions']
    for a,b in zip(first,poisoned):torch.testing.assert_close(a,b,rtol=0,atol=0)
    m([s],kind=kind)[0].mean().backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in m.online.parameters())
    numeric=torch.tensor([c.physical.encoded()[0] for c in s.channels]);condition=torch.cat((torch.zeros(3,32),numeric,torch.zeros(3,2)),-1)
    assert (m.decoder.condition(condition)[0]-m.decoder.condition(condition)[2]).norm()>.001


def test_twenty_update_warmup_preserves_cosine_budget():
    from encoder_v3.hierarchical.train import warmup_updates,lr_multiplier
    config={'warmup_steps':20,'warmup_epochs':1.}
    assert warmup_updates(config,232,2320)==20
    assert warmup_updates({'warmup_epochs':1.},232,2320)==232
    parameter=torch.nn.Parameter(torch.zeros(()));opt=torch.optim.SGD([parameter],lr=1e-4)
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:lr_multiplier(n,2320,20))
    rates=[]
    for _ in range(22):
        rates.append(opt.param_groups[0]['lr']);opt.step();schedule.step()
    assert rates[0]==pytest.approx(5e-6) and rates[19]==pytest.approx(1e-4)
    assert rates[20]==pytest.approx(1e-4) and rates[21]<rates[20]
    for invalid in (-1,True,20.5,2320):
        with pytest.raises(ValueError):warmup_updates(dict(config,warmup_steps=invalid),232,2320)


@pytest.mark.parametrize('kind',['A','B1'])
def test_residual_initialization_is_visible_only_and_reports_baseline(kind):
    torch.manual_seed(17);original=HierarchicalMAE(patch_metres=4.)
    torch.manual_seed(17);model=HierarchicalMAE(patch_metres=4.,residual_readout=True)
    for a,b in zip(original.online.parameters(),model.online.parameters()):assert torch.equal(a,b)
    s=sample(side=12,channels=3);s.channels[0].valid[0,0]=False
    loss,details=model([s],kind=kind,return_predictions=True)
    d=details[0];keeps,targets,_=model.masks(s,model.layout(s),kind,17,0)
    for c,keep,prediction in zip(s.channels,keeps,d['predictions']):
        expected=torch.where(keep,c.values,0.).sum()/keep.sum().clamp_min(1)
        torch.testing.assert_close(prediction,expected.expand_as(prediction),rtol=0,atol=0)
    assert d['mae']==pytest.approx(d['baseline_mae'])
    assert d['baseline_fallback_channels']==(sum(bool(mask.any()) for mask in targets) if kind=='B1' else 0)
    changed=copy.deepcopy(s)
    for c,mask in zip(changed.channels,targets):c.values[mask]+=100.
    changed.channels[0].values[0,0]=float('nan')
    after=model([changed],kind=kind,return_predictions=True)[1][0]['predictions']
    for a,b in zip(d['predictions'],after):torch.testing.assert_close(a,b,rtol=0,atol=0)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    loss.mean().backward();assert model.output[-1].weight.grad.abs().sum()>0
    opt.step();opt.zero_grad(set_to_none=True)
    before=[p.detach().clone() for p in model.online.parameters()]
    observed=model([s],kind=kind,return_predictions=True)[1][0]['predictions']
    poisoned=model([changed],kind=kind,return_predictions=True)[1][0]['predictions']
    for a,b in zip(observed,poisoned):torch.testing.assert_close(a,b,rtol=0,atol=0)
    model([s],kind=kind)[0].mean().backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.online.parameters())
    opt.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,model.online.parameters()))


def test_enmap_exclusion_is_config_bound_and_train_only(monkeypatch):
    from types import SimpleNamespace
    from encoder_v3.hierarchical import train
    from encoder_v3.latent_response.enmap_data import CombinedDataset
    base=SimpleNamespace(rows=[{'id':'s','sensor':'S2'}],identities={'base':'frozen'})
    enmap=SimpleNamespace(rows=[{'id':'e','sensor':'EnMAP'}],identities={'enmap':'frozen'})
    original={'TRAIN':CombinedDataset(base,enmap),'VAL':SimpleNamespace(rows=[{'id':'v','sensor':'S2'}])}
    monkeypatch.setattr(train,'base_load_data',lambda:(original,None))
    selected,_=train.load_data({'exclude_enmap':True})
    assert selected['TRAIN'] is base and selected['VAL'] is original['VAL']
    assert [r['id'] for r in original['TRAIN'].rows]==['s','e']
    assert train.load_data()[0]['TRAIN'] is original['TRAIN']
    with pytest.raises(ValueError,match='boolean'):train.load_data({'exclude_enmap':'yes'})
    with pytest.raises(ValueError,match='EnMAP TRAIN excluded'):
        train.update(None,None,[{'id':'e','sensor':'EnMAP','role':'TRAIN'}],None,None,None,{}, {'exclude_enmap':True})


def test_two_device_secondary_reader_excludes_enmap(monkeypatch):
    from types import SimpleNamespace
    from encoder_v3.hierarchical import train
    from encoder_v3.hierarchical.parallel import TwoDeviceBatch
    from encoder_v3.latent_response.enmap_data import CombinedDataset
    base=SimpleNamespace(rows=[{'id':'s','sensor':'S2'}],identities={'data':'same'})
    enmap=SimpleNamespace(rows=[{'id':'e','sensor':'EnMAP'}],identities={'data':'enmap'})
    datasets={'TRAIN':CombinedDataset(base,enmap),'VAL':SimpleNamespace(rows=[])}
    resolver=SimpleNamespace(catalog_sha='same')
    monkeypatch.setattr(train,'base_load_data',lambda:(copy.deepcopy(datasets),resolver))
    model=HierarchicalMAE(patch_metres=4.);worker=TwoDeviceBatch(model,base,resolver,{'devices':['cpu','cpu'],'exclude_enmap':True})
    try:assert all([r['sensor'] for r in ds.rows]==['S2'] for ds in worker.datasets)
    finally:worker.close()


def test_descriptor_free_has_no_physical_path_and_distinguishes_bands():
    from encoder_v3.shared_mae.model import SharedMemory,ConditionalConvolution
    torch.manual_seed(17);s=sample(side=12);s.instrument='test'
    keys=[('test',c.name) for c in s.channels]
    m=HierarchicalMAE(patch_metres=4.,frontend='shared_static_band_embedding',channel_keys=keys)
    assert not any(isinstance(layer,(SharedMemory,ConditionalConvolution)) for layer in m.modules())
    for c in s.channels:c.physical=None
    loss,details=m([s],return_predictions=True)
    conditions=m.online.conditions(s)
    assert float((conditions['output'][0]-conditions['output'][1]).detach().norm())>1.
    preds=details[0]['predictions'];assert float((preds[0]-preds[1]).detach().abs().max())>1e-4
    loss.mean().backward()
    for p in (m.online.convolution.weight,m.online.memory.fusion.weight,m.online.memory.output.weight):
        assert p.grad is not None and float(p.grad.norm())>0
    original=[p.detach().clone() for p in preds]
    targets=details[0]['target_masks']
    for c,mask in zip(s.channels,targets):c.values[mask]+=100
    changed=m([s],return_predictions=True)[1][0]['predictions']
    for a,b in zip(original,changed):torch.testing.assert_close(a,b)
    s.instrument='unknown'
    with pytest.raises(ValueError,match='Untrained'):m([s])


def test_static_batched_convolution_matches_expanded_grouped_path():
    from encoder_v3.hierarchical.descriptor_free import DescriptorFreeEncoder
    from encoder_v3.hierarchical.model import Encoder
    torch.manual_seed(7);m=DescriptorFreeEncoder(1,[('test','b')])
    values=torch.randn(3,8,8);keep=torch.rand(3,8,8)>.3
    kernel,bias=m.convolution(torch.zeros(3,32));clean=torch.where(keep,values,0.)
    expected=Encoder.convolve(m,clean,kernel);actual=m.convolve(clean,kernel)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5)
    a=torch.autograd.grad(actual.square().mean(),m.convolution.weight,retain_graph=True)[0]
    b=torch.autograd.grad(expected.square().mean(),m.convolution.weight)[0]
    torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)


@pytest.mark.parametrize('residual',[False,True,'native'])
def test_short_trial_stops_without_shortening_lr_schedule(tmp_path,monkeypatch,residual):
    import json,sys
    from encoder_v3.hierarchical import train
    class Dataset:
        def __init__(self,role,count):
            self.samples=[sample(side=8,name=f'{role}{i}') for i in range(count)]
            for s in self.samples:s.instrument='test'
            self.rows=[dict(id=s.identifier,role=role,sensor='S2',instrument='test',bands=[dict(shape=(8,8))]) for s in self.samples]
    datasets={'TRAIN':Dataset('TRAIN',3),'VAL':Dataset('VAL',1)}
    monkeypatch.setattr(train,'load_data',lambda *args:(datasets,None))
    monkeypatch.setattr(train,'sample_from_dataset',lambda ds,i,resolver,device:ds.samples[i])
    binding={'test':'descriptor-free-short'};monkeypatch.setattr(train,'identity',lambda *args:binding)
    config=json.loads(Path('configs/config_v3_h_descriptor_free_v1.json').read_text())
    if residual:config.update(frontend='physical_descriptor',residual_readout=True,warmup_steps=20,epochs=11)
    if residual=='native':config.update(decoder_type='native_residual',warmup_steps=50,epochs=31)
    config.pop('devices');config.update(device='cpu',patch_metres=4.,batch_size=2,max_updates=2,
        channel_keys=[['test','0'],['test','1']],viewer_root=str(tmp_path/'viewer'))
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config))
    auth=tmp_path/'auth.json';auth.write_text(json.dumps({'source':'user_in_current_chat','identity':binding}))
    gate=tmp_path/'gate.json';gate.write_text(json.dumps({'identity':binding,'training_release_ready':True}))
    folder=tmp_path/'run';monkeypatch.setattr(sys,'argv',['train','--config',str(cfg),'--authorization',str(auth),'--readiness',str(gate),'--run-dir',str(folder)])
    train.main();payload=torch.load(folder/'last.pt',weights_only=False)
    assert payload['runner']['cursor']==2 and payload['runner']['status']=='TRIAL_COMPLETE'
    assert payload['scheduler']['last_epoch']==2
    assert payload['optimizer']['param_groups'][0]['lr']==pytest.approx(config['lr']*(3/config['warmup_steps'] if residual else 1))
    if residual:
        result=json.loads((folder/'validation_2.json').read_text())
        assert result['baseline_metric']==pytest.approx(result['members'][0]['tasks']['A']['baseline_mae'])
        initial=json.loads((folder/'validation_0.json').read_text())
        assert initial['metric']==pytest.approx(initial['baseline_metric'])


@pytest.mark.parametrize('side,gsd,length',[(17,1.3,5.7),(19,2.1,12.8),(637,1.,128.)])
def test_cartesian_neighborhood_exactly_matches_tree(side,gsd,length):
    from scipy.spatial import cKDTree
    s=sample(side=side,gsd=gsd,channels=1);_,parts,ref=partitions((s.channels[0].grid,),length)
    for part in parts[:1] if side>100 else parts:
        radii=((ref[0]*3,ref[1]*3),) if side>100 else ((ref[0]*3,ref[1]*3),(np.nextafter(length,0),np.nextafter(length,np.inf)))
        for radius in radii:
            points=part.centers/np.array(radius);neighbors=cKDTree(points).query_ball_point(points,1.,p=np.inf,return_sorted=True)
            expected=np.stack((np.repeat(np.arange(len(part)),[len(n) for n in neighbors]),np.concatenate(neighbors)))
            chunks=neighborhood(part,radius,'cpu');actual=np.concatenate([np.stack((c[2][0].numpy()+c[0],c[2][1].numpy())) for c in chunks],axis=1)
            assert np.array_equal(actual,expected)


def test_a_only_never_executes_disabled_task(monkeypatch):
    from encoder_v3.hierarchical import train
    from encoder_v3.hierarchical.verification import gradient_contributions
    torch.manual_seed(51);model=HierarchicalMAE(patch_metres=4.);s=sample(side=12)
    row=dict(id=s.identifier,sensor='S2',role='TRAIN',bands=[dict(shape=(12,12))]);ds=type('Dataset',(),{'rows':[row]})()
    monkeypatch.setattr(train,'sample_from_dataset',lambda *args:s)
    config=dict(device='cpu',precision='float32',seed=17,task_weights={'A':1.,'B1':0.},lambda_jepa=0.)
    original=model.forward;calls=[]
    def forward(*args,**kwargs):
        calls.append(kwargs.get('kind','A'));assert calls[-1]=='A'
        return original(*args,**kwargs)
    monkeypatch.setattr(model,'forward',forward)
    value,records=train.batch_loss(model,ds,[row],None,config,backward=True)
    expected=original([s],kind='A',seed=17)[0].item();assert abs(value-expected)<1e-7
    assert set(records[0]['tasks'])=={'A'} and records[0]['tasks']['A']['task_weight']==1.
    diagnostics=gradient_contributions(model,s,config);assert set(diagnostics['tasks'])=={'A'} and calls==['A','A']
    monkeypatch.setattr(model,'available_tasks',lambda sample:['B1'])
    with pytest.raises(ValueError,match='No enabled'):train.enabled_tasks(model,s,config)


def test_two_device_validation_disables_gradients_and_restores_state(monkeypatch):
    from encoder_v3.hierarchical import train
    from encoder_v3.hierarchical.parallel import TwoDeviceBatch
    from encoder_v3.hierarchical.verification import compare_snapshots
    from encoder_v3.latent_response.checkpoint import snapshot
    torch.manual_seed(52)
    samples=[sample(side=8,name=str(i)) for i in range(3)]
    train_rows=[dict(id=s.identifier,sensor='S2',role='TRAIN',bands=[dict(shape=(8,8))]) for s in samples]
    val_rows=[dict(r,role='VAL') for r in train_rows]
    ds=type('Dataset',(),{'rows':train_rows})();val=type('Dataset',(),{'rows':val_rows})()
    monkeypatch.setattr(train,'sample_from_dataset',lambda ds,i,resolver,device:samples[i])
    config=dict(device='cpu',devices=['cpu','cpu'],precision='float32',seed=17,validation_seed=1729,task_weights={'A':1.,'B1':0.})
    model=HierarchicalMAE(patch_metres=4.);worker=TwoDeviceBatch(model,ds,None,config,secondary_data=(copy.deepcopy(ds),None),validation_dataset=val,secondary_validation_dataset=copy.deepcopy(val))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4);scheduler=torch.optim.lr_scheduler.StepLR(opt,5);state={'cursor':0}
    calls=[]
    def inspect(module,args):calls.append((torch.is_grad_enabled(),module.training))
    handles=[m.register_forward_pre_hook(inspect) for m in (model,worker.replica)]
    before=snapshot(model,opt,state,scheduler)
    try:
        expected=train.validate(model,val,None,opt,scheduler,state,config)
        received=train.validate(model,val,None,opt,scheduler,state,config,parallel=worker)
        assert abs(expected['metric']-received['metric'])<1e-7 and received['observations']==3
        assert calls and all(not grad and not training for grad,training in calls)
        assert all(compare_snapshots(before,snapshot(model,opt,state,scheduler)).values())
        assert worker.replica.training==model.training
    finally:
        for handle in handles:handle.remove()
        worker.close()


def test_large_output_readout_matches_dense_values_and_gradients():
    torch.manual_seed(29)
    model=HierarchicalMAE(patch_metres=128.)
    features=torch.randn(100001,64,requires_grad=True)
    condition=torch.randn(32,requires_grad=True)
    parameters=tuple(model.output.parameters())
    dense=model.output(torch.cat((features,condition.expand(len(features),-1)),-1)).squeeze(-1)
    expected=torch.autograd.grad(dense.square().mean(),(features,condition,*parameters))
    actual=model.readout(features,condition)
    received=torch.autograd.grad(actual.square().mean(),(features,condition,*parameters))
    torch.testing.assert_close(actual,dense,atol=1e-6,rtol=1e-5)
    for a,b in zip(received,expected):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


def test_two_device_observation_mean_tail_ema_and_exact_recovery(monkeypatch,tmp_path):
    from encoder_v3.hierarchical import train
    from encoder_v3.hierarchical.parallel import TwoDeviceBatch,partition_rows
    from encoder_v3.hierarchical.verification import compare_snapshots
    from encoder_v3.latent_response.checkpoint import snapshot
    torch.manual_seed(38)
    samples=[sample(side=8,name='a',channels=1),sample(side=12,name='b'),sample(side=8,name='c')]
    rows=[dict(id=s.identifier,sensor='S2',role='TRAIN',bands=[dict(shape=c.grid.shape) for c in s.channels]) for s in samples]
    ds=type('Dataset',(),{'rows':rows})();second=copy.deepcopy(ds)
    monkeypatch.setattr(train,'sample_from_dataset',lambda dataset,i,resolver,device:samples[i])
    config=dict(device='cpu',devices=['cpu','cpu'],precision='float32',seed=17,task_weights={'A':.7,'B1':.3},clip_norm=3.,batches_per_epoch=100)
    serial=HierarchicalMAE(patch_metres=4.,lambda_jepa=.2);parallel_model=copy.deepcopy(serial)
    worker=TwoDeviceBatch(parallel_model,ds,None,config,secondary_data=(second,None))
    shards=partition_rows(rows);assert sorted(r['id'] for part in shards for r in part)==['a','b','c']
    try:
        expected,records=train.batch_loss(serial,ds,rows,None,config,backward=True)
        value,actual=worker.backward(rows,0)
        assert abs(expected-value)<1e-7 and [r['id'] for r in actual]==[r['id'] for r in records]
        for a,b in zip(serial.parameters(),parallel_model.parameters()):
            assert (a.grad is None)==(b.grad is None)
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=1e-4)
        optimizers=[torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4) for m in (serial,parallel_model)]
        schedulers=[torch.optim.lr_scheduler.StepLR(o,10,.5) for o in optimizers];states=[dict(cursor=0),dict(cursor=0)]
        for cohort in (rows,rows[:1]):
            train.update(serial,ds,cohort,None,optimizers[0],schedulers[0],states[0],config)
            train.update(parallel_model,ds,cohort,None,optimizers[1],schedulers[1],states[1],config,parallel=worker)
            for a,b in zip(serial.parameters(),parallel_model.parameters()):torch.testing.assert_close(a,b,atol=3e-6,rtol=1e-4)
            for name,value in parallel_model.state_dict().items():assert torch.equal(value,worker.replica.state_dict()[name])
        assert int(parallel_model.ema_updates)==2 and states[1]['cursor']==2
        opt=optimizers[1];scheduler=schedulers[1];state=states[1];binding={'test':'two-device'}
        train.save(tmp_path/'checkpoint.pt',parallel_model,opt,scheduler,state,binding)
        train.update(parallel_model,ds,rows,None,opt,scheduler,state,config,parallel=worker)
        expected=snapshot(parallel_model,opt,state,scheduler)
        train.resume(tmp_path/'checkpoint.pt',parallel_model,opt,scheduler,state,binding)
        train.update(parallel_model,ds,rows,None,opt,scheduler,state,config,parallel=worker)
        assert all(compare_snapshots(expected,snapshot(parallel_model,opt,state,scheduler)).values())
    finally:worker.close()


def test_two_device_worker_failure_cannot_apply_optimizer(monkeypatch):
    from encoder_v3.hierarchical import train
    from encoder_v3.hierarchical.parallel import TwoDeviceBatch
    from encoder_v3.hierarchical.verification import fingerprint
    rows=[dict(id=str(i),sensor='S2',role='TRAIN',bands=[dict(shape=(8,8))]) for i in range(2)]
    ds=type('Dataset',(),{'rows':rows})();second=copy.deepcopy(ds)
    config=dict(device='cpu',devices=['cpu','cpu'],clip_norm=3.,batches_per_epoch=100)
    model=HierarchicalMAE(patch_metres=4.);worker=TwoDeviceBatch(model,ds,None,config,secondary_data=(second,None))
    def fail(*args,**kwargs):raise RuntimeError('simulated worker failure')
    monkeypatch.setattr(train,'batch_loss',fail)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4);scheduler=torch.optim.lr_scheduler.StepLR(opt,10);state={'cursor':0}
    before=fingerprint(model.state_dict())
    try:
        with pytest.raises(RuntimeError,match='simulated worker'):
            train.update(model,ds,rows,None,opt,scheduler,state,config,parallel=worker)
        assert state=={'cursor':0} and not opt.state and fingerprint(model.state_dict())==before
    finally:worker.close()


def test_failed_execution_continuation_preserves_budget_and_source(tmp_path):
    import json,shutil,hashlib
    from encoder_v3.hierarchical import train
    from encoder_v3.hierarchical.verification import prepare_execution_continuation,fingerprint
    from encoder_v3.latent_response.run_control import reconcile_logs
    source=tmp_path/'source';source.mkdir();viewer={'run_id':'old','directory':'source-viewer'}
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,5);state={'cursor':1,'spent_updates':1,'status':'RUNNING','viewer':viewer,'best':.5}
    old={'data':{'TRAIN':'a'},'members':{'TRAIN':'b'},'physical_catalog':'c','architecture':'hierarchical_v3_h','code':'old','config':'single'}
    new=dict(old,code='new',config='dual')
    model(torch.ones(1,2)).sum().backward();opt.step();scheduler.step()
    train.save(source/'last.pt',model,opt,scheduler,state,old);shutil.copy2(source/'last.pt',source/'best.pt')
    (source/'status.json').write_text(json.dumps(dict(state,status='FAILED',cursor=2,spent_updates=2)))
    (source/'updates.jsonl').write_text(''.join(json.dumps({'cursor':i})+'\n' for i in (1,2)))
    (source/'optimizer_attempts.jsonl').write_text(''.join(json.dumps({'event':event,'sequence':i,'cursor':i})+'\n' for i in (1,2) for event in ('intent','applied')))
    before=hashlib.sha256((source/'last.pt').read_bytes()).hexdigest()
    checkpoint=prepare_execution_continuation(source,source/'last.pt',tmp_path/'new',model,opt,new)
    old_payload=torch.load(source/'last.pt',weights_only=False);payload=torch.load(checkpoint,weights_only=False)
    for key in ('model','optimizer','scheduler','rng','modes'):assert fingerprint(old_payload[key])==fingerprint(payload[key])
    assert payload['identity']==new and payload['runner']['source_viewer']==viewer and 'viewer' not in payload['runner']
    state=payload['runner'];evidence=reconcile_logs(tmp_path/'new',state)
    assert state['cursor']==1 and state['spent_updates']==2 and evidence['uncommitted']==1
    assert (tmp_path/'new/best.pt').exists() and hashlib.sha256((source/'last.pt').read_bytes()).hexdigest()==before
    with pytest.raises(ValueError,match='changes members'):
        prepare_execution_continuation(source,source/'last.pt',tmp_path/'bad',model,opt,dict(new,members={}))


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
    from encoder_v3.latent_response.checkpoint import snapshot
    from encoder_v3.hierarchical.verification import compare_snapshots
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
    step();expected=parameter_sha(m);full_expected=snapshot(m,opt,state,scheduler)
    resume(tmp_path/'last.pt',m,opt,scheduler,state,binding);step()
    assert parameter_sha(m)==expected and state['cursor']==2 and int(m.ema_updates)==2
    assert all(compare_snapshots(full_expected,snapshot(m,opt,state,scheduler)).values())


def test_explicit_token_support_and_gradient_contributions():
    from encoder_v3.hierarchical.verification import gradient_contributions
    torch.manual_seed(6);m=HierarchicalMAE(patch_metres=4.,lambda_jepa=.2);s=sample()
    output=m.online(s,m.layout(s),[c.valid for c in s.channels])
    assert len(output.tokens)==3
    for t in output.tokens:
        assert t.support_ratio.shape==(2,len(t.features)) and t.coordinates.shape==(len(t.features),2)
        assert torch.all((t.support_ratio>=0)&(t.support_ratio<=1.00001))
    report=gradient_contributions(m,s,{'seed':17,'task_weights':{'A':.7,'B1':.3},'lambda_jepa':.2})
    assert report['tasks']['A']['terms']['mae']['encoder_gradient_norm']>0
    assert report['tasks']['A']['terms']['jepa']['encoder_gradient_norm']>0
    assert len([n for n in report['tasks']['A']['terms'] if n.startswith('jepa_scale_')])==3


def test_relative_geometry_cache_keeps_absolute_positions_and_view_permissions():
    from dataclasses import replace
    m=HierarchicalMAE(patch_metres=4.);a=sample()
    b=copy.deepcopy(a)
    for c in b.channels:
        affine=list(c.grid.affine);affine[2]+=100000.;affine[5]+=200000.
        c.grid=replace(c.grid,affine=tuple(affine))
    first=m.layout(a);second=m.layout(b)
    assert first[0].origin!=second[0].origin
    assert first[2][0][0] is second[2][0][0]
    a_loss,_=m([a],kind='A');b_loss,_=m([b],kind='A')
    assert len(m._layouts)==1 and torch.isfinite(a_loss).all() and torch.isfinite(b_loss).all()
    torch.testing.assert_close(a_loss,b_loss,atol=1e-6,rtol=1e-5)


def test_pruned_parent_attention_matches_one_key_attention_and_has_no_dead_qk():
    from encoder_v3.hierarchical.model import ParentAttention
    torch.manual_seed(17);original=SparseAttention(64,96)
    torch.manual_seed(17);pruned=ParentAttention(64,96)
    x=torch.randn(8,64);context=torch.randn(3,96)
    index=np.stack((np.arange(8),np.arange(8)%3));edges=sparse_edges(index,8,'cpu')
    valid=torch.ones(8,dtype=torch.bool);source_valid=torch.tensor([True,False,True])
    torch.testing.assert_close(original(x,context,edges,valid,source_valid),pruned(x,context,edges,valid,source_valid),atol=1e-6,rtol=1e-6)
    assert not any(name.startswith(('q.','norm.','kv.')) for name,_ in pruned.named_parameters())


def test_equivalent_checkpoint_migration_preserves_active_adam_momenta():
    from encoder_v3.hierarchical.model import canonical_weights
    from encoder_v3.hierarchical.verification import migrate_single_parent_checkpoint
    from encoder_v3.latent_response.checkpoint import snapshot,restore
    torch.manual_seed(11);legacy=HierarchicalMAE(patch_metres=4.,lambda_jepa=.2)
    legacy.online.cross['m_to_f']=SparseAttention(64,96)
    legacy.online.cross['c_to_m']=SparseAttention(96,128)
    legacy.target=copy.deepcopy(legacy.online).requires_grad_(False);legacy.target.eval()
    legacy_opt=torch.optim.AdamW([p for p in legacy.parameters() if p.requires_grad],lr=1e-4)
    old_scheduler=torch.optim.lr_scheduler.StepLR(legacy_opt,10,.5);old_state={'cursor':0}
    s=sample()
    def step(model,opt,scheduler,state):
        opt.zero_grad(set_to_none=True);loss,_=model([s]);loss.mean().backward();opt.step();model.update_ema();scheduler.step();state['cursor']+=1
    step(legacy,legacy_opt,old_scheduler,old_state)
    payload=dict(schema='v3_h_checkpoint_v1',identity={'source':'synthetic'},**snapshot(legacy,legacy_opt,old_state,old_scheduler))
    candidate=HierarchicalMAE(patch_metres=4.,lambda_jepa=.2)
    candidate_opt=torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad],lr=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(candidate_opt,10,.5);state={}
    migrated=migrate_single_parent_checkpoint(payload,legacy,candidate,candidate_opt,{'destination':'synthetic'})
    restore(migrated,candidate,candidate_opt,state,scheduler)
    assert state==old_state and scheduler.state_dict()==old_scheduler.state_dict()
    old_names=[n for n,p in legacy.named_parameters() if p.requires_grad]
    old_ids=dict(zip(old_names,payload['optimizer']['param_groups'][0]['params']))
    for name,param in candidate.named_parameters():
        if not param.requires_grad:continue
        old_name=name;offset=None
        if name.endswith(('value_weight','value_bias')) and '.cross.' in name:
            prefix,tail=name.rsplit('.',1);old_name=prefix+('.kv.weight' if tail=='value_weight' else '.kv.bias')
            offset=dict(legacy.named_parameters())[old_name].shape[0]//2
        if old_ids[old_name] in payload['optimizer']['state']:
            original=payload['optimizer']['state'][old_ids[old_name]]
            expected=original['exp_avg'][offset:] if offset is not None else original['exp_avg']
            torch.testing.assert_close(candidate_opt.state[param]['exp_avg'],expected,atol=0,rtol=0)
    step(legacy,legacy_opt,old_scheduler,old_state);step(candidate,candidate_opt,scheduler,state)
    expected=canonical_weights(legacy.state_dict())
    for name,value in candidate.state_dict().items():torch.testing.assert_close(value,expected[name],atol=3e-6,rtol=3e-4)


def test_padding_extreme_logits_and_invalid_nan_are_fully_ignored():
    block=SparseAttention(4)
    q=torch.ones(1,4,1,requires_grad=True)
    k=torch.tensor([[[-1e10]]*4,[[float('nan')]]*4],requires_grad=True)
    v=torch.tensor([[[1.]]*4,[[float('nan')]]*4],requires_grad=True)
    neighbors=torch.tensor([[0,1]]);present=torch.tensor([[True,False]])
    result=block.padded_attend(q,k,v,neighbors,present,torch.tensor([True]),torch.tensor([True,False]))
    torch.testing.assert_close(result,torch.ones(1,4),atol=0,rtol=0)
    result.sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()
    assert not k.grad[1].any() and not v.grad[1].any()
    empty=block.padded_attend(q,k,v,neighbors,torch.zeros_like(present),torch.tensor([True]),torch.tensor([False,False]))
    torch.testing.assert_close(empty,torch.zeros_like(empty),atol=0,rtol=0)


def test_invalid_nan_context_has_finite_parameter_gradients():
    block=SparseAttention(8)
    x=torch.randn(3,8,requires_grad=True);context=torch.randn(3,8)
    context[1]=float('nan');context.requires_grad_()
    edges=sparse_edges(np.array([[0,0,1,2],[0,1,1,2]]),3,'cpu')
    y=block(x,context,edges,torch.tensor([True,False,True]),torch.tensor([True,False,True]))
    y.sum().backward()
    assert torch.isfinite(y).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters())
    assert not context.grad[1].any()


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


def test_mid_run_cli_resume_matches_uninterrupted_state_and_viewer(tmp_path,monkeypatch):
    import json,sys
    from encoder_v3.hierarchical import train
    from encoder_v3.hierarchical.verification import fingerprint
    class Dataset:
        def __init__(self,role,count):
            self.samples=[sample(side=8,name=f'{role}_{i}') for i in range(count)]
            self.rows=[dict(id=s.identifier,role=role,sensor='S2',bands=[dict(shape=(8,8))]) for s in self.samples]
    datasets={'TRAIN':Dataset('TRAIN',3),'VAL':Dataset('VAL',1)}
    monkeypatch.setattr(train,'load_data',lambda *args:(datasets,None))
    monkeypatch.setattr(train,'sample_from_dataset',lambda ds,i,resolver,device:ds.samples[i])
    binding={'test':'mid-run-cli'}
    monkeypatch.setattr(train,'identity',lambda *args:binding)
    config=dict(architecture='hierarchical_v3_h',objective='observation_equal',
        scale_policy='finest_loaded_native_gsd_xy_times_2_and_4_nested_at_L',patch_metres=4.,mask_ratio=.55,encoder_depth=2,
        lambda_jepa=.2,ema_momentum=.996,task_weights={'A':.7,'B1':.3},epochs=1,batch_size=2,micro_batch_observations=1,
        micro_batch_pixels=99999,seed=17,validation_seed=1729,device='cpu',precision='float32',deterministic=True,lr=1e-4,
        weight_decay=.04,clip_norm=3.,warmup_epochs=0.,validations_per_epoch=1,checkpoint_every_batches=1,viewer_root=str(tmp_path/'viewer'))
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    auth=tmp_path/'authorization.json';auth.write_text(json.dumps({'source':'user_in_current_chat','identity':binding}))
    gate=tmp_path/'readiness.json';gate.write_text(json.dumps({'identity':binding,'training_release_ready':True}))
    run=tmp_path/'interrupted';original_write=train.write;interrupted={'done':False}
    def write(path,value):
        original_write(path,value)
        if Path(path)==run/'status.json' and value.get('cursor')==1 and not interrupted['done']:
            (run/'STOP_REQUESTED').write_text('synthetic safe mid-run stop');interrupted['done']=True
    monkeypatch.setattr(train,'write',write)
    def launch(directory,resume_path=None):
        args=['train','--config',str(config_path),'--authorization',str(auth),'--readiness',str(gate),'--run-dir',str(directory)]
        if resume_path:args+=['--resume',str(resume_path)]
        monkeypatch.setattr(sys,'argv',args);train.main()
    launch(run)
    first=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    assert first['runner']['cursor']==1 and first['runner']['status']=='STOPPED'
    viewer=copy.deepcopy(first['runner']['viewer']);(run/'STOP_REQUESTED').unlink()
    launch(run,run/'last.pt')
    restored=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    assert restored['runner']['cursor']==2 and restored['runner']['status']=='COMPLETE'
    assert restored['runner']['viewer']==viewer
    events=[json.loads(line) for line in (Path(viewer['directory'])/'events.jsonl').read_text().splitlines()]
    assert len({e['session_id'] for e in events if e['kind'] in ('started','resumed')})==2
    records=[json.loads(line) for line in (run/'updates.jsonl').read_text().splitlines()]
    assert [r['cursor'] for r in records]==[1,2]
    launch(tmp_path/'control')
    control=torch.load(tmp_path/'control/last.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler','rng','modes'):
        assert fingerprint(restored[key])==fingerprint(control[key]),key
    for key in ('cursor','spent_updates','best','epoch','observations_seen'):
        assert restored['runner'][key]==control['runner'][key]


def test_new_identity_migration_cli_inherits_journal_and_starts_new_viewer_run(tmp_path,monkeypatch):
    import json,sys,hashlib
    from encoder_v3.hierarchical import train
    from encoder_v3.hierarchical.verification import prepare_migrated_run
    original_builder=train.build_model
    def legacy_builder(config):
        m=original_builder(config)
        m.online.cross['m_to_f']=SparseAttention(64,96);m.online.cross['c_to_m']=SparseAttention(96,128)
        m.target=copy.deepcopy(m.online).requires_grad_(False);m.target.eval()
        return m
    class Dataset:
        def __init__(self,role,count):
            self.samples=[sample(side=8,name=f'{role}_{i}') for i in range(count)]
            self.rows=[dict(id=s.identifier,role=role,sensor='S2',bands=[dict(shape=(8,8))]) for s in self.samples]
    datasets={'TRAIN':Dataset('TRAIN',3),'VAL':Dataset('VAL',1)}
    monkeypatch.setattr(train,'load_data',lambda *args:(datasets,None))
    monkeypatch.setattr(train,'sample_from_dataset',lambda ds,i,resolver,device:ds.samples[i])
    phase={'identity':{'code':'legacy-synthetic'},'builder':legacy_builder}
    monkeypatch.setattr(train,'identity',lambda *args:phase['identity'])
    monkeypatch.setattr(train,'build_model',lambda config:phase['builder'](config))
    config=dict(architecture='hierarchical_v3_h',objective='observation_equal',scale_policy='finest_loaded_native_gsd_xy_times_2_and_4_nested_at_L',
        patch_metres=4.,mask_ratio=.55,encoder_depth=2,lambda_jepa=.2,ema_momentum=.996,task_weights={'A':.7,'B1':.3},
        epochs=1,batch_size=2,micro_batch_observations=1,micro_batch_pixels=99999,seed=17,validation_seed=1729,
        device='cpu',precision='float32',deterministic=True,lr=1e-4,weight_decay=.04,clip_norm=3.,warmup_epochs=0.,
        validations_per_epoch=1,checkpoint_every_batches=1,viewer_root=str(tmp_path/'viewer'))
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config));auth=tmp_path/'auth.json';gate=tmp_path/'gate.json'
    def permissions():
        auth.write_text(json.dumps({'source':'user_in_current_chat','identity':phase['identity']}))
        gate.write_text(json.dumps({'identity':phase['identity'],'training_release_ready':True}))
    source=tmp_path/'source';write_original=train.write
    def write(path,value):
        write_original(path,value)
        if Path(path)==source/'status.json' and value.get('cursor')==1 and value.get('status')=='RUNNING':
            (source/'STOP_REQUESTED').write_text('synthetic migration test')
    monkeypatch.setattr(train,'write',write)
    def launch(folder,checkpoint=None):
        args=['train','--config',str(cfg),'--authorization',str(auth),'--readiness',str(gate),'--run-dir',str(folder)]
        if checkpoint:args+=['--resume',str(checkpoint)]
        monkeypatch.setattr(sys,'argv',args);train.main()
    permissions();launch(source)
    old=torch.load(source/'last.pt',map_location='cpu',weights_only=False);old_viewer=old['runner']['viewer']
    old_sha=hashlib.sha256((source/'last.pt').read_bytes()).hexdigest()
    candidate=original_builder(config);opt=torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad],lr=1e-4)
    phase.update(identity={'code':'optimized-synthetic'},builder=original_builder)
    destination=tmp_path/'destination'
    checkpoint=prepare_migrated_run(source,source/'last.pt',destination,legacy_builder(config),candidate,opt,phase['identity'])
    permissions();launch(destination,checkpoint)
    result=torch.load(destination/'last.pt',map_location='cpu',weights_only=False)
    assert result['runner']['cursor']==2 and result['runner']['status']=='COMPLETE'
    assert result['runner']['viewer']['run_id']!=old_viewer['run_id']
    assert result['runner']['source_viewer']==old_viewer
    metadata=json.loads((Path(result['runner']['viewer']['directory'])/'run.json').read_text())
    assert metadata['config']['source_run_id']==old_viewer['run_id'] and metadata['config']['source_checkpoint_cursor']==1
    records=[json.loads(line) for line in (destination/'updates.jsonl').read_text().splitlines()]
    assert [r['cursor'] for r in records]==[1,2] and records[0]['inherited_from_run_id']==old_viewer['run_id']
    assert (destination/'best.pt').exists() and hashlib.sha256((source/'last.pt').read_bytes()).hexdigest()==old_sha
