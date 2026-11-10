from dataclasses import replace
from pathlib import Path
import copy
import json
import torch
import pytest
from encoder_v3.latent_response.types import NativeGrid
from encoder_v3.shared_mae.geometry import AreaCache, metric_geometry
from encoder_v3.shared_mae.metadata import MetadataResolver, PhysicalChannel, MissingMetadata
from encoder_v3.shared_mae.model import SharedMAE, SharedMemory, NativeChannel, Sample
from encoder_v3.shared_mae.data import EpochPlan


torch.set_num_threads(2)


def sample(identifier='synthetic', side=16, channels=3, role='TRAIN', scale=1., offset=(0.,0.)):
    result = []
    for c in range(channels):
        grid = NativeGrid((scale,0.,offset[0],0.,-scale,offset[1]+side*scale),(side,side),'EPSG:32631')
        physical = PhysicalChannel('optical','surface_reflectance',scale,scale,.45+.2*c,.04,source='synthetic:test')
        result.append(NativeChannel(torch.rand(side,side),torch.ones(side,side,dtype=torch.bool),grid,physical,str(c)))
    return Sample(identifier, role, result)


def model():
    return SharedMAE(patch_metres=4.,width=16,layout=4,encoder_depth=2,decoder_depth=2,decoder_width=16)


def test_physical_metadata_sidecar_catalog_missing_and_no_sensor_parameters(tmp_path):
    path=tmp_path/'catalog.json';path.write_text(json.dumps(dict(sensors={'NEW':dict(bands=[dict(band='B2',centre_nm=490.,fwhm_nm=60.)])})))
    resolver=MetadataResolver(path)
    band=dict(band_key='B02',transform=(10.,0.,0.,0.,-10.,100.))
    row=dict(id='image',sensor='future_sensor',instrument='NEW',modality='optical',units='surface_reflectance')
    resolved=resolver.resolve(row,band)
    assert resolved.center_um==.49 and resolved.fwhm_um==.06 and '#sha256=' in resolved.source
    row['channel_metadata']={'B02':dict(center_um=.51,source='product:metadata.xml')}
    assert resolver.resolve(row,band).center_um==.51
    with pytest.raises(MissingMetadata):resolver.resolve(dict(row,channel_metadata={'B02':dict(source='missing')}),band)
    sar=PhysicalChannel('sar','gamma0_linear',10,10,frequency_ghz=5.405,polarization='VV',source='synthetic:sar')
    memory=SharedMemory()
    assert not any('sensor' in n or 'registry' in n for n,_ in memory.named_parameters())
    first=memory([resolved,sar]);second=memory([replace(resolved,source='another satellite'),sar])
    torch.testing.assert_close(first['conv'],second['conv'])
    assert not torch.equal(first['conv'][0],first['conv'][1])
    first['conv'].square().sum().backward()
    assert memory.slots.grad.abs().sum()>0


def test_exact_area_constant_fields_inverse_resolution_and_edge_coverage():
    a=sample(side=16,scale=1.)
    b=sample(side=32,scale=.5)
    ga,gb=metric_geometry(a.channels,4),metric_geometry(b.channels,4)
    assert ga.shape==gb.shape==(4,4)
    cache=AreaCache();mapping=cache.get(a.channels[0].grid,ga,4,'cpu')
    assert cache.get(a.channels[0].grid,ga,4,'cpu') is mapping
    torch.testing.assert_close(mapping.native_prediction(torch.ones(16,16)),torch.ones(256))
    c=sample(side=9,scale=1.3,offset=(100.2,300.7));g=metric_geometry(c.channels,4)
    m=cache.get(c.channels[0].grid,g,4,'cpu')
    torch.testing.assert_close(m.native_prediction(torch.ones(g.shape[0]*g.shape[1],16)),torch.ones(81),atol=1e-6,rtol=1e-6)
    shifted=sample(side=16,scale=1.,offset=(500000.,4000000.))
    shifted_geometry=metric_geometry(shifted.channels,4)
    assert cache.get(shifted.channels[0].grid,shifted_geometry,4,'cpu') is mapping
    fractional=sample(side=16,scale=1.,offset=(.3,.7))
    # A shift of the entire scene leaves all relative footprints unchanged.
    assert cache.get(fractional.channels[0].grid,metric_geometry(fractional.channels,4),4,'cpu') is mapping
    combined=metric_geometry(a.channels+fractional.channels,4)
    first=cache.get(a.channels[0].grid,combined,4,'cpu')
    different=cache.get(fractional.channels[0].grid,combined,4,'cpu')
    assert first is not different
    torch.testing.assert_close(different.native_prediction(torch.ones(combined.shape[0]*combined.shape[1],16)),torch.ones(256))


@pytest.mark.parametrize('kind',['A','B1'])
def test_hidden_value_poisoning_predictions_and_input_gradients(kind):
    torch.manual_seed(4);m=model();s=sample()
    loss,details=m([s],kind=kind,return_predictions=True)
    poisoned=copy.deepcopy(s)
    for c,mask in zip(poisoned.channels,details[0]['target_masks']):c.values[mask]=1e4
    _,again=m([poisoned],kind=kind,return_predictions=True)
    for first,second in zip(details[0]['predictions'],again[0]['predictions']):
        if first is not None:torch.testing.assert_close(first,second,atol=0,rtol=0)
    for c in s.channels:c.values.requires_grad_()
    _,result=m([s],kind=kind,return_predictions=True)
    sum(p.sum() for p in result[0]['predictions'] if p is not None).backward()
    for c,mask in zip(s.channels,result[0]['target_masks']):
        if c.values.grad is not None:assert torch.count_nonzero(c.values.grad[mask])==0
    assert details[0]['encoder_tokens']<details[0]['patches'] if kind=='A' else details[0]['encoder_tokens']==details[0]['patches']


def test_multi_scene_batch_isolation_gradients_and_conditional_bias():
    torch.manual_seed(8);m=model();a,b=sample('a'),sample('b',side=20,scale=.7)
    batch,_=m([a,b],kind='A');separate=torch.cat([m([s],kind='A')[0] for s in (a,b)])
    torch.testing.assert_close(batch,separate,atol=1e-6,rtol=1e-5)
    batch.mean().backward();actual={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    for s in (a,b):(m([s],kind='A')[0].sum()/2).backward()
    for n,p in m.named_parameters():
        if p.grad is not None:torch.testing.assert_close(p.grad,actual[n],atol=2e-5,rtol=2e-4)
    assert m.memory.slots.grad.abs().sum()>0 and m.convolution.bias_bases.grad.abs().sum()>0
    condition=m.memory([c.physical for c in a.channels])['conv']
    _,bias=m.convolution(condition)
    assert not torch.equal(bias[0],bias[1])


def test_single_channel_sampling_and_spatial_task():
    rows=[dict(id=str(i),role='TRAIN',sensor='SAR',bands=[{}]) for i in range(3)]
    plan=EpochPlan(rows,17,2)
    assert len(plan)==2 and [len(plan.batch(i)) for i in range(2)]==[2,1]
    s=sample(channels=1)
    s.channels[0].physical=PhysicalChannel('sar','gamma0_linear',1,1,frequency_ghz=5.405,polarization='VV',source='synthetic:test')
    loss,_=model()([s],kind='A');assert torch.isfinite(loss).all()
    with pytest.raises(ValueError,match='two valid channels'):model()([s],kind='B1')


def test_ordered_patch_embedding_distinguishes_internal_positions():
    m=model();first=torch.zeros(1,4*4*33);second=first.clone()
    first[0,0]=1;second[0,32]=1
    assert not torch.equal(m.patch_embedding(first),m.patch_embedding(second))


def test_global_attention_connects_distant_tokens_and_channel_permutation():
    from encoder_v3.shared_mae.model import GlobalTransformer
    block=GlobalTransformer(16,1)
    x=torch.randn(1,16,requires_grad=True);y=torch.randn(1,16,requires_grad=True)
    features=torch.cat((x,y))
    result=block([features],[torch.tensor([[0,0],[1000,1000]])])[0]
    result[0,0].backward()
    assert y.grad.abs().sum()>0
    torch.manual_seed(9);m=model();s=sample();permuted=copy.deepcopy(s)
    permuted.channels=list(reversed(permuted.channels))
    first,_=m([s],kind='A');second,_=m([permuted],kind='A')
    torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-5)


def test_checkpoint_restores_exact_next_update_and_rejects_legacy(tmp_path):
    from encoder_v3.shared_mae.train import save,resume
    from encoder_v3.latent_response.checkpoint import parameter_sha
    torch.manual_seed(23);m=model();s=sample();opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=2,gamma=.5)
    state=dict(cursor=0);binding={'test':'synthetic-only'}
    def advance():
        opt.zero_grad(set_to_none=True)
        loss,_=m([s],kind='A',epoch=state['cursor']);loss.mean().backward()
        opt.step();scheduler.step();state['cursor']+=1
    advance();save(tmp_path/'last.pt',m,opt,scheduler,state,binding)
    advance();expected=parameter_sha(m)
    resume(tmp_path/'last.pt',m,opt,scheduler,state,binding);advance()
    assert parameter_sha(m)==expected and state['cursor']==2
    torch.save({'schema':'lr_checkpoint_v1.3'},tmp_path/'legacy.pt')
    with pytest.raises(ValueError,match='identity mismatch'):
        resume(tmp_path/'legacy.pt',m,opt,scheduler,state,binding)


def test_mixed_resolution_hidden_isolation():
    torch.manual_seed(31);m=model();s=sample()
    coarse=sample(side=8,scale=2.,channels=1).channels[0]
    s.channels[-1]=coarse
    _,details=m([s],kind='A',return_predictions=True)
    poisoned=copy.deepcopy(s)
    for c,mask in zip(poisoned.channels,details[0]['target_masks']):c.values[mask]=float('nan')
    _,again=m([poisoned],kind='A',return_predictions=True)
    for a,b in zip(details[0]['predictions'],again[0]['predictions']):
        if a is not None:torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_full_epoch_coverage_tail_and_microbatch_objective(monkeypatch):
    from encoder_v3.shared_mae import train
    rows=[dict(id=str(i),role='TRAIN',sensor='S2',bands=[dict(shape=(16,16))]) for i in range(7442)]
    plan=EpochPlan(rows,17,32)
    ids=[r['id'] for i in range(len(plan)) for r in plan.batch(i)]
    assert len(plan)==233 and len(plan.batch(232))==18 and len(set(ids))==7442
    assert ids!=[r['id'] for r in rows]
    torch.manual_seed(3);m=model();samples=[sample(str(i),channels=1 if i==0 else 3) for i in range(3)]
    selected=rows[:3]
    class Dataset:pass
    ds=Dataset();ds.rows=selected
    monkeypatch.setattr(train,'sample_from_dataset',lambda dataset,i,resolver,device:samples[i])
    config=dict(device='cpu',seed=17,micro_batch_observations=3,micro_batch_pixels=999999)
    loss,_=train.batch_loss(m,ds,selected,None,config,backward=True)
    expected={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    actual,_=train.batch_loss(m,ds,selected,None,dict(config,micro_batch_observations=1),backward=True)
    assert abs(loss-actual)<1e-6
    for n,p in m.named_parameters():
        if p.grad is not None:torch.testing.assert_close(p.grad,expected[n],atol=2e-5,rtol=2e-4)


def test_bfloat16_mask_insertion_and_float32_module_gradients():
    torch.manual_seed(17);m=model();s=sample()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        loss,_=m([s],kind='A')
    assert loss.dtype==torch.float32 and torch.isfinite(loss).all()
    # CUDA BF16 backward is checked by precision_check; this CPU lacks BF16 backward support.
    full_precision,_=m([s],kind='A');full_precision.mean().backward()
    for prefix in ('memory','convolution','patch_embedding','fusion','position','encoder','decoder','output'):
        gradients=[p.grad for n,p in m.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients)>0,prefix


def test_validation_transaction_and_real_checkpoint_viewer_session(tmp_path,monkeypatch):
    from encoder_v3.shared_mae import train
    from encoder_v3.latent_response.checkpoint import parameter_sha
    torch.manual_seed(19);m=model();s=sample(role='VAL')
    class Dataset:pass
    ds=Dataset();ds.rows=[dict(id=s.identifier,role='VAL',sensor='S2',bands=[dict(shape=(16,16))])]
    monkeypatch.setattr(train,'sample_from_dataset',lambda *args:s)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    state=dict(cursor=0);config=json.loads((Path(__file__).resolve().parents[3]/'configs/config_shared_mae_v1.json').read_text())
    config.update(device='cpu',patch_metres=4.)
    before=parameter_sha(m);rng=torch.get_rng_state().clone()
    result=train.validate(m,ds,None,opt,scheduler,state,config)
    assert parameter_sha(m)==before and torch.equal(torch.get_rng_state(),rng) and m.training
    monkeypatch.setattr(train,'training_runs_root',lambda:tmp_path/'training-runs')
    binding={'synthetic_test':True};run=tmp_path/'artifacts'
    with train.writer_for(config,binding,state,run) as writer:
        writer.log(step=0,epoch=0.,values={'val_loss':result['metric']})
        first_id=writer.meta['run_id'];first_session=writer.session
        train.save(tmp_path/'state.pt',m,opt,scheduler,state,binding)
    with pytest.raises(ValueError,match='checkpoint load'):
        train.writer_for(config,binding,state,run)
    train.resume(tmp_path/'state.pt',m,opt,scheduler,state,binding)
    with train.writer_for(config,binding,state,run,resuming=True) as writer:
        assert writer.meta['run_id']==first_id and writer.session!=first_session
        assert writer.meta['view']['charts']==['train_loss','val_loss']
        assert writer.meta['metrics']['val_loss']['unit']=='z²'
        writer.log(step=0,epoch=0.,values={'val_loss':result['metric']})
    directory=Path(state['viewer']['directory'])
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    assert [e['kind'] for e in events]==['started','completed','resumed','completed']


def test_fixed_learning_cohort_is_distinct_balanced_and_deterministic():
    from encoder_v3.shared_mae.check import learning_cohort
    class Dataset:pass
    ds=Dataset();ds.rows=[dict(id=f'{sensor}_{i}',sensor=sensor,bands=[dict(shape=(i+1,i+1))])
                          for sensor in ('S2','L8','EnMAP') for i in range(40)]
    chosen=learning_cohort(ds)
    assert len(chosen)==len({r['id'] for r in chosen})==24
    assert all(sum(r['sensor']==sensor for r in chosen)==8 for sensor in ('S2','L8','EnMAP'))
    ds.rows.reverse()
    assert [r['id'] for r in chosen]==[r['id'] for r in learning_cohort(ds)]


@pytest.mark.parametrize('kind',['A','B1'])
def test_grouped_heads_equal_native_channel_mean_with_invalid_nan(kind):
    torch.manual_seed(41);m=model();s=sample()
    s.channels[-1]=sample(side=8,scale=2.,channels=1).channels[0]
    for c in s.channels:
        c.valid[::3,::2]=False;c.values[~c.valid]=float('nan')
    losses,detail=m([s],kind=kind,return_predictions=True)
    individual=[]
    for c,pred,mask in zip(s.channels,detail[0]['predictions'],detail[0]['target_masks']):
        if pred is not None:individual.append((pred[mask]-c.values[mask]).square().mean())
    expected=torch.stack(individual).mean()
    torch.testing.assert_close(losses[0],expected)
    actual=torch.autograd.grad(losses[0],tuple(m.parameters()),retain_graph=True,allow_unused=True)
    reference=torch.autograd.grad(expected,tuple(m.parameters()),allow_unused=True)
    for a,b in zip(actual,reference):
        if a is not None:torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-4)
