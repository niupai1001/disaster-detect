import pytest
import torch
from landslide_seg.models.use_v2 import USEv2, initialize_pair, FrozenReadout


def observation(c=3,h=17,w=19):
    return dict(x=torch.randn(2,c,h,w),pixel_valid=torch.ones(2,c,h,w,dtype=torch.bool))


@pytest.mark.parametrize('attention',[False,True])
def test_interface(attention):
    torch.set_num_threads(2)
    model=USEv2(attention).eval()
    for c in (1,3,4,6,10,16):
        o=observation(c)
        out=model(o)
        assert out['fine'].shape==(2,32,17,19)
        assert [t.shape[-2:] for t in out['context']]==[(5,5),(3,3),(2,2),(1,1)]
        assert all(torch.isfinite(t).all() for t in [out['fine']]+out['context'])
    o=observation(); ref=model(o)
    order=torch.tensor([2,0,1])
    p={k:v[:,order] for k,v in o.items()}
    torch.testing.assert_close(model(p)['fine'],ref['fine'],atol=2e-6,rtol=2e-5)
    o['pixel_valid'][:,1]=False
    masked=model(o)
    removed=model({k:v[:,[0,2]] for k,v in o.items()})
    for a,b in zip([masked['fine']]+masked['context'],[removed['fine']]+removed['context']):
        torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
    o['x'][:,1]=float('nan')
    torch.testing.assert_close(model(o)['fine'],masked['fine'])
    o['pixel_valid'][:]=False
    with pytest.raises(ValueError): model(o)


def test_single_band_fine_and_common_initialization():
    n,a=initialize_pair()
    for k,v in n.state_dict().items(): torch.testing.assert_close(v,a.state_dict()[k],rtol=0,atol=0)
    o=observation(1,8,8); o['x'].zero_(); o['x'][:,:,0,1]=2
    f=a(o)['fine']
    assert not torch.allclose(f[:,:,0,0],f[:,:,0,1])


def test_frozen_nested_state():
    head=FrozenReadout(USEv2(),fine=True)
    head.train()
    optimizer=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad])
    head(observation()).mean().backward(); optimizer.step()
    head.use.verify(optimizer)
    assert not any(t.requires_grad for t in head.use(observation())['context'])


def test_hidden_values_never_reach_readout():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from pretrain_use_v2 import task_view,initialize,objective
    o=observation(3,33,35)
    o['band_valid']=torch.ones(2,3,dtype=torch.bool)
    o['band_metadata']=torch.tensor([.5,.1,10.]).expand(2,3,3).clone()
    o['known_mask']=torch.ones(2,3,3,dtype=torch.bool)
    for task in ('spectral','fine','context'):
        view,hidden,summary=task_view(o,task,22)
        changed={k:v.clone() for k,v in o.items()}
        changed['x'][hidden]=12345
        other,mask,other_summary=task_view(changed,task,22)
        assert torch.equal(hidden,mask)
        torch.testing.assert_close(view['x'],other['x'])
        torch.testing.assert_close(summary,other_summary)
        result=objective(initialize('mean'),o,task,22)
        if result is not None:
            result[0].backward()
            assert torch.isfinite(result[0])


def test_metadata_and_empty_windows():
    model=USEv2().eval()
    o=observation(3,33,35)
    o['pixel_valid'][:,:,4:]=False
    original=model(o)
    o['physical']=torch.zeros(2,3,3)
    o['known_mask']=torch.zeros(2,3,3,dtype=torch.bool)
    unknown=model(o)
    for a,b in zip([original['fine']]+original['context'],[unknown['fine']]+unknown['context']):
        torch.testing.assert_close(a,b)
        assert torch.isfinite(b).all()
    assert not unknown['fine'][:,:,4:].any()
    o['physical'][:,:,0]=500
    o['known_mask'][:,:,0]=True
    with pytest.raises(ValueError): model(o)


def test_diagnostic_error_definition():
    from evaluate_use_v2 import errors,baseline
    target=torch.zeros(1,1,2,2)
    pred=torch.tensor([[[[1.,3.],[99.,99.]]]])
    mask=torch.tensor([[[[True,True],[False,False]]]])
    result=errors(pred,target,mask)
    assert result['mae']==2
    assert result['rmse']==pytest.approx(5**.5)
    assert result['adjacent_difference_mae']==2
    o={'x':torch.ones(1,2,9,9),'pixel_valid':torch.ones(1,2,9,9,dtype=torch.bool)}
    o['x'][:,:,4,4]=0; o['pixel_valid'][:,:,4,4]=False
    _,local=baseline(o)
    torch.testing.assert_close(local,torch.ones_like(local))


def test_all_spatial_stages_receive_gradients():
    from pretrain_use_v2 import initialize
    model=initialize('mean')
    o=observation(3,32,32)
    result=model(o,torch.zeros(2,3,3),torch.zeros(2,3,3,dtype=torch.bool),torch.zeros(2,3,2))
    result.square().mean().backward()
    for stage in model.use.stages:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in stage.parameters())


def test_probe_capacity_and_freeze():
    from pretrain_use_v2 import initialize,RecoveryProbe
    from landslide_seg.models.frozen_use import state_sha256
    heads=[]
    for task in ('fine','context'):
        probe=RecoveryProbe(initialize('mean').use,task)
        heads.append({k:v.clone() for k,v in probe.state_dict().items() if not k.startswith('use.')})
        probe.train(); opt=torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad])
        o=observation(3,32,32)
        result=probe(o,torch.zeros(2,3,3),torch.zeros(2,3,3,dtype=torch.bool),torch.zeros(2,3,2))
        result.mean().backward(); opt.step(); probe.use.verify(opt)
    assert heads[0].keys()==heads[1].keys()
    for k in heads[0]: torch.testing.assert_close(heads[0][k],heads[1][k],atol=0,rtol=0)


def test_optimizer_and_rng_resume_equivalence(tmp_path):
    from pretrain_use_v2 import initialize,checkpoint,atomic_save,restore,state_sha256
    torch.set_num_threads(2)
    o=observation(3,16,16)
    def update(m,opt):
        opt.zero_grad(); result=m(o,torch.zeros(2,3,3),torch.zeros(2,3,3,dtype=torch.bool),torch.zeros(2,3,2))
        # Stochastic readout term makes restoring RNG necessary to pass.
        (result*torch.rand_like(result)).mean().backward(); opt.step()
    m=initialize('mean'); opt=torch.optim.AdamW(m.parameters())
    update(m,opt); atomic_save(checkpoint(m,opt,1,1.,{'id':'test'},[]),tmp_path/'last.pt')
    update(m,opt); expected=state_sha256(m); random_expected=torch.rand(3)
    other=initialize('mean'); other_opt=torch.optim.AdamW(other.parameters())
    restore(other,other_opt,tmp_path/'last.pt',{'id':'test'}); update(other,other_opt)
    assert state_sha256(other)==expected
    torch.testing.assert_close(torch.rand(3),random_expected,atol=0,rtol=0)
    with pytest.raises(ValueError): restore(other,other_opt,tmp_path/'last.pt',{'id':'wrong'})


def test_matrix_and_formal_gate(tmp_path):
    from run_use_v2 import matrix,verify_release,write_json
    jobs=matrix()
    assert len(jobs)==30
    assert sum(j['steps'] for j in jobs.values())==92000
    write_json(tmp_path/'manifest.json',dict(status='INCOMPLETE'),True)
    (tmp_path/'manifest.sha256').write_text('invalid')
    with pytest.raises(ValueError): verify_release(tmp_path)


def test_formal_source_checkpoint_strict_gate(tmp_path,monkeypatch):
    import run_use_v2 as runtime
    from audit_use_independent_data import sha
    monkeypatch.setattr(runtime,'BASE',tmp_path)
    job=runtime.matrix()['probe_mean_u5000_fine']
    path=tmp_path/'runs'/job['source']; path.parent.mkdir(parents=True)
    path.write_bytes(b'checkpoint')
    runtime.write_json(path.parent/'summary.json',dict(complete=True,updates=20000,checkpoints={path.name:sha(path)}),True)
    assert runtime.source_checkpoint(job)==path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError): runtime.source_checkpoint(job)


def test_metadata_modes_share_spatial_masks_and_head_initialization():
    from pretrain_use_v2 import task_view,initialize
    from run_use_v2 import build,matrix
    o=observation(3,32,32); o['band_valid']=torch.ones(2,3,dtype=torch.bool)
    o['band_metadata']=torch.tensor([.5,.1,10.]).expand(2,3,3).clone()
    o['known_mask']=torch.ones(2,3,3,dtype=torch.bool)
    masks=[task_view(o,'fine',555,metadata_mode=mode)[1] for mode in ('known','partial','unknown')]
    assert all(torch.equal(masks[0],m) for m in masks)
    reference=None
    for candidate in ('mean','attention'):
        for fine in (False,True):
            job=matrix()[f'downstream_{candidate}_{"context_fine" if fine else "context"}']
            m=build(job,engineering=True)
            head={k:v for k,v in m.state_dict().items() if not k.startswith(('use.','fine.'))}
            if reference is None: reference=head
            else:
                for k in reference: torch.testing.assert_close(head[k],reference[k],atol=0,rtol=0)


def test_physical_spectral_mean_and_spatial_masked_targets():
    from evaluate_use_v2 import baseline
    x=torch.tensor([[[[1.]],[[2.]],[[0.]]]])
    v=torch.tensor([[[[True]],[[True]],[[False]]]])
    statistics={'mean':[.1,.2,.3],'std':[.1,.2,.3]}
    spectral,_=baseline(dict(x=x,pixel_valid=v),statistics)
    # Physical visible values are .2 and .6; their mean .4 is target-zscored.
    torch.testing.assert_close(spectral[0,2,0,0],torch.tensor((.4-.3)/.3))


def test_resampled_footprint_fraction_includes_partial_edge_cells():
    from evaluate_use_v2 import footprint_coverage
    from affine import Affine
    import numpy as np
    source=Affine(20,0,500000,0,-20,100000)
    target=Affine(30,0,500000,0,-30,100000)
    fractions=footprint_coverage(source,(4,4),'EPSG:32632',target,(3,3),'EPSG:32632')
    np.testing.assert_allclose(fractions[:2,:2],1)
    assert fractions[2,2]==pytest.approx(4/9)
    assert fractions.sum()*900==pytest.approx(80*80)


def test_paired_bootstrap_and_geographic_zero_difference():
    from evaluate_use_v2 import paired_bootstrap,geographic_bootstrap
    left={'s2':{'records':[{'file':'a','task':'fine','model':{'mae':2.}}]}}
    right={'s2':{'records':[{'file':'a','task':'fine','model':{'mae':1.}}]}}
    difference=paired_bootstrap(left,right,20)[0]
    assert difference['mean']==1 and difference['ci95']==[1.,1.]
    tiles=[dict(sample_id=str(i),event_id='hiroshima',bbox=[500000+i*20000,0,501000+i*20000,1000],
                crs='EPSG:32632',tp=5,fp=1,fn=2) for i in range(3)]
    evaluation=dict(tiles=tiles,event_mean_iou=5/8)
    result=geographic_bootstrap(evaluation,evaluation,20)
    assert result['delta_event_mean_iou']==0 and result['ci95']==[0.,0.]


def test_calibration_members_are_train_only():
    from run_use_v2 import load_configuration
    _,cal,splits,_=load_configuration()
    for sensor,files in cal['members'].items():
        assert len(files)==32
        assert set(files)<={r['file'] for r in splits['TRAIN'][sensor]}
        assert not set(files)&{r['file'] for r in splits['PRETRAIN_VAL'][sensor]}
        assert all(v>0 for v in cal['coefficients'][sensor].values())
