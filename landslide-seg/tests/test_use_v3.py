import pytest
import torch
from landslide_seg.models.use_v3 import USEv3
from landslide_seg.models.use_v3_reconstruction import NativeProjection,Reconstruction


def obs(c=3,h=17,w=19):
    return dict(x=torch.randn(2,c,h,w),pixel_valid=torch.ones(2,c,h,w,dtype=torch.bool))


@pytest.mark.parametrize('c',[1,3,6,10,16])
def test_interface(c):
    torch.set_num_threads(2)
    m=USEv3().eval(); o=obs(c); f=m(o,return_pyramid=True)
    assert f['main'].shape==(2,32,5,5)
    assert f['fine'].shape==(2,32,17,19)
    assert [t.shape[1] for t in f['context']]==[32,64,128,256]
    order=torch.randperm(c)
    p={k:v[:,order] for k,v in o.items()}
    torch.testing.assert_close(m(p)['fine'],f['fine'],atol=3e-6,rtol=3e-5)
    if c>1:
        o['pixel_valid'][:,0]=False
        expected=m(o)['fine']
        o['x'][:,0]=float('nan')
        torch.testing.assert_close(m(o)['fine'],expected)
        torch.testing.assert_close(m({k:v[:,1:] for k,v in o.items()})['fine'],expected,atol=3e-6,rtol=3e-5)
    o['pixel_valid'].zero_()
    with pytest.raises(ValueError): m(o)


def test_feedback_and_native_projection():
    torch.set_num_threads(2)
    m=USEv3(); f=m(obs(),return_pyramid=True)
    f['fine'].square().mean().backward()
    assert m.inject.weight.grad.abs().sum()>0
    assert m.projections[-1].weight.grad.abs().sum()>0
    p=NativeProjection()
    t=[torch.randn(1,d,2,2) for d in (32,64,128,256)]
    p(t,(4,4)).sum().backward()
    assert all(layer.weight.grad is not None for layer in p.layers)


@pytest.mark.parametrize('task',['fine','context'])
def test_frozen(task):
    torch.set_num_threads(2); torch.manual_seed(1235)
    m=Reconstruction(USEv3(),task).train()
    optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad])
    o=obs(); z=torch.zeros(2,3,3); known=z.bool(); summary=torch.zeros(2,3,2)
    m(o,z,known,summary).mean().backward(); optimizer.step()
    m.use.verify(optimizer)


def test_same_seed_and_resume(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from pretrain_use_v3 import initialize,checkpoint,restore,atomic_save
    from landslide_seg.models.frozen_use import state_sha256
    torch.set_num_threads(2)
    a=initialize(); b=initialize()
    assert state_sha256(a)==state_sha256(b)
    oa=torch.optim.AdamW(a.parameters()); ob=torch.optim.AdamW(b.parameters())
    o=obs(); meta=torch.zeros(2,3,3); known=meta.bool(); summary=torch.zeros(2,3,2)
    for _ in range(2):
        oa.zero_grad(); a(o,meta,known,summary).square().mean().backward(); oa.step()
        if _==0:
            atomic_save(checkpoint(a,oa,1,None,{'test':1},[]),tmp_path/'a.pt')
    restore(b,ob,tmp_path/'a.pt',{'test':1})
    ob.zero_grad(); b(o,meta,known,summary).square().mean().backward(); ob.step()
    assert state_sha256(a)==state_sha256(b)
    with pytest.raises(ValueError): restore(b,ob,tmp_path/'a.pt',{'test':2})


def test_microbatch_gradient():
    torch.set_num_threads(2); torch.manual_seed(1)
    a=USEv3(); b=USEv3(); b.load_state_dict(a.state_dict()); o=obs()
    a(o)['fine'].square().mean().backward()
    for i in range(2):
        b({k:v[i:i+1] for k,v in o.items()})['fine'].square().mean().div(2).backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.grad is not None: torch.testing.assert_close(pa.grad,pb.grad,atol=2e-6,rtol=1e-4)


def test_physical_metadata_permutation_and_partial():
    m=USEv3().eval(); o=obs(3)
    o['band_metadata']=torch.tensor([[.5,.1,10.],[.8,.2,20.],[1.6,.3,20.]]).expand(2,3,3).clone()
    o['known_mask']=torch.ones(2,3,3,dtype=torch.bool)
    o['known_mask'][:,1,1]=False; o['band_metadata'][:,1,1]=0
    ref=m(o)['fine']; order=torch.tensor([2,0,1])
    torch.testing.assert_close(m({k:v[:,order] for k,v in o.items()})['fine'],ref,atol=3e-6,rtol=3e-5)
    o['known_mask'].zero_(); o['band_metadata'].zero_()
    assert torch.isfinite(m(o)['main']).all()


def test_cross_window_gradient():
    from landslide_seg.models.use_v3 import SpatialBlock
    torch.manual_seed(1)
    blocks=torch.nn.ModuleList([SpatialBlock(32),SpatialBlock(32,True)])
    x=torch.randn(1,32,8,8,requires_grad=True); valid=torch.ones(1,8,8,dtype=torch.bool)
    y=x
    for block in blocks: y=block(y,valid)
    y[:,:,3,3].square().sum().backward()
    assert x.grad[:,:,4,4].abs().sum()>0


def test_projection_retains_cancelling_direction():
    p=NativeProjection()
    features=[torch.zeros(1,d,1,1) for d in (32,64,128,256)]
    base=p(features,(1,1))
    features[1][:,0]=1; features[1][:,1]=-1
    assert not torch.allclose(base,p(features,(1,1)))


def test_job_budget_and_formal_gate(tmp_path):
    import json,sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_use_v3 as run
    assert sum(j['steps'] for j in run.matrix().values())==62000
    assert len(run.matrix())==17
    assert not any(j.get('point')=='5000' for j in run.matrix().values())
    (tmp_path/'manifest.json').write_text(json.dumps(dict(files={})))
    (tmp_path/'marker.json').write_text(json.dumps(dict(status='READY_FOR_TRAINING',release_sha256='wrong')))
    with pytest.raises(ValueError,match='Independent READY_FOR_TRAINING'):
        run.run_job('pretrain','cpu',tmp_path,tmp_path/'marker.json')


def test_probe_metadata_layers_and_selection(monkeypatch,tmp_path):
    import sys,json
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import evaluate_use_v3 as evaluation
    import pretrain_use_v3 as pt
    torch.set_num_threads(2)
    o=obs(3); o['band_valid']=torch.ones(2,3,dtype=torch.bool)
    o['band_metadata']=torch.tensor([.5,.1,10.]).expand(2,3,3).clone(); o['known_mask']=torch.ones(2,3,3,dtype=torch.bool)
    monkeypatch.setattr(pt,'load_rows',lambda *args,**kwargs:o)
    splits={'PRETRAIN_VAL':{s:[dict(file='fixed')] for s in ('s2','landsat')}}
    calibration={'coefficients':{s:{k:1. for k in ('spectral','fine','context')} for s in ('s2','landsat')}}
    model=pt.initialize(task='context')
    for mode in ('known','partial','unknown'):
        result=evaluation.evaluate(model,'context',splits,{},calibration,'cpu',mode)
        assert result['metadata_mode']==mode
        if mode=='unknown': assert result['sensors']['s2']['means']['spectral'] is None
        else: assert result['sensors']['s2']['means']['context'] is not None
    for step in (10000,20000,30000):
        for task in ('fine','context'):
            path=tmp_path/f'probe_v3_{step}_{task}';path.mkdir()
            means={'fine':1.} if task=='fine' else {'spectral':1.,'context':1.}
            (path/'evaluation.json').write_text(json.dumps(dict(metadata_mode='known',sensors={s:dict(means=means) for s in ('s2','landsat')})))
    (tmp_path/'pretrain').mkdir();(tmp_path/'pretrain/u10000.pt').write_bytes(b'fixed')
    result=evaluation.select(tmp_path,tmp_path/'selection.json')
    assert result['step']==10000
    assert (tmp_path/'pretrain/best_unlabeled.pt').read_bytes()==b'fixed'


def test_probe_common_initial_tensors():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import pretrain_use_v3 as pt
    from landslide_seg.models.use_v2 import USEv2
    for task in ('fine','context'):
        a=pt.initialize(task=task,encoder=USEv2()); b=pt.initialize(task=task,encoder=USEv3())
        sa={k:v for k,v in a.state_dict().items() if not k.startswith('use.')}
        sb={k:v for k,v in b.state_dict().items() if not k.startswith('use.')}
        assert sa.keys()==sb.keys()
        assert all(torch.equal(sa[k],sb[k]) for k in sa)


@pytest.mark.parametrize('damage',['missing_current','missing_earlier','wrong_identity'])
def test_milestone_interruption_and_resume(monkeypatch,tmp_path,damage):
    import sys,json
    from pathlib import Path
    from contextlib import nullcontext
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_use_v3 as run
    root=tmp_path/'work';root.mkdir(); config=root/'execution.yaml';config.write_text('{}')
    (root/'engineering').mkdir();(root/'engineering/calibration.json').write_text('{}')
    release=tmp_path/'release';release.mkdir();(release/'manifest.json').write_text('{}')
    marker=tmp_path/'marker.json';marker.write_text(json.dumps(dict(status='READY_FOR_TRAINING',release_sha256=run.sha(release/'manifest.json'))))
    monkeypatch.setattr(run,'BASE',root);monkeypatch.setattr(run,'CONFIG',config)
    monkeypatch.setattr(run,'PRETRAIN_MILESTONES',(1,2))
    monkeypatch.setattr(run,'matrix',lambda:{'pretrain':dict(mode='pretrain',steps=2,warmup=1)})
    monkeypatch.setattr(run,'verify_release',lambda release:None)
    monkeypatch.setattr(run.pt,'inputs',lambda *args:({'TRAIN':{}},{},'audit','members'))
    monkeypatch.setattr(run.pt,'initialize',lambda device:torch.nn.Linear(1,1))
    monkeypatch.setattr(run.rt,'gpu_guard',lambda device:nullcontext({'device':'cpu'}))
    calls=[]
    monkeypatch.setattr(run.pt,'update',lambda *args: calls.append(args[4]) or dict(loss=0.))
    monkeypatch.setattr(run.pt,'checkpoint',lambda model,opt,step,best,identity,history:dict(step=step,best=best,identity=identity,history=list(history)))
    monkeypatch.setattr(run.pt,'restore',lambda model,opt,path,identity:torch.load(path,weights_only=False))
    original_save=run.pt.atomic_save;order=[]
    def interrupt(payload,path):
        order.append(path.name);original_save(payload,path)
        if path.name=='last.pt': raise RuntimeError('after last commit')
    monkeypatch.setattr(run.pt,'atomic_save',interrupt)
    with pytest.raises(RuntimeError,match='after last commit'):run.run_job('pretrain','cpu',release,marker)
    out=root/'runs/pretrain'
    assert order.index('u2.pt')<order.index('last.pt')
    assert not json.loads((out/'summary.json').read_text())['complete']
    updates_before=len(calls)
    if damage=='missing_current': (out/'u2.pt').unlink()
    elif damage=='missing_earlier': (out/'u1.pt').unlink()
    else:
        state=torch.load(out/'u1.pt',weights_only=False);state['identity']={};original_save(state,out/'u1.pt')
    monkeypatch.setattr(run.pt,'atomic_save',original_save)
    if damage=='missing_current':
        run.run_job('pretrain','cpu',release,marker,resume=True)
        last=torch.load(out/'last.pt',weights_only=False);recovered=torch.load(out/'u2.pt',weights_only=False)
        assert recovered==last
        summary=json.loads((out/'summary.json').read_text())
        assert summary['complete'] and set(summary['checkpoints'])=={'initial.pt','last.pt','u1.pt','u2.pt'}
    else:
        with pytest.raises(ValueError,match='checkpoint|Checkpoint'):run.run_job('pretrain','cpu',release,marker,resume=True)
        assert not json.loads((out/'summary.json').read_text())['complete']
    assert len(calls)==updates_before


def test_packed_frozen_boundary():
    from landslide_seg.models.use_v3_reconstruction import FrozenUSE,packed_state_sha256
    from landslide_seg.models.frozen_use import state_sha256
    model=torch.nn.Linear(3,2)
    model.register_buffer('runtime',torch.arange(12).reshape(3,4).T,persistent=False)
    model.register_buffer('empty',torch.empty(0,dtype=torch.bfloat16))
    assert packed_state_sha256(model)==state_sha256(model)
    wrapper=FrozenUSE(model)
    assert wrapper.verify()==state_sha256(model)
    optimizer=torch.optim.AdamW(model.parameters())
    with pytest.raises(RuntimeError,match='Optimizer'): wrapper.verify(optimizer)
    model.runtime[0,0]+=1
    with pytest.raises(RuntimeError,match='SHA256'): wrapper.verify()
