import torch
from test_use_v4_mit import model,obs
from landslide_seg.models.use_v4_mit import V3,FRONTEND,load_visual
from landslide_seg.models.frozen_use import state_sha256


def test_sources_and_immutable(model):
    state=torch.load(V3,map_location='cpu',weights_only=False)['use']
    for name,value in model.state_dict().items():
        if name.split('.')[0] in FRONTEND: assert torch.equal(value,state[name])
    original=load_visual().segformer.encoder.state_dict()
    for name,value in model.mit.state_dict().items(): assert torch.equal(value,original[name])
    before=state_sha256(model.mit)
    model.zero_grad(); sum(x.mean() for x in model(obs())['context']).backward()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]); opt.step()
    assert state_sha256(model.mit)==before


def test_probe_capacity_and_consumer():
    import pretrain_use_v4 as pt
    from landslide_seg.models.use_v4_reconstruction import Segmenter,NativeProjection
    from landslide_seg.models.use_v3_reconstruction import NativeProjection as Old
    assert sum(p.numel() for p in NativeProjection().parameters())-sum(p.numel() for p in Old().parameters())==1024
    for task in ('fine','context'):
        m=pt.initialize(task=task); optimizer=pt.optimizer(m)
        o=obs(); z=torch.zeros(1,3,3)
        before=state_sha256(m.decode)
        m(o,z,z.bool(),torch.zeros(1,3,2)).square().mean().backward(); optimizer.step()
        m.use.verify(optimizer); assert before!=state_sha256(m.decode)
    torch.manual_seed(1235); m=Segmenter(pt.USEv4()).train()
    optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad])
    output=m(obs()); assert output.shape==(1,1,128,128)
    output.square().mean().backward(); optimizer.step(); m.use.verify(optimizer)
    assert not m.decoder.batch_norm.training


def test_release_and_formal_gate(tmp_path,monkeypatch):
    import json
    import pytest
    import run_use_v4 as run
    from audit_use_independent_data import sha
    jobs=run.matrix(); assert len(jobs)==6 and sum(j['steps'] for j in jobs.values())==29000
    assert {j['point'] for j in jobs.values() if j['mode']=='probe'}=={10000,20000}
    root=tmp_path/'workspace'; root.mkdir(); (root/'source.py').write_text('fixed')
    release=tmp_path/'release'; (release/'snapshot').mkdir(parents=True)
    (release/'snapshot/source.py').write_text('fixed')
    (release/'manifest.json').write_text(json.dumps(dict(files={'source.py':sha(root/'source.py')})))
    monkeypatch.setattr(run,'ROOT',root)
    run.verify_release(release)
    (root/'source.py').write_text('changed')
    with pytest.raises(ValueError,match='Release changed'): run.verify_release(release)
    monkeypatch.setattr(run,'verify_release',lambda path: {})
    marker=tmp_path/'marker.json'
    for status,digest in [('READY_FOR_AUDIT',sha(release/'manifest.json')),('READY_FOR_TRAINING','wrong')]:
        marker.write_text(json.dumps(dict(status=status,release_sha256=digest)))
        with pytest.raises(ValueError,match='independent READY_FOR_TRAINING'):
            run.run_job('pretrain','cpu',release,marker)


def test_card_lock_precedes_probe(monkeypatch):
    import pytest
    import run_use_v4 as run
    events=[]
    class Lock:
        def unlink(self,**kwargs): events.append('unlock')
    monkeypatch.setattr(run.rt,'physical_index',lambda device:(1,None))
    monkeypatch.setattr(run.rt,'claim_card',lambda card:(events.append('lock') or Lock()))
    def probe(device):
        assert events==['lock']; events.append('probe'); return {'device':device}
    monkeypatch.setattr(run.rt,'gpu_available',probe)
    with run.guarded_device('cuda:1'): assert events==['lock','probe']
    assert events[-1]=='unlock'
    events.clear()
    def failed_probe(device):
        assert events==['lock']; raise ValueError('capacity')
    monkeypatch.setattr(run.rt,'gpu_available',failed_probe)
    with pytest.raises(ValueError,match='capacity'):
        with run.guarded_device('cuda:1'): pass
    assert events==['lock','unlock']


def test_formal_initialization_waits_for_guard(tmp_path,monkeypatch):
    import json
    import pytest
    from contextlib import contextmanager
    import run_use_v4 as run
    from audit_use_independent_data import sha
    release=tmp_path/'release'; release.mkdir(); (release/'manifest.json').write_text('{}')
    marker=tmp_path/'marker.json'; marker.write_text(json.dumps(dict(status='READY_FOR_TRAINING',release_sha256=sha(release/'manifest.json'))))
    config=tmp_path/'execution.yaml'; config.write_text('{}')
    monkeypatch.setattr(run,'CONFIG',config); monkeypatch.setattr(run,'BASE',tmp_path/'outputs')
    monkeypatch.setattr(run,'verify_release',lambda p:{})
    monkeypatch.setattr(run,'preflight',lambda:None)
    monkeypatch.setattr(run.pt,'inputs',lambda variant:({}, {}, 'audit', 'members'))
    events=[]
    @contextmanager
    def guard(device):
        events.append('lock+gate')
        try: yield {'device':device}
        finally: events.append('unlock')
    monkeypatch.setattr(run,'guarded_device',guard)
    def initialize(device):
        assert events==['lock+gate']; assert not (tmp_path/'outputs/runs/pretrain/initial.pt').exists()
        events.append('initialize'); raise ValueError('synthetic stop before model allocation')
    monkeypatch.setattr(run.pt,'initialize',initialize)
    with pytest.raises(ValueError,match='synthetic stop'): run.run_job('pretrain','cpu',release,marker)
    assert events==['lock+gate','initialize','unlock']


def test_fast_boundary_and_checkpoint_hashes(tmp_path,monkeypatch):
    import pytest
    import run_use_v4 as run
    from landslide_seg.models.use_v3_reconstruction import FrozenUSE
    m=torch.nn.Module(); m.use=FrozenUSE(torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2)))
    m.head=torch.nn.Linear(2,1); opt=torch.optim.AdamW(m.head.parameters())
    original=run.packed_state_sha256; reference=original(m.use.encoder); calls=[]
    monkeypatch.setattr(run,'packed_state_sha256',lambda module:(calls.append('hash') or original(module)))
    run.check_frozen_boundary(m,opt,'segmentation'); assert calls==[]
    m.use.encoder.train()
    with pytest.raises(RuntimeError,match='boundary'): run.check_frozen_boundary(m,opt,'segmentation')
    m.use.eval(); p=next(m.use.encoder.parameters()); opt.param_groups[0]['params'].append(p)
    with pytest.raises(RuntimeError,match='Optimizer'): run.check_frozen_boundary(m,opt,'segmentation')
    opt.param_groups[0]['params'].pop(); p.grad=torch.ones_like(p)
    with pytest.raises(RuntimeError,match='boundary'): run.check_frozen_boundary(m,opt,'segmentation')
    p.grad=None
    for name in ('initial','best','last'):
        calls.clear(); path=tmp_path/(name+'.pt')
        run.save_checked(m,opt,0,None,{},[],path,reference,'segmentation')
        assert calls==['hash','hash']
        proof=run.rt.read(path.with_suffix('.freeze.json'))
        assert proof['frozen_before_sha256']==proof['frozen_after_sha256']==reference
        assert proof['checkpoint_sha256']==run.sha(path)
    save=run.pt.atomic_save
    def corrupt(payload,path):
        save(payload,path)
        with torch.no_grad(): next(m.use.encoder.buffers()).add_(1)
    monkeypatch.setattr(run.pt,'atomic_save',corrupt)
    with pytest.raises(RuntimeError,match='Frozen state changed'):
        run.save_checked(m,opt,0,None,{},[],tmp_path/'corrupt.pt',reference,'segmentation')
    assert not (tmp_path/'corrupt.freeze.json').exists()


def test_statistics_identity(tmp_path,monkeypatch):
    import run_use_v4 as run
    monkeypatch.setattr(run,'ROOT',tmp_path); monkeypatch.setattr(run,'BASE',tmp_path/'outputs/use_v4')
    paths=['outputs/use/pretraining/resolved_config.json','manifests/stats.json',
           'outputs/use_v4/engineering/calibration.json','outputs/use_v3/engineering/calibration.json']
    for name in paths:
        p=tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}')
    config={'segformer':{'data':{'stats':'manifests/stats.json'}}}
    a=run.statistics_identity(config,{'s2':{'mean':[0.]}})
    assert set(a['files'])==set(paths)
    b=run.statistics_identity(config,{'s2':{'mean':[1.]}})
    assert a['actual_input_stats_sha256']!=b['actual_input_stats_sha256']
    (tmp_path/paths[1]).write_text('{"changed":true}')
    assert run.statistics_identity(config,{})['files'][paths[1]]!=a['files'][paths[1]]


def test_segment_update_numerically_matches_legacy():
    import copy
    import run_use_v4 as run
    from landslide_seg.models.use_v3_reconstruction import FrozenUSE
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.use=FrozenUSE(torch.nn.Conv2d(3,3,1)); self.head=torch.nn.Conv2d(3,1,1)
        def forward(self,x):
            with torch.no_grad(): x=self.use.encoder(x)
            return self.head(x)
    class Data:
        records=[{'sample_id':str(i)} for i in range(5)]
        def __init__(self):
            self.samples=[dict(image=torch.randn(3,16,16),mask=(torch.rand(16,16)>.7).float(),
                valid=torch.ones(16,16,dtype=torch.bool),sample_id=str(i),event_id='synthetic') for i in range(5)]
        def __len__(self): return len(self.samples)
        def __getitem__(self,i): return self.samples[i]
        def set_epoch(self,epoch): pass
    torch.manual_seed(456); a=Tiny(); b=copy.deepcopy(a); dataset=Data()
    config={'segformer':{'loss':{'bce_weight':.5,'dice_weight':.5}}}
    oa=torch.optim.AdamW(a.head.parameters(),lr=6e-4); ob=torch.optim.AdamW(b.head.parameters(),lr=6e-4)
    expected=run.rt.downstream_update(a,oa,dataset,1,config,'cpu')
    actual=run.segment_update(b,ob,dataset,1,config,'cpu')
    assert expected==actual
    assert state_sha256(a)==state_sha256(b)
    b.use.verify(ob)


def test_real_dataset_segment_update_and_evaluation():
    import yaml
    import run_use_v4 as run
    from landslide_seg.models.use_v4_reconstruction import Segmenter
    from landslide_seg.data.optical_observation import OpticalBatch
    config=yaml.safe_load(run.CONFIG.read_text(encoding='utf8')); train,val=run.rt.datasets(config)
    assert (len(train),len(val))==(2223,669)
    torch.manual_seed(1235); m=Segmenter(run.load_encoder()).train()
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=6e-4,weight_decay=.01)
    reference=m.use.verify(opt); before=state_sha256(m.decoder); observations=[]
    def inspect_input(module,args):
        o=args[0]; assert isinstance(o,OpticalBatch) and not torch.is_tensor(o)
        assert o['x'].shape[1]==10 and o['pixel_valid'].shape==o['x'].shape
        assert not ({'mask','valid','event_id'} & set(o))
        assert o['known_mask'][...,0].all() and not o['known_mask'][...,1].any()
        assert all(meta['units']=='core25_zscore' for meta in o['spatial_metadata'])
        observations.append(dict(batch=o['x'].shape[0],shape=list(o['x'].shape),
            valid_quality_pixels=int(o['pixel_valid'].sum()),known_metadata=int(o['known_mask'].sum())))
    handle=m.register_forward_pre_hook(inspect_input)
    run.check_frozen_boundary(m,opt,'segmentation')
    result=run.segment_update(m,opt,train,1,config,'cpu')
    run.check_frozen_boundary(m,opt,'segmentation'); assert m.use.verify(opt)==reference
    assert state_sha256(m.decoder)!=before
    evaluation=run.rt.downstream_evaluate(m,val,config,'cpu',limit=1)
    handle.remove(); assert m.use.verify(opt)==reference and len(observations)==5
    run.rt.write_json(run.BASE/'engineering/repair_b1_real_consumer.json',dict(status='PASS',device='cpu',
        updates=1,formal_updates=0,discarded=True,training_result=result,evaluation=evaluation,
        observations=observations,frozen_before_sha256=reference,frozen_after_sha256=m.use.verify(opt),
        quality_source='USESegDataset accepted imagery-quality sidecars; label valid never enters encoder',
        interface='OpticalObservation -> collate -> OpticalBatch(dict).to(cpu) -> Segmenter'))


def test_segment_evaluation_keeps_best_and_last(tmp_path,monkeypatch):
    import numpy as np
    import run_use_v4 as run
    import evaluate_use_v4 as ev
    import landslide_seg.models.use_v4_reconstruction as reconstruction
    monkeypatch.setattr(run,'BASE',tmp_path); config=tmp_path/'config.yaml'; config.write_text('{}')
    monkeypatch.setattr(run,'CONFIG',config); monkeypatch.setattr(run,'load_encoder',lambda path:None)
    monkeypatch.setattr(run.rt,'datasets',lambda config:(None,None))
    class Model:
        def to(self,device): return self
        def load_state_dict(self,state,strict): pass
    monkeypatch.setattr(reconstruction,'Segmenter',lambda encoder:Model())
    def evaluation(model,data,config,device,prediction_dir):
        prediction_dir.mkdir(parents=True,exist_ok=True)
        np.savez(prediction_dir/'sample.npz',prob=np.array([.2,.8]),mask=np.array([0,1]),valid=np.array([1,1]))
        return {'event_mean_iou':.5}
    monkeypatch.setattr(run.rt,'downstream_evaluate',evaluation)
    out=tmp_path/'runs/segment'; out.mkdir(parents=True)
    for name in ('best','last'):
        path=out/(name+'.pt'); torch.save({'step':5000,'model':{}},path)
        ev.evaluate_job('segment',path,'cpu')
        assert (out/f'evaluation_{name}.json').is_file()
    assert not (out/'evaluation.json').exists()


def test_formal_loop_records_each_step_cost_without_gpu(tmp_path,monkeypatch):
    import json
    from contextlib import contextmanager
    import run_use_v4 as run
    release=tmp_path/'release'; release.mkdir(); (release/'manifest.json').write_text('{}')
    marker=tmp_path/'marker.json'; marker.write_text(json.dumps(dict(status='READY_FOR_TRAINING',release_sha256=run.sha(release/'manifest.json'))))
    config=tmp_path/'execution.yaml'; config.write_text('{}')
    base=tmp_path/'outputs'; (base/'engineering').mkdir(parents=True); (base/'engineering/calibration.json').write_text('{}')
    monkeypatch.setattr(run,'BASE',base); monkeypatch.setattr(run,'CONFIG',config)
    monkeypatch.setattr(run,'verify_release',lambda p:{})
    monkeypatch.setattr(run,'preflight',lambda:None)
    monkeypatch.setattr(run,'matrix',lambda:{'pretrain':dict(mode='pretrain',steps=2,warmup=1)})
    monkeypatch.setattr(run,'statistics_identity',lambda config,stats:{'files':{'bound_stats':'digest'}})
    monkeypatch.setattr(run.pt,'inputs',lambda variant:({'TRAIN':{}},{},'audit','members'))
    active=[]; telemetry=[]
    @contextmanager
    def guard(device):
        active.append(True)
        try: yield {'device':device,'synthetic':True}
        finally: active.clear()
    monkeypatch.setattr(run,'guarded_device',guard)
    class Use(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.mit=torch.nn.Linear(2,2).requires_grad_(False).eval()
        def train(self,mode=True):
            super().train(mode); self.mit.eval(); return self
    def initialize(device):
        assert active
        m=torch.nn.Module(); m.use=Use(); m.head=torch.nn.Linear(2,1); return m
    monkeypatch.setattr(run.pt,'initialize',initialize)
    monkeypatch.setattr(run.pt,'optimizer',lambda model:torch.optim.AdamW(model.head.parameters()))
    def update(*args):
        assert active; telemetry.append('update'); return {'gradient_norm':0.,'synthetic_no_optimizer_update':True}
    monkeypatch.setattr(run.pt,'update',update)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device:telemetry.append('sync'))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda device:telemetry.append('reset'))
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda device:123)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda device:456)
    # CUDA APIs are mocked; models, checkpoints and all arithmetic stay on CPU.
    run.run_job('pretrain','cuda:1',release,marker)
    history=run.rt.read(base/'runs/pretrain/curve.json')
    assert [r['step'] for r in history]==[1,2]
    assert all(r['peak_allocated']==123 and r['peak_reserved']==456 and r['seconds']>=0 for r in history)
    assert telemetry==['sync','reset','update','sync']*2 and not active
    for name in ('initial','last'):
        proof=run.rt.read(base/f'runs/pretrain/{name}.freeze.json')
        assert proof['frozen_before_sha256']==proof['frozen_after_sha256']
    saved=torch.load(base/'runs/pretrain/last.pt',map_location='cpu',weights_only=False)
    assert saved['identity']['config_sha256']==run.sha(config)
    assert saved['identity']['statistics']['files']=={'bound_stats':'digest'}
