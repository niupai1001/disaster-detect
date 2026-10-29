"""Numerical contracts exercised with the real local v3/ADE initialization."""
import copy
import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_use_mit_dual_routes as run
from landslide_seg.models.use_mit_dual_routes import (visible_rgb,alignment_loss,
    student_alignment_support,support_fraction,PatchTeacher)
from landslide_seg.models.use_v4_mit import FRONTEND,staged


@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(4)
    return {r:run.load_encoder(r) for r in ('A','B','C0')}


def observation(channels=10,size=128):
    generator=torch.Generator().manual_seed(37)
    ids=['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
    ids=(ids+['extra'+str(i) for i in range(6)])[:channels]
    x=torch.randn(1,channels,size,size,generator=generator)*.1
    return dict(x=x,pixel_valid=torch.ones_like(x,dtype=torch.bool),
        band_valid=torch.ones(1,channels,dtype=torch.bool),band_ids=[tuple(ids)],
        band_metadata=torch.tensor([[[.5,.02,10.]]*channels]),known_mask=torch.ones(1,channels,3,dtype=torch.bool),
        spatial_metadata=[dict(normalization_mean=[.1]*channels,normalization_std=[.2]*channels)])


def test_shared_initialization(models):
    a,b,c=[models[r] for r in ('A','B','C0')]
    for key,value in a.state_dict().items():
        if key.split('.')[0] in FRONTEND or key.startswith('mit.'):
            assert torch.equal(value,b.state_dict()[key])
            assert torch.equal(value,c.state_dict()[key])
    assert all(torch.equal(value,b.adapter.state_dict()[key]) for key,value in a.adapter.state_dict().items())
    assert all(torch.count_nonzero(p)==0 for p in a.residual_zero.parameters())


def test_native_equivalence_and_zero_gradient_transition(models):
    a=models['A']; state=copy.deepcopy(a.state_dict()); obs=observation()
    try:
        a.train(); rgb,valid=visible_rgb(obs)
        native=a.mit(rgb,output_hidden_states=True).hidden_states
        result=a(obs,diagnostics=True)
        baseline,support=staged(a.mit,rgb,valid.any(1))
        for left,right,official in zip(result['context'],baseline,native):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
            torch.testing.assert_close(left,official,rtol=2e-5,atol=2e-6)
        assert torch.count_nonzero(result['residual'])==0
        loss=sum(x.square().mean() for x in result['context']); loss.backward()
        assert a.residual_zero.weight.grad.norm()>0
        assert torch.count_nonzero(a.adapter.weight.grad)==0
        with torch.no_grad(): a.residual_zero.weight.add_(a.residual_zero.weight.grad,alpha=-.1)
        a.zero_grad(set_to_none=True)
        sum(x.square().mean() for x in a(obs)['context']).backward()
        assert a.adapter.weight.grad.norm()>0 and a.readout.in_proj_weight.grad.norm()>0
        assert all(p.grad is None and not p.requires_grad for p in a.mit.parameters())
        assert not any(m.training for m in a.mit.modules())
    finally:
        a.load_state_dict(state); a.zero_grad(set_to_none=True)


def test_teacher_stop_gradient_and_no_forward_dependency(models):
    b=models['B']; obs=observation(); teacher=PatchTeacher(b).eval()
    target,support=teacher(obs)
    assert not target.requires_grad and not teacher.training
    result=b(obs); before=result['context'][-1].detach().clone()
    loss,coverage=alignment_loss(result['patch_tokens'],target,support,obs)
    changed=alignment_loss(result['patch_tokens'],target.roll(1,-1),support,obs)[0]
    assert loss!=changed and all(n>0 for n in coverage['tokens'])
    loss.backward(); assert b.adapter.weight.grad.norm()>0
    torch.testing.assert_close(before,b(obs)['context'][-1],rtol=0,atol=0)
    assert all(p.grad is None for p in teacher.parameters())
    assert not any('teacher' in k for k in b.state_dict())
    b.zero_grad(set_to_none=True)
    b(obs)['context'][-1].square().mean().backward()
    assert b.adapter.weight.grad.norm()>0 and b.readout.in_proj_weight.grad.norm()>0
    assert all(p.grad is None for p in b.mit.parameters())
    b.zero_grad(set_to_none=True)


@pytest.mark.parametrize('channels',[1,3,6,10,16])
@pytest.mark.parametrize('size',[127,128,129,256])
def test_b_legal_shapes(models,channels,size):
    with torch.no_grad():
        result=models['B'](observation(channels,size))
    assert result['fine'].shape==(1,32,size,size)
    assert [x.shape[-1] for x in result['context']]==[(size+s-1)//s for s in (4,8,16,32)]


@pytest.mark.parametrize('route',['A','C0'])
@pytest.mark.parametrize('size',[127,128,129,256])
def test_visual_legal_and_reject_missing_rgb(models,route,size):
    with torch.no_grad(): assert models[route](observation(3,size))['fine'].shape[-1]==size
    with pytest.raises(ValueError,match='RGB'): models[route](observation(1,size))


def test_masks_hidden_truth_and_metadata(models):
    obs=observation(); obs['pixel_valid'][:,:,20:40,20:40]=False
    obs['pixel_valid'][:,0]=False
    corrupt=copy.deepcopy(obs); corrupt['x'][~corrupt['pixel_valid']]=float('nan')
    for route in ('A','B','C0'):
        with torch.no_grad():
            first=models[route](obs); second=models[route](corrupt)
        for key in ('fine',): torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
        for left,right in zip(first['context'],second['context']): torch.testing.assert_close(left,right,rtol=0,atol=0)
    rgb,valid=visible_rgb(corrupt)
    assert not valid[:,2].any()
    torch.testing.assert_close(rgb[:,2],torch.full_like(rgb[:,2],-.406/.225))
    corrupt['pixel_valid'].zero_()
    with pytest.raises(ValueError,match='Empty'): models['B'](corrupt)
    corrupt=copy.deepcopy(obs); corrupt['label']=torch.ones(1)
    with pytest.raises(ValueError,match='labels'): models['B'](corrupt)


def test_permutation_padding_deletion_unknown(models):
    obs=observation(); order=torch.tensor([2,1,0,6,4,5,3,7,9,8])
    perm=copy.deepcopy(obs)
    for key in ('x','pixel_valid','band_valid','band_metadata','known_mask'): perm[key]=perm[key][:,order]
    perm['band_ids']=[tuple(obs['band_ids'][0][i] for i in order)]
    for key in ('normalization_mean','normalization_std'): perm['spatial_metadata'][0][key]=[obs['spatial_metadata'][0][key][i] for i in order]
    with torch.no_grad():
        for route in ('A','B','C0'):
            first=models[route](obs); second=models[route](perm)
            torch.testing.assert_close(first['fine'],second['fine'],atol=2e-6,rtol=2e-5)
            for x,y in zip(first['context'],second['context']): torch.testing.assert_close(x,y,atol=3e-5,rtol=3e-5)
        obs['known_mask'].zero_();obs['band_metadata'].zero_()
        assert torch.isfinite(models['B'](obs)['fine']).all()
        padded=copy.deepcopy(obs)
        for key in ('x','pixel_valid','band_valid','band_metadata','known_mask'):
            value=padded[key]; shape=list(value.shape);shape[1]=2
            padded[key]=torch.cat((value,torch.zeros(shape,dtype=value.dtype)),1)
        padded['band_ids']=[obs['band_ids'][0]+('pad1','pad2')]
        for key in ('normalization_mean','normalization_std'): padded['spatial_metadata'][0][key]+=[.1,.2]
        torch.testing.assert_close(models['B'](obs)['fine'],models['B'](padded)['fine'],rtol=2e-5,atol=2e-6)
        deleted=copy.deepcopy(obs)
        for key in ('x','pixel_valid','band_valid','band_metadata','known_mask'): deleted[key]=deleted[key][:,:6]
        deleted['band_ids']=[obs['band_ids'][0][:6]]
        assert models['B'](deleted)['fine'].shape==models['B'](obs)['fine'].shape


def test_alignment_normalization_and_edges():
    obs=observation(1,127)
    support=student_alignment_support(obs)
    assert support.all() and support.shape==(1,32,32)
    tokens=torch.randn(2,1024,32,requires_grad=True);target=torch.randn_like(tokens)
    two=observation(1,127)
    for key in ('x','pixel_valid','band_valid','band_metadata','known_mask'): two[key]=two[key].expand(2,*two[key].shape[1:]).clone()
    mask=torch.ones(2,32,32,dtype=torch.bool);mask[1]=False
    full=alignment_loss(tokens,target,mask,two)[0]
    one=alignment_loss(tokens[:1],target[:1],mask[:1],obs)[0]
    torch.testing.assert_close(full,one/2)
    zero=alignment_loss(tokens,target,torch.zeros_like(mask),two)[0]
    assert zero==0 and zero.requires_grad


def test_task_head_initialization(models):
    states=[]
    for route in ('A','B','C0'):
        encoder=copy.deepcopy(models[route]);torch.manual_seed(1235)
        model=run.Segmenter(encoder)
        states.append({k:v for k,v in model.state_dict().items() if not k.startswith('use.')})
        assert all(not m.training for m in model.decoder.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
    for state in states[1:]: assert all(torch.equal(v,state[k]) for k,v in states[0].items())


def test_export_b_no_rgb(models,tmp_path):
    saved=tmp_path/'encoder.pt';torch.save(models['B'].state_dict(),saved)
    loaded=run.load_encoder('B');loaded.load_state_dict(torch.load(saved,weights_only=True),strict=True)
    obs=observation(1,128);del obs['spatial_metadata'];del obs['band_ids']
    with torch.no_grad(): assert torch.isfinite(loaded(obs)['context'][-1]).all()


def test_budget():
    assert sum(j['steps'] for j in run.matrix().values())==59000
    assert len(run.matrix())==9
    assert not any('10000' in k or '5000' in k for k in run.matrix())


@pytest.mark.parametrize('job',['probe_A_fine','segment_C0'])
def test_evaluation_active_lock_precedes_cuda_factory(monkeypatch,tmp_path,job):
    monkeypatch.setattr(run.rt,'BASE',tmp_path)
    lock=tmp_path/'runtime/gpu0.lock';lock.parent.mkdir();lock.write_text('98765')
    monkeypatch.setattr(run.rt,'card_process_alive',lambda pid:True)
    def forbidden(*args,**kwargs):raise AssertionError('CUDA gate/model factory called before refusing active lock')
    monkeypatch.setattr(run.rt,'gpu_available',forbidden)
    monkeypatch.setattr(run,'initialize',forbidden);monkeypatch.setattr(run,'segmenter',forbidden)
    monkeypatch.setattr(run.torch,'load',lambda *a,**k:dict(step=1000,model={}))
    with pytest.raises(ValueError,match='already hosts'):run.evaluate(job,tmp_path/'not_read.pt','cuda:0')
    assert lock.read_text()=='98765'


@pytest.mark.parametrize('job,factory',[('probe_A_fine','initialize'),('segment_C0','segmenter')])
def test_evaluation_gate_precedes_factory(monkeypatch,tmp_path,job,factory):
    from contextlib import contextmanager
    events=[]
    @contextmanager
    def guard(device):
        events.append('guard');yield {}
    def construct(*args,**kwargs):
        events.append('factory');raise RuntimeError('STOP_BEFORE_GPU')
    monkeypatch.setattr(run.v4,'guarded_device',guard)
    monkeypatch.setattr(run,factory,construct)
    monkeypatch.setattr(run.torch,'load',lambda *a,**k:dict(step=1000,model={}))
    with pytest.raises(RuntimeError,match='STOP_BEFORE_GPU'):run.evaluate(job,tmp_path/'not_read.pt','cuda:0')
    assert events==['guard','factory']


def test_final_benchmark_requires_completed_heads(monkeypatch,tmp_path):
    import verify_use_mit_dual_routes as verify
    monkeypatch.setattr(run,'BASE',tmp_path)
    for route in ('A','B'):
        path=tmp_path/f'runs/pretrain_{route}/u20000.pt';path.parent.mkdir(parents=True);path.touch()
    assert not verify.final_benchmark_available()
    for route in ('C0','A','B'):
        path=tmp_path/f'runs/segment_{route}';path.mkdir(parents=True)
        (path/'best.pt').touch();run.rt.write_json(path/'summary.json',dict(complete=True,updates=5000))
    assert verify.final_benchmark_available()
    run.rt.write_json(tmp_path/'runs/segment_B/summary.json',dict(complete=False,updates=4500))
    with pytest.raises(ValueError,match='completed'):verify.final_benchmark_available()


def test_final_benchmark_loads_exact_trained_head(models,tmp_path):
    import verify_use_mit_dual_routes as verify
    from landslide_seg.models.use_v3_reconstruction import packed_state_sha256
    torch.manual_seed(1235);trained=run.Segmenter(copy.deepcopy(models['C0'])).eval()
    with torch.no_grad():trained.decoder.classifier.bias.add_(.625)
    payload=dict(model=trained.state_dict(),model_sha256=packed_state_sha256(trained),
        use_sha256=packed_state_sha256(trained.use.encoder),step=1500,
        identity=dict(job=run.matrix()['segment_C0'],engineering=False,release_sha256='test-release'))
    checkpoint=tmp_path/'best.pt';torch.save(payload,checkpoint)
    loaded,digest=verify.benchmark_segmenter(models['C0'],'C0','cpu',checkpoint,'test-release')
    assert digest==run.sha(checkpoint)
    assert packed_state_sha256(loaded)==payload['model_sha256']
    assert not any(m.training for m in loaded.modules())
    with pytest.raises(ValueError,match='another release'):
        verify.benchmark_segmenter(models['C0'],'C0','cpu',checkpoint,'wrong-release')
    with pytest.raises(ValueError,match='identity'):
        verify.benchmark_segmenter(models['C0'],'A','cpu',checkpoint,'test-release')
    payload['model_sha256']='wrong';torch.save(payload,checkpoint)
    with pytest.raises(ValueError,match='full model hash'):
        verify.benchmark_segmenter(models['C0'],'C0','cpu',checkpoint,'test-release')
