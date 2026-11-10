import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from encoder_v3.latent_response import train,telemetry
from encoder_v3.latent_response.epoch_batches import EpochBatches
from encoder_v3.latent_response.checkpoint import digest
from tests.encoder_v3.latent_response.tests.test_pipeline import fixture


def test_epoch_coverage_tail_shuffle_and_resume_cursor():
    rows=[dict(id=str(i),role='TRAIN',sensor='S2' if i<6 else 'EnMAP',bands=[{},{}]) for i in range(7)]
    rows.append(dict(id='val',role='VAL',sensor='S2',bands=[{},{}]))
    plan=EpochBatches(rows,17,3)
    epoch=[[r['id'] for _,r in plan.batch(i)] for i in range(len(plan))]
    assert [len(b) for b in epoch]==[3,3,1]
    assert sorted(sum(epoch,[]))==[str(i) for i in range(7)]
    assert plan.progress(2)==6/7 and plan.progress(3)==1 and plan.observations_seen(4)==10
    resumed=EpochBatches(list(reversed(rows)),17,3)
    assert resumed.batch(2)==plan.batch(2)
    next_epoch=[[r['id'] for _,r in plan.batch(i)] for i in range(3,6)]
    assert next_epoch!=epoch and sorted(sum(next_epoch,[]))==[str(i) for i in range(7)]
    assert [i for i in range(1,4) if plan.validation_due(i,2)]==[2,3]


def test_observation_mean_gradient_and_validation_do_not_equalize_sources():
    torch.set_num_threads(2)
    rows=[dict(id=str(v),role='TRAIN',sensor=s,bands=[{},{}]) for v,s in ((1,'S2'),(3,'S2'),(9,'L8'))]
    class Data:
        def __init__(self):self.rows=rows
        def __getitem__(self,i):return SimpleNamespace(identifier=rows[i]['id'])
        def channels(self,*_):return []
    model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(1)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
    config=dict(epochs=1,objective='observation_equal',phase='E2',seed=17,tile_rows=1,chunk_rows=1,chunk_pixels=1,clip_norm=1e6)
    state=dict(cursor=0,spent_updates=99)
    def measured(model,view,backward=False,loss_weight=1.,**_):
        loss=model.weight.square().sum()*int(view)
        if backward:(loss*loss_weight).backward()
        return dict(loss=float(loss.detach()))
    with patch.object(train,'make_view',side_effect=lambda obs,*a,**kw:obs.identifier) as make,patch.object(train,'view_loss',side_effect=measured):
        result=train.update(model,Data(),optimizer,scheduler,state,config,'cpu',batch=EpochBatches(rows,17,3).batch(0))
        assert abs(result['loss']-13/3)<1e-6
        torch.testing.assert_close(model.weight.grad,torch.tensor([[26/3]]))
        assert state['cursor']==1 and state['spent_updates']==100
        model.weight.data.fill_(1)
        panel=[dict(row,role='VAL') for row in rows]
        make.reset_mock()
        validation=train.validate(model,Data(),optimizer,scheduler,state,panel,config,'cpu')
        assert validation['metric']==13/3 and validation['per_source']=={'S2':2.,'Landsat':9.}
        assert all(call.kwargs==dict(seed=1729,update=0) for call in make.call_args_list)


def test_epoch_authorization_has_no_step_budget():
    config=dict(phase='E3',epochs=3,batch_size=32,validations_per_epoch=4,checkpoint_every_batches=10,
                warmup_epochs=.5,objective='observation_equal')
    approval=dict(source='user_in_current_chat',phase='E3',epochs=3,batch_size=32,config_sha256=digest(config))
    train.require_training(approval,config)
    invalid=dict(config,steps=1000);approval['config_sha256']=digest(invalid)
    try:train.require_training(approval,invalid)
    except ValueError:pass
    else:raise AssertionError('Epoch configuration accepted an ambiguous step budget')


def test_unimplemented_distributed_launch_is_rejected():
    with patch.dict(os.environ,{'WORLD_SIZE':'2'}):
        try:train.main()
        except RuntimeError as exc:assert 'gradient synchronization' in str(exc)
        else:raise AssertionError('Unsynchronized distributed training was allowed')


def test_spectral_masks_cover_variable_channel_counts_and_exclude_invalid_bands():
    from encoder_v3.tasks import hidden_band_indices
    from encoder_v3.latent_response.data import make_view
    for count in (2,4,6,219):
        masks=[hidden_band_indices(count,seed=17,update=epoch,observation_id='variable-channels') for epoch in range(128)]
        assert all(1<=len(mask)<count for mask in masks)
        assert set.union(*masks)==set(range(count))
        assert len({tuple(sorted(mask)) for mask in masks})>1
        assert masks[0]==hidden_band_indices(count,seed=17,update=0,observation_id='variable-channels')
    _,channels,obs=fixture(16)
    obs.bands[-1].valid_mask.zero_();channels[-1].valid.zero_()
    targets=set()
    for epoch in range(16):
        view=make_view(obs,channels,'B1',seed=17,update=epoch)
        names={c.channel_meta.channel_id for c,_,_ in view.targets}
        assert len(names)==1 and obs.bands[-1].record_id not in names
        targets.update(names)
    assert targets=={b.record_id for b in obs.bands[:-1]}


def test_actual_epoch_loop_interrupt_resume_and_tvlp(tmp_path):
    torch.set_num_threads(2)
    class Data:
        def __init__(self,cs,obs,role):
            self.cs=cs;self.obs=obs
            self.rows=[dict(id=f'{role}-{i}',role=role,sensor='S2',bands=[{}, {}, {}]) for i in range(3)]
        def __getitem__(self,i):
            obs=copy.deepcopy(self.obs);obs.metadata.update(self.rows[i]);return obs
        def channels(self,*_):return copy.deepcopy(self.cs)
    model,cs,obs=fixture(16);model.double();initial=copy.deepcopy(model.state_dict())
    identity=dict(code='epoch-synthetic',config='fixture',registry=model.registry.identity_sha256(),catalog='fixture',data='fixture',sampling='fixture')
    preflight=tmp_path/'e1.json';preflight.write_text(json.dumps(dict(status='PASS',identity=identity)))
    config=dict(phase='E2',epochs=2,batch_size=2,objective='observation_equal',seed=17,device='cpu',lr=1e-4,
        weight_decay=.04,clip_norm=3.,warmup_epochs=.5,validations_per_epoch=2,checkpoint_every_batches=1,
        tile_rows=8,chunk_rows=1,chunk_pixels=256,preflight=str(preflight))
    approval=dict(source='user_in_current_chat',phase='E2',epochs=2,batch_size=2,config_sha256=digest(config),model_identity=identity)
    (tmp_path/'config.json').write_text(json.dumps(config));(tmp_path/'approval.json').write_text(json.dumps(approval))
    def build():
        m,channels,observation=fixture(16);m.double();m.load_state_dict(initial)
        for channel in channels:channel.values=channel.values.double()
        return {},{role:Data(channels,observation,role) for role in ('TRAIN','VAL')},{},m,identity
    real_update=train.update
    def invoke(name,resume=False,interrupt=False):
        run=tmp_path/name
        argv=['train','--config',str(tmp_path/'config.json'),'--authorization',str(tmp_path/'approval.json'),'--run-dir',str(run)]
        if resume:argv+=['--resume',str(run/'last.pt')]
        def updating(*args,**kwargs):
            result=real_update(*args,**kwargs)
            if interrupt:(run/'STOP_REQUESTED').touch()
            return result
        with patch.object(train,'build_inputs',side_effect=build),patch.object(train,'update',side_effect=updating),patch('sys.argv',argv),patch.object(telemetry,'training_runs_root',return_value=tmp_path/'viewer'):
            train.main()
    invoke('golden');invoke('restored',interrupt=True)
    stopped=json.loads((tmp_path/'restored/status.json').read_text())
    assert stopped['status']=='STOPPED' and stopped['cursor']==1 and stopped['observations_seen']==2
    (tmp_path/'restored/STOP_REQUESTED').unlink();invoke('restored',resume=True)
    expected=torch.load(tmp_path/'golden/last.pt',weights_only=False)
    actual=torch.load(tmp_path/'restored/last.pt',weights_only=False)
    for name,tensor in expected['model'].items():torch.testing.assert_close(tensor,actual['model'][name],rtol=1e-10,atol=1e-12)
    state=json.loads((tmp_path/'restored/status.json').read_text())
    assert state['status']=='COMPLETE' and state['epoch']==2 and state['observations_seen']==6
    assert state['validation']['panels']==3 and state['validation']['reduction']=='observation_equal'
    updates=[json.loads(line) for line in (tmp_path/'restored/updates.jsonl').read_text().splitlines()]
    assert [r['batch_size'] for r in updates]==[2,1,2,1]
    golden_updates=list(map(json.loads,(tmp_path/'golden/updates.jsonl').read_text().splitlines()))
    assert [[m['id'] for m in r['members']] for r in updates]==[[m['id'] for m in r['members']] for r in golden_updates]
    directory=Path(state['viewer']['directory']);meta=json.loads((directory/'run.json').read_text())
    assert meta['config']['batch_size']==2 and 'budget_updates' not in meta['config']
    assert meta['metrics']['val_loss']['label']=='完整验证集平均损失（观测等权）'
    assert [e['kind'] for e in map(json.loads,(directory/'events.jsonl').read_text().splitlines())]==['started','failed','resumed','completed']
    metrics=list(map(json.loads,(directory/'metrics.jsonl').read_text().splitlines()))
    epochs=[r['epoch'] for r in metrics if 'train_loss' in r['values']]
    assert epochs==[2/3,1,5/3,2]
