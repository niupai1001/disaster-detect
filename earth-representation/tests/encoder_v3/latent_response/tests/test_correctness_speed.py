import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from encoder_v3.latent_response.run_control import lr_multiplier,reconcile_logs,append_event,require_execution_identity,BudgetExhausted
from encoder_v3.latent_response.train import update
from encoder_v3.latent_response.execution import StopRequested
from encoder_v3.latent_response.checkpoint import parameter_sha,load_checkpoint
from encoder_v3.latent_response.geometry_cache import GeometryCache
from encoder_v3.latent_response.geometry import FullGeometry
from encoder_v3.latent_response.fusion import channel_front,grouped_channel_front,LocalEmbed
from encoder_v3.latent_response import ChannelObservation
from tests.encoder_v3.latent_response.tests.test_pipeline import fixture
from tests.encoder_v3.latent_response.tests.test_training_loop import SyntheticDataset


class CorrectnessSpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_warmup_reaches_peak_then_cosine(self):
        values=[lr_multiplier(i,20,10) for i in range(21)]
        self.assertEqual(values[0],.1);self.assertEqual(values[9],1.)
        self.assertEqual(max(values),1.);self.assertEqual(values[-1],0.)
        self.assertTrue(all(values[i]>=values[i+1] for i in range(10,20)))
        for bad in ((0,20,20),(0,0,0),(0,20,-1)):
            with self.assertRaises(ValueError):lr_multiplier(*bad)

    def test_authorization_identity_and_enmap_requirement(self):
        identity=dict(code='a',data='b');approval=dict(model_identity=identity)
        with self.assertRaises(PermissionError):require_execution_identity(approval,dict(enmap_required=True),identity,[dict(role='TRAIN',sensor='S2')])
        with self.assertRaises(PermissionError):require_execution_identity(approval,{},dict(code='changed'),[])
        require_execution_identity(approval,dict(enmap_required=True),identity,[dict(role='TRAIN',sensor='EnMAP')])

    def test_grouped_front_matches_reference_and_gradients(self):
        model,channels,_=fixture(17);grid=channels[0].grid;geometry=FullGeometry.from_channels(channels)
        cache=GeometryCache();plans=tuple(cache.plan(grid,geometry,'cpu',i,min(i+7,17)) for i in range(0,17,7))
        values=torch.randn(3,17,17,dtype=torch.float64,requires_grad=True)
        keep=torch.rand(3,17,17)>.3
        kernels=torch.randn(3,32,1,3,3,dtype=torch.float64,requires_grad=True)
        bias=torch.randn(32,dtype=torch.float64,requires_grad=True);embed=LocalEmbed().double()
        h,s=grouped_channel_front(values,keep,kernels,bias,plans,geometry,embed,7)
        references=[channel_front(values[i],keep[i],kernels[i:i+1],bias,channels[i],geometry,embed,7) for i in range(3)]
        expected=torch.stack([a for a,b in references]);support=torch.stack([b for a,b in references])
        torch.testing.assert_close(h,expected,atol=1e-10,rtol=1e-10);torch.testing.assert_close(s,support)
        parameters=(values,kernels,bias,*embed.parameters())
        actual=torch.autograd.grad(h.square().sum(),parameters,retain_graph=True)
        wanted=torch.autograd.grad(expected.square().sum(),parameters)
        for a,b in zip(actual,wanted):torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-9)
        cache.plan(grid,geometry,'cpu',0,7);self.assertEqual(cache.hits,1)
        tiny=GeometryCache(limit_bytes=1);tiny.plan(grid,geometry,'cpu',0,7);self.assertEqual(tiny.bytes,0)

    def test_stop_prevents_optimizer_update(self):
        model,channels,observation=fixture(16);dataset=SyntheticDataset(channels,observation)
        opt=torch.optim.AdamW(model.parameters());scheduler=torch.optim.lr_scheduler.StepLR(opt,1)
        state=dict(cursor=0);before=parameter_sha(model);calls=[]
        def stop():calls.append(1);return len(calls)>1
        with self.assertRaises(StopRequested):update(model,dataset,opt,scheduler,state,dict(phase='E2',seed=17,tile_rows=8,chunk_rows=1,chunk_pixels=256,clip_norm=3.),'cpu',stop)
        self.assertEqual(state['cursor'],0);self.assertEqual(parameter_sha(model),before)

    def test_partial_journal_is_preserved_and_conservatively_charged(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);append_event(run/'optimizer_attempts.jsonl',dict(event='intent',sequence=1,cursor=1))
            with (run/'optimizer_attempts.jsonl').open('a') as f:f.write('{"event":')
            state=dict(cursor=0,spent_updates=0);r=reconcile_logs(run,state)
            self.assertEqual(r['spent_updates'],1)
            archive=json.loads((run/'recovery_attempts.jsonl').read_text())
            self.assertEqual(archive['partial_event_tail'],'{"event":')

    def test_process_crash_recovery_log_and_real_attempt_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            script="""
import os,json,torch,sys
from pathlib import Path
from tests.encoder_v3.latent_response.tests.test_pipeline import fixture
from tests.encoder_v3.latent_response.tests.test_training_loop import SyntheticDataset
from encoder_v3.latent_response.train import update
from encoder_v3.latent_response.checkpoint import save_checkpoint
from encoder_v3.latent_response.run_control import append_event
torch.set_num_threads(2)
model,cs,obs=fixture(16);model.double()
for c in cs:c.values=c.values.double()
data=SyntheticDataset(cs,obs);opt=torch.optim.AdamW(model.parameters(),lr=1e-4);sched=torch.optim.lr_scheduler.StepLR(opt,100)
state=dict(cursor=0,spent_updates=0,validation={});config=dict(phase='E2',seed=17,tile_rows=8,chunk_rows=1,chunk_pixels=256,clip_norm=3.,steps=3,run_dir=sys.argv[1]);run=Path(sys.argv[1])
identity=dict(code='test',config='test',registry=model.registry.identity_sha256(),catalog='test',data='test',sampling='test')
r=update(model,data,opt,sched,state,config,'cpu');append_event(run/'updates.jsonl',r);save_checkpoint(run/'last.pt',model,opt,state,identity,sched)
r=update(model,data,opt,sched,state,config,'cpu');append_event(run/'updates.jsonl',r);torch.save(model.state_dict(),run/'golden.pt')
os._exit(17)
"""
            root=Path(__file__).resolve().parents[4]
            result=subprocess.run([sys.executable,'-c',script,str(run)],cwd=root,timeout=90,capture_output=True,text=True)
            self.assertEqual(result.returncode,17,result.stderr)
            model,cs,obs=fixture(16);model.double()
            for c in cs:c.values=c.values.double()
            data=SyntheticDataset(cs,obs);opt=torch.optim.AdamW(model.parameters(),lr=1e-4);sched=torch.optim.lr_scheduler.StepLR(opt,100)
            identity=dict(code='test',config='test',registry=model.registry.identity_sha256(),catalog='test',data='test',sampling='test');state={}
            load_checkpoint(run/'last.pt',model,opt,state,identity,sched);r=reconcile_logs(run,state)
            self.assertEqual((r['committed'],r['spent_updates'],r['uncommitted']),(1,2,1))
            config=dict(phase='E2',seed=17,tile_rows=8,chunk_rows=1,chunk_pixels=256,clip_norm=3.,steps=3,run_dir=str(run))
            record=update(model,data,opt,sched,state,config,'cpu');append_event(run/'updates.jsonl',record)
            self.assertEqual((state['cursor'],state['spent_updates']),(2,3))
            self.assertEqual([json.loads(l)['cursor'] for l in (run/'updates.jsonl').read_text().splitlines()],[1,2])
            golden=torch.load(run/'golden.pt',weights_only=True)
            for n,v in model.state_dict().items():torch.testing.assert_close(v,golden[n],atol=1e-10,rtol=1e-8)
            with self.assertRaises(BudgetExhausted):update(model,data,opt,sched,state,config,'cpu')


    def test_cli_stop_file_saves_recoverable_checkpoint(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);root=Path(__file__).resolve().parents[4]
            script="""
import sys,json,torch
from pathlib import Path
from unittest.mock import patch
from tests.encoder_v3.latent_response.tests.test_pipeline import fixture
from tests.encoder_v3.latent_response.tests.test_training_loop import SyntheticDataset
from encoder_v3.latent_response import train
from encoder_v3.latent_response import telemetry
from encoder_v3.latent_response.checkpoint import digest
model,cs,obs=fixture(16);dataset=SyntheticDataset(cs,obs);base=Path(sys.argv[1])
identity=dict(code='synthetic',config='synthetic',registry=model.registry.identity_sha256(),catalog='synthetic',data='synthetic',sampling='synthetic')
config=dict(phase='E2',steps=3,seed=17,device='cpu',lr=1e-4,weight_decay=.04,clip_norm=3.,warmup_steps=1,validate_every=3,checkpoint_every=3,tile_rows=8,chunk_rows=1,chunk_pixels=256,preflight=str(base/'e1.json'))
(base/'e1.json').write_text(json.dumps(dict(status='PASS',identity=identity)))
(base/'config.json').write_text(json.dumps(config));(base/'approval.json').write_text(json.dumps(dict(source='user_in_current_chat',phase='E2',steps=3,config_sha256=digest(config),model_identity=identity)))
sys.argv=['train','--config',str(base/'config.json'),'--authorization',str(base/'approval.json'),'--run-dir',str(base/'run')]
if (base/'resume').exists():sys.argv+=['--resume',str(base/'run/last.pt')]
with patch.object(train,'build_inputs',return_value=({},dict(TRAIN=dataset,VAL=dataset),{},model,identity)),patch.object(train,'cases',return_value=[dict(role='VAL',sensor='S2',id=obs.identifier)]),patch.object(telemetry,'training_runs_root',return_value=base/'training-runs'):train.main()
"""
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
            child=subprocess.Popen([sys.executable,'-c',script,str(directory)],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            deadline=time.monotonic()+90;status=directory/'run/status.json'
            while not status.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.01)
            self.assertTrue(status.exists())
            (directory/'run/STOP_REQUESTED').write_text('user stop')
            stdout,stderr=child.communicate(timeout=90)
            self.assertEqual(child.returncode,0,stderr)
            result=json.loads(status.read_text());self.assertEqual(result['status'],'STOPPED')
            checkpoint=torch.load(directory/'run/last.pt',weights_only=False)
            self.assertEqual(checkpoint['runner']['cursor'],result['cursor'])
            self.assertEqual(checkpoint['runner']['spent_updates'],result['spent_updates'])
            self.assertTrue((directory/'run/last.pt').is_file())
            viewer=checkpoint['runner']['viewer'];viewer_dir=Path(viewer['directory'])
            self.assertEqual(json.loads((viewer_dir/'run.json').read_text())['run_id'],viewer['run_id'])
            self.assertEqual(json.loads((viewer_dir/'events.jsonl').read_text().splitlines()[-1])['kind'],'failed')
            (directory/'run/STOP_REQUESTED').unlink();(directory/'resume').touch()
            child=subprocess.Popen([sys.executable,'-c',script,str(directory)],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            deadline=time.monotonic()+90;running=False
            while child.poll() is None and time.monotonic()<deadline:
                try:running=json.loads(status.read_text())['status']=='RUNNING'
                except (json.JSONDecodeError,OSError):pass
                if running:break
                time.sleep(.01)
            (directory/'run/STOP_REQUESTED').write_text('user stop after resume')
            stdout,stderr=child.communicate(timeout=90)
            self.assertEqual(child.returncode,0,stderr);self.assertTrue(running)
            self.assertEqual(json.loads(status.read_text())['status'],'STOPPED')
            resumed=torch.load(directory/'run/last.pt',weights_only=False)
            self.assertEqual(resumed['runner']['viewer'],viewer)
            events=[json.loads(line) for line in (viewer_dir/'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['kind']=='resumed' for e in events),1)
            self.assertEqual(len({e['session_id'] for e in events}),2)


    def test_runtime_precision_matches_equivalence_policy(self):
        from encoder_v3.latent_response.preflight import configure_fp32
        old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
            configure_fp32()
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32);self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32=old[0];torch.backends.cudnn.allow_tf32=old[1]
