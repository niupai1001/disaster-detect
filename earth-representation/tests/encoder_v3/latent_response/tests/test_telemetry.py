import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from encoder_v3.latent_response import telemetry
from encoder_v3.latent_response.checkpoint import save_checkpoint,load_checkpoint
from encoder_v3.latent_response.data import make_view
from encoder_v3.latent_response.execution import view_loss
from tests.encoder_v3.latent_response.tests.test_pipeline import fixture


class TelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_temporary_windows_reader_lock_retries_then_preserves_failure(self):
        from encoder_v3.latent_response.checkpoint import atomic_replace
        from encoder_v3.latent_response import checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'status.tmp';target=Path(tmp)/'status.json';source.write_text('new')
            replace=os.replace;attempts=[]
            def briefly_locked(a,b):
                attempts.append(1)
                if len(attempts)==1:raise PermissionError('reader holds file')
                return replace(a,b)
            with patch.object(checkpoint.os,'replace',side_effect=briefly_locked),patch.object(checkpoint.time,'sleep') as pause:
                atomic_replace(source,target);self.assertEqual(target.read_text(),'new');pause.assert_called_once_with(.02)
            with patch.object(checkpoint.os,'replace',side_effect=PermissionError('persistent')),patch.object(checkpoint.time,'sleep'):
                with self.assertRaises(PermissionError):atomic_replace(source,target)

    def test_metadata_epoch_and_variant_validation(self):
        config=dict(phase='E2',seed=17)
        first=telemetry.metadata(config);second=telemetry.metadata(dict(config,seed=999))
        self.assertEqual(first,second)
        self.assertNotIn('seed',json.dumps(first).lower())
        self.assertEqual(telemetry.METRICS['train_loss']['role'],'loss')
        self.assertEqual(telemetry.METRICS['val_loss']['direction'],'min')
        for kind in ('baseline','ablation','modification'):
            custom=dict(id='custom-'+kind,name='自定义方案',kind=kind,description='移除全局读出' if kind=='ablation' else '参考与改动说明')
            self.assertEqual(telemetry.metadata(dict(config,viewer=dict(variant=custom)))[1],custom)
        with self.assertRaises(ValueError):telemetry.metadata(dict(config,viewer=dict(variant=dict(id='bad',name='无说明',kind='ablation'))))
        rows=[dict(id=str(i),role='TRAIN',sensor=s,bands=[{},{}]) for i,s in enumerate(('S2','S2','S2','L8','L9','EnMAP'))]
        rows.append(dict(id='val',role='VAL',sensor='S2',bands=[{},{}]))
        self.assertEqual(telemetry.epoch_length(rows,config),3)
        self.assertEqual(telemetry.epoch_length(rows,dict(config,e2_members=['0','3','5'])),1)

    def test_non_primary_does_not_open_writer_or_directory(self):
        with patch.dict(os.environ,{'RANK':'1'}),patch.object(torch.distributed,'is_initialized',return_value=False),patch.object(telemetry,'training_runs_root') as root:
            with telemetry.training_writer({}, {}, {}, Path('unused'), [],resuming=False) as writer:
                self.assertIsNone(writer)
            root.assert_not_called()

    def test_new_runs_have_new_ids_and_checkpoint_resume_has_new_session(self):
        model,channels,observation=fixture(16)
        optimizer=torch.optim.AdamW(model.parameters());scheduler=torch.optim.lr_scheduler.StepLR(optimizer,100)
        config=dict(phase='E2',steps=2,lr=.0003,weight_decay=.04)
        rows=[dict(id=observation.identifier,role='TRAIN',sensor='S2',bands=[{},{}])]
        identity=dict(code='synthetic-unit-test',config='fixture',registry=model.registry.identity_sha256(),catalog='fixture',data='fixture',sampling='fixture')
        with tempfile.TemporaryDirectory() as tmp,patch.object(telemetry,'training_runs_root',return_value=Path(tmp)/'training-runs'):
            run=Path(tmp)/'checkpoints';run.mkdir();state=dict(cursor=0,spent_updates=0)
            with telemetry.training_writer(config,identity,state,run,rows,resuming=False) as writer:
                original_id=writer.meta['run_id'];session=writer.session
                # Log a measured model loss, not an invented smoke-test metric.
                value=view_loss(model,make_view(observation,channels,'A'),chunk_pixels=256)['loss']
                writer.log(step=0,epoch=0.,values=dict(train_loss=value))
                save_checkpoint(run/'last.pt',model,optimizer,state,identity,scheduler)
            with self.assertRaises(ValueError):telemetry.training_writer(config,identity,state,run,rows,resuming=False)
            restored={};load_checkpoint(run/'last.pt',model,optimizer,restored,identity,scheduler)
            with telemetry.training_writer(config,identity,restored,run,rows,resuming=True) as writer:
                self.assertEqual(writer.meta['run_id'],original_id);self.assertNotEqual(writer.session,session)
            directory=Path(restored['viewer']['directory'])
            events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
            self.assertEqual([event['kind'] for event in events],['started','completed','resumed','completed'])
            with telemetry.training_writer(config,identity,{},Path(tmp)/'second',rows,resuming=False) as writer:
                self.assertNotEqual(writer.meta['run_id'],original_id)
                self.assertEqual(writer.meta['variant']['id'],telemetry.VARIANT['id'])
            bad=copy.deepcopy(restored);bad['viewer']['run_id']='different'
            with self.assertRaises(ValueError):telemetry.training_writer(config,identity,bad,run,rows,resuming=True)
            with self.assertRaisesRegex(RuntimeError,'test failure'):
                with telemetry.training_writer(config,identity,{},Path(tmp)/'failed',rows,resuming=False) as writer:
                    failed=writer.path
                    raise RuntimeError('test failure')
            self.assertEqual(json.loads((failed/'events.jsonl').read_text().splitlines()[-1])['kind'],'failed')
            records=events+[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(sorted(r['seq'] for r in records),list(range(1,len(records)+1)))
