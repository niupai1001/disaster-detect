import copy,hashlib,json,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import test_scheduled_data as fixtures
from .validation_panels import FrozenValidationPanels
from .participation_encoder import ParticipationEncoder,merge_recipe
from .participation_layout import AreaEquivalentHillRate
from .model import ReflectanceDecoder
from .route_control import frozen_identity


class ValidationPanelChecks(unittest.TestCase):
    def setup_case(self,folder,edit=None):
        ds=fixtures.DatasetFixture();ds.role='VAL'
        for row,obs in zip(ds.rows,ds.observations):row['role']='VAL';obs.metadata['role']='VAL'
        actual=json.loads((Path(__file__).parent/'config.json').read_text());config=dict(required_before_training={k:'SYNTHETIC_ONLY' for k in actual['required_before_training']},formal_step={'chunk_pixels':7});config['required_before_training']['formal_training_authorization']=True
        curve=AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556)
        pair=dict(source_id='fixture0',target_id='fixture1',checks={k:True for k in ['space','registration','time_difference','surface_change','qa','srf','role_isolation']})
        common=dict(source_id='fixture0',origin_xy=[0.,0.],P_m=60.)
        cases=[dict(common,panel_id='A_fixed',kind='A',rectangle=[0.,0.,10.,20.]),dict(common,panel_id='blue_hidden',kind='B1',hidden_band_ids=['b0']),dict(common,panel_id='other_hidden',kind='B1',hidden_band_ids=['b1']),dict(common,panel_id='cross_sensor',kind='B2',pair_id='pair')]
        doc=dict(kind='frozen_validation_panels_v1',source_identities=ds.identities,eligible_members={row['id']:row['cache_sha256'] for row in ds.rows[:2]},panels=cases,pair_admissions={'pair':pair})
        if edit:edit(doc,ds)
        path=Path(folder)/'SYNTHETIC_VAL_PANEL_ONLY.json';path.write_text(json.dumps(doc));frozen=dict(formal_config_sha256=frozen_identity(config),source_identities=ds.identities,merge_recipe=merge_recipe(curve,0),validation_panel_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return ds,path,config,frozen,curve
    def reader(self,args):
        ds,path,config,frozen,curve=args
        return FrozenValidationPanels(path,config=config,frozen_ids=frozen,dataset=ds,rate_curve=curve,selection_seed=0)

    def test_fixed_cases_target_TRAIN_statistics_and_native_grids(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));args=self.setup_case(folder);reader=self.reader(args);views=list(reader.iter_views(cell_indexed=True,retain_layouts=False));self.assertEqual([label for label,v in views],['A_fixed','blue_hidden','other_hidden','cross_sensor']);self.assertEqual(views[-1][1].train_stds['b0'],2.);self.assertEqual(args[0].reads,[0,1]);self.assertEqual(len(args[0].stats_reads),2);self.assertTrue(all(tuple(v.task.visible[0].reflectance.shape)==(3,4) for label,v in views))

    def test_collection_preserves_all_modes_gradients_and_training_RNG(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));args=self.setup_case(folder);reader=self.reader(args);encoder=ParticipationEncoder(rate_curve=args[-1],selection_seed=0,hyper_hidden=64);decoder=ReflectanceDecoder(indexed=True,cell_indexed=True);encoder.train();list(encoder.modules())[1].eval();decoder.eval();parts=list(encoder.modules())+list(decoder.modules());flags=[p.training for p in parts];parameter=next(encoder.parameters());parameter.grad=torch.full_like(parameter,.123);grad=parameter.grad.clone();before_weights=parameter.detach().clone();python=random.getstate();numpy=np.random.get_state();rng=torch.get_rng_state().clone()
            def fake(enc,dec,task,plan,**kwargs):
                self.assertFalse(torch.is_grad_enabled());self.assertFalse(enc.training);self.assertFalse(dec.training);self.assertFalse(kwargs['backward']);random.random();np.random.rand();torch.rand(3)
                return dict(status='fixture',loss=None if task.kind=='A' else .25,bands=[dict(band='fixture',loss=.25)])
            with patch('encoder_v3.validation_panels.stream_adaptive_task',side_effect=fake):result=reader.collect(encoder,decoder,cell_indexed=True,retain_layouts=False)
            self.assertIsNone(result['metric']);self.assertIsNone(result['rows'][0]['result']['loss']);self.assertEqual(len(result['rows']),4);self.assertEqual([p.training for p in parts],flags);self.assertEqual(random.getstate(),python);np.testing.assert_array_equal(np.random.get_state()[1],numpy[1]);torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0);torch.testing.assert_close(parameter.grad,grad,rtol=0,atol=0);torch.testing.assert_close(parameter,before_weights,rtol=0,atol=0)

    def test_exception_still_restores_modes_and_RNG(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));args=self.setup_case(folder);reader=self.reader(args);encoder=ParticipationEncoder(rate_curve=args[-1],selection_seed=0,hyper_hidden=64);decoder=ReflectanceDecoder();encoder.train();decoder.eval();rng=torch.get_rng_state().clone()
            def fail(*args,**kwargs):torch.rand(5);raise OSError('synthetic validation failure')
            with patch('encoder_v3.validation_panels.stream_adaptive_task',side_effect=fail):
                with self.assertRaises(OSError):reader.collect(encoder,decoder,cell_indexed=True,retain_layouts=False)
            self.assertTrue(encoder.training);self.assertFalse(decoder.training);torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)

    def test_role_duplicate_cases_and_changed_file_are_rejected(self):
        for edit in [lambda d,s:s.rows[0].update(role='TRAIN'),lambda d,s:d['panels'][1].update(panel_id='A_fixed'),lambda d,s:s.rows[1].update(cache_sha256='b'*64)]:
            with self.subTest(edit=edit),tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
                self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));args=self.setup_case(folder,edit)
                with self.assertRaises(ValueError):self.reader(args)
                self.assertEqual(args[0].reads,[])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));args=self.setup_case(folder);reader=self.reader(args);args[1].write_text('{}')
            with self.assertRaises(ValueError):list(reader.iter_views(cell_indexed=True,retain_layouts=False))
            self.assertEqual(args[0].reads,[])

    def test_actual_configuration_does_not_read_any_validation_file(self):
        config=json.loads((Path(__file__).parent/'config.json').read_text())
        with self.assertRaisesRegex(ValueError,'Training blocked'):FrozenValidationPanels('not-a-real-file',config=config,frozen_ids={},dataset=None,rate_curve=None,selection_seed=None)

    def test_unreviewed_precision_is_rejected_before_validation_reads(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));args=self.setup_case(folder);reader=self.reader(args);encoder=ParticipationEncoder(rate_curve=args[-1],selection_seed=0,hyper_hidden=64).double()
            with self.assertRaisesRegex(ValueError,'FP32 learned parameters'):reader.collect(encoder,ReflectanceDecoder(),cell_indexed=True,retain_layouts=False)
            self.assertEqual(args[0].reads,[])
