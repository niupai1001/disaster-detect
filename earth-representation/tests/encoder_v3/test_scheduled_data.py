import copy,hashlib,json,random,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from encoder_v3.data import Observation
from encoder_v3.model import BandObservation
from encoder_v3.scheduled_data import FrozenABSchedule
from encoder_v3.participation_layout import AreaEquivalentHillRate
from encoder_v3.participation_encoder import merge_recipe
from encoder_v3.route_control import frozen_identity
from encoder_v3.plan_prefetch import move_views
from encoder_v3.route_step import update_signature
from encoder_v3.formal_test_fixtures import populate as populate_approvals


class DatasetFixture:
    role='TRAIN';identities={'members':'SYNTHETIC_METADATA_ONLY'}
    def __init__(self,single=False):
        self.rows=[];self.observations=[];self.reads=[];self.stats_reads=[]
        for i in range(8):
            name=f'fixture{i}';version=hashlib.sha256(name.encode()).hexdigest();bands=[BandObservation(torch.zeros(3,4),torch.ones(3,4,dtype=torch.bool),torch.tensor([450.,500.,550.]),torch.tensor([0.,1.,0.]),(10.,0.,0.,0.,-10.,30.),'EPSG:32631',f'b{j}') for j in range(1 if single and i==0 else 2)];metadata=dict(id=name,role='TRAIN',cache_sha256=version,frozen_source_ids=self.identities,sensor='S2' if i%2==0 else 'L8')
            self.observations.append(Observation(bands,metadata));self.rows.append(dict(id=name,role='TRAIN',cache_sha256=version,sensor=metadata['sensor'],bands=[dict(band_key=b.record_id) for b in bands]))
    def __getitem__(self,index):self.reads.append(index);return copy.deepcopy(self.observations[index])
    def engineering_train_stds(self,obs):self.stats_reads.append(obs.identifier);return {b.record_id:float(int(obs.identifier[-1])+1) for b in obs.bands}


class FrozenScheduleChecks(unittest.TestCase):
    def setup_case(self,folder,*,pairs=False,single=False,edit=None):
        ds=DatasetFixture(single);real=json.loads(((Path(__file__).resolve().parents[2] / 'src/encoder_v3')/'config.json').read_text());config=dict(required_before_training={k:None for k in real['required_before_training']},formal_step={'chunk_pixels':7});populate_approvals(config)
        curve=AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556);slots=[dict(source_id=r['id'],origin_xy=[0.,0.],A_rectangle=[0.,0.,10.,20.],B=dict(kind='B1_ABSENT') if single and i==0 else dict(kind='B1',hidden_band_ids=['b0'])) for i,r in enumerate(ds.rows)];admissions={}
        if pairs:
            for i,target in [(0,1),(1,0)]:admissions[str(i)]=dict(source_id=f'fixture{i}',target_id=f'fixture{target}',checks={k:True for k in ['space','registration','time_difference','surface_change','qa','srf','role_isolation']})
            for i in range(4):slots[i]['B']=dict(kind='B2',pair_id=str(i)) if i<2 else dict(kind='B2_ABSENT')
        document=dict(kind='frozen_AB_schedule_v1',source_identities=ds.identities,eligible_members={r['id']:r['cache_sha256'] for r in ds.rows},pair_admissions=admissions,updates=[dict(logical_update=0,P_m=60.,slots=slots)])
        if edit:edit(document,ds)
        path=Path(folder)/'SYNTHETIC_NOT_OPERATIONAL.json';path.write_text(json.dumps(document));frozen=dict(formal_config_sha256=frozen_identity(config),source_identities=ds.identities,merge_recipe=merge_recipe(curve,0),AB_schedule_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return ds,path,config,frozen,curve
    def loader(self,args):
        ds,path,config,frozen,curve=args
        return FrozenABSchedule(path,config=config,frozen_ids=frozen,dataset=ds,rate_curve=curve,selection_seed=0)

    def test_same_frozen_update_replays_without_sampling_or_cropping(self):
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')));args=self.setup_case(folder);loader=self.loader(args);ds=args[0];python=random.getstate();numpy=np.random.get_state();rng=torch.get_rng_state().clone();a=loader.prepare(0,cell_indexed=True,retain_layouts=False);b=loader.prepare(0,cell_indexed=True,retain_layouts=False)
            self.assertEqual(a['AB_input_signature'],b['AB_input_signature']);self.assertEqual(a['source_members'],[f'fixture{i}' for i in range(8)]);self.assertEqual(random.getstate(),python);np.testing.assert_array_equal(np.random.get_state()[1],numpy[1]);self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(len(a['views']),16)
            self.assertTrue(all(tuple(v.task.visible[0].reflectance.shape)==(3,4) for v in a['views']));self.assertEqual(update_signature(move_views(a['views'],'cpu')),a['AB_input_signature']);self.assertEqual(ds.reads[:8],list(range(8)))

    def test_B2_uses_target_statistics_and_verified_missing_pairs(self):
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')));args=self.setup_case(folder,pairs=True);packet=self.loader(args).prepare(0,cell_indexed=True,retain_layouts=False);ds=args[0]
            B2=[v for v in packet['views'] if v.task.kind=='B2' and v.absence_reason is None];self.assertEqual(len(B2),2);self.assertEqual(B2[0].train_stds['b0'],2.);self.assertEqual(B2[1].train_stds['b0'],1.);self.assertEqual(len(packet['absent_tasks']),2);self.assertEqual(len(ds.reads),8);self.assertEqual(len(ds.stats_reads),8)

    def test_single_band_absence_is_derived_from_frozen_band_inventory(self):
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')));packet=self.loader(self.setup_case(folder,single=True)).prepare(0,cell_indexed=True,retain_layouts=False);self.assertEqual(packet['absent_tasks'][0]['reason'],'NO_HIDEABLE_BAND')

    def test_false_absence_and_slot_imbalance_rejected_before_pixel_reads(self):
        mutations=[lambda d,s:d['updates'][0]['slots'][0].update(B={'kind':'B1_ABSENT'}),lambda d,s:d['updates'][0]['slots'][0].update(B={'kind':'B2_ABSENT'}),lambda d,s:d['updates'][0]['slots'][4].update(B={'kind':'B2_ABSENT'})]
        for mutate in mutations:
            with self.subTest(mutation=mutate),tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
                self.assertTrue(Path(folder).resolve().is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')));args=self.setup_case(folder,pairs=True,edit=mutate)
                with self.assertRaises(ValueError):self.loader(args)
                self.assertEqual(args[0].reads,[])

    def test_role_version_schedule_changes_and_invalid_indices_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')));args=self.setup_case(folder);args[0].rows[0]['role']='VAL'
            with self.assertRaises(ValueError):self.loader(args)
            args=self.setup_case(folder);loader=self.loader(args)
            for index in [True,-1,1]:
                with self.assertRaises(ValueError):loader.prepare(index,cell_indexed=True,retain_layouts=False)
            args[1].write_text('{}')
            with self.assertRaises(ValueError):loader.prepare(0,cell_indexed=True,retain_layouts=False)
            self.assertEqual(args[0].reads,[])

    def test_actual_unfrozen_configuration_is_blocked_before_reading_schedule(self):
        config=json.loads(((Path(__file__).resolve().parents[2] / 'src/encoder_v3')/'config.json').read_text())
        with self.assertRaisesRegex(ValueError,'Training blocked'):FrozenABSchedule('not-a-real-file',config=config,frozen_ids={},dataset=None,rate_curve=None,selection_seed=None)

    def test_same_sensor_cannot_be_presented_as_cross_sensor_pair(self):
        def change(document,dataset):dataset.rows[1]['sensor']='S2'
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')));args=self.setup_case(folder,pairs=True,edit=change)
            with self.assertRaisesRegex(ValueError,'cross-sensor pair'):self.loader(args)
            self.assertEqual(args[0].reads,[])
