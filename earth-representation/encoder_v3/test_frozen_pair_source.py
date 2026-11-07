import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import torch
from .data import Observation
from .model import BandObservation
from .tasks import spatial_task
from .pair_geometry import reviewed_pair_task,observation_version,canonical_sha,frame_for_task
from .route_step import validate_frozen_task_source


class FrozenPairSourceChecks(unittest.TestCase):
    def fixtures(self,folder):
        def observation(name,left):
            band=BandObservation(torch.zeros(3,4),torch.ones(3,4,dtype=torch.bool),torch.tensor([450.,500.,550.]),torch.tensor([0.,1.,0.]),(10.,0.,left,0.,-10.,30.),'EPSG:32631','fixture_band')
            return Observation([band],dict(id=name,role='TRAIN',geometry_data_version='synthetic_'+name,frozen_source_ids={'members':'a'*64}))
        source,target=observation('synthetic_S2',0.),observation('synthetic_Landsat',20.)
        payload=dict(mechanism='query_offset_native_encoder_v1',S2_id=source.identifier,Landsat_id=target.identifier,translation_m=[20.,0.],crs='EPSG:32631',role='TRAIN',versions={source.identifier:observation_version(source),target.identifier:observation_version(target)})
        path=Path(folder)/'SYNTHETIC_UNIT_FIXTURE_ONLY.json';path.write_text(json.dumps(dict(status='USER_CONFIRMED_PAIR_OFFSET_VALUES',record_sha256=canonical_sha(payload),fixture='NOT_OPERATIONAL_APPROVAL')))
        record=dict(payload,offset_review_file=str(path),offset_review_sha256=hashlib.sha256(path.read_bytes()).hexdigest());admission=dict(source_id=source.identifier,target_id=target.identifier,checks={k:True for k in ['space','registration','time_difference','surface_change','qa','srf','role_isolation']},pair_geometry=record)
        task=reviewed_pair_task(source,target,admission);frozen=dict(source_identities={'members':'a'*64},pair_geometry_records=[canonical_sha(payload)])
        return source,target,admission,task,frozen

    def test_reviewed_and_run_frozen_query_extension_is_accepted_bidirectionally(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));source,target,admission,task,frozen=self.fixtures(folder)
            before=copy.deepcopy(task.metadata);validate_frozen_task_source(task,frozen);self.assertEqual(before,task.metadata)
            inverse=dict(admission,source_id=target.identifier,target_id=source.identifier);reverse=reviewed_pair_task(target,source,inverse);validate_frozen_task_source(reverse,frozen)
            self.assertEqual(frame_for_task(task).query_offset_xy,(-20.,-0.));self.assertEqual(frame_for_task(reverse).query_offset_xy,(20.,0.))

    def test_missing_or_different_run_allowlist_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));_,_,_,task,frozen=self.fixtures(folder)
            for records in [None,[],['b'*64]]:
                bad=dict(frozen,pair_geometry_records=records)
                with self.assertRaises(ValueError):validate_frozen_task_source(task,bad)

    def test_data_version_and_unknown_extensions_remain_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));source,_,_,task,frozen=self.fixtures(folder)
            bad=copy.deepcopy(task);bad.metadata['frozen_source_ids']['extra']='unfrozen'
            with self.assertRaises(ValueError):validate_frozen_task_source(bad,frozen)
            bad=copy.deepcopy(task);bad.metadata['geometry_data_version']='changed'
            with self.assertRaises(ValueError):validate_frozen_task_source(bad,frozen)
            ordinary=spatial_task(source,(0.,0.,10.,20.));validate_frozen_task_source(ordinary,frozen);ordinary.metadata['frozen_source_ids']['pair_geometry_record_sha256']='b'*64
            with self.assertRaises(ValueError):validate_frozen_task_source(ordinary,frozen)

    def test_mechanism_only_does_not_approve_offset_values(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));_,_,admission,task,frozen=self.fixtures(folder)
            Path(admission['pair_geometry']['offset_review_file']).write_text(json.dumps(dict(status='USER_CONFIRMED_PAIR_QUERY_MECHANISM_ONLY')))
            with self.assertRaises(ValueError):validate_frozen_task_source(task,frozen)
