from contextlib import closing
import unittest
from .data import Observation
from .prefetch import ReplayLoader
from .test_model import band


class FixtureDataset:
    identities={'members':'synthetic-fixture'}
    rows=[dict(id=str(i)) for i in range(4)]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        value=band();value.reflectance.fill_(index/10)
        return Observation([value],dict(id=self.rows[index]['id'],role='TRAIN',frozen_source_ids=self.identities))


class PrefetchChecks(unittest.TestCase):
    def loader(self,workers):
        entries=[dict(logical_update=i,members=[str(i%2),str(2+i%2)],patch_m=60 if i%2 else 120) for i in range(3)]
        return ReplayLoader(FixtureDataset(),entries,source_identities=FixtureDataset.identities,effective_batch=2,workers=workers,pin_memory=False)

    def test_prefetch_cursor_replay(self):
        for workers in (0,2):
            loader=self.loader(workers)
            with closing(iter(loader)) as iterator:
                first=next(iterator);self.assertEqual(first.logical_update,0)
                state=loader.state_dict();self.assertEqual(state['next_update'],0)
            recovered=self.loader(workers);recovered.load_state_dict(state)
            with closing(iter(recovered)) as iterator:
                repeated=next(iterator)
                self.assertEqual(first.plan,repeated.plan);self.assertEqual([o.identifier for o in first.observations],[o.identifier for o in repeated.observations])
                import torch
                for a,b in zip(first.observations,repeated.observations):torch.testing.assert_close(a.bands[0].reflectance,b.bands[0].reflectance)
                recovered.commit(0);next_state=recovered.state_dict()
                with self.assertRaises(ValueError):recovered.commit(1)
            resumed=self.loader(workers);resumed.load_state_dict(next_state)
            with closing(iter(resumed)) as iterator:self.assertEqual(next(iterator).logical_update,1)

    def test_schedule_change_rejected(self):
        a=self.loader(0);b=self.loader(0);state=a.state_dict();state['schedule_sha256']='changed'
        with self.assertRaises(ValueError):b.load_state_dict(state)
        with self.assertRaises(ValueError):b.commit(0)
