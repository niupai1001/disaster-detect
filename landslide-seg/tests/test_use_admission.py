import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from landslide_seg.data.use_admission import AdmittedUSEData, sha256
from build_use_data_admission import counterpart_requests


def row(name, sensor, x=500000, role='TRAIN', gsd=10):
    return dict(file=name, sha256='0'*64, sensor=sensor, role=role, crs='EPSG:32650',
                transform=[gsd, 0, x, 0, -gsd, 4000000],
                date='2024-01-01', cache_audit={'shape':[10,128,128]})


class AdmissionTest(unittest.TestCase):
    def test_disjoint_is_not_paired(self):
        pairs, tasks, conflicts = counterpart_requests([row('a','s2'),row('b','landsat',510000)])
        self.assertEqual(pairs, [])
        self.assertEqual(len(tasks), 2)
        self.assertEqual(conflicts, [])

    def test_asymmetric_native_footprints(self):
        pairs, tasks, _ = counterpart_requests([row('a','s2'),row('b','landsat',gsd=30)])
        self.assertEqual([p['anchor'] for p in pairs], ['a'])
        self.assertEqual([t['anchor_file'] for t in tasks], ['b'])

    def test_role_conflict_never_pairs(self):
        pairs, tasks, conflicts = counterpart_requests([row('a','s2'),row('b','landsat',role='PRETRAIN_VAL')])
        self.assertEqual(pairs, [])
        self.assertEqual(len(conflicts), 2)

    def test_guard_never_requested(self):
        pairs, tasks, _ = counterpart_requests([row('g','s2',role='GUARD')])
        self.assertEqual((pairs,tasks), ([],[]))

    def test_release_rejects_tamper_and_guard(self):
        with self.assertRaises(ValueError):
            AdmittedUSEData('does-not-exist', role='GUARD')
        temporary_root = ROOT/'outputs/use_data_admission/test_tmp'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            root = Path(directory)
            (root/'admitted.jsonl').write_text('{}\n')
            contract = dict(ok=True, project_root=directory,
                artifacts={'admitted.jsonl': sha256(root/'admitted.jsonl')}, source_sha256={})
            (root/'release.json').write_text(json.dumps(contract))
            (root/'admitted.jsonl').write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                AdmittedUSEData(root)


if __name__ == '__main__':
    unittest.main()
