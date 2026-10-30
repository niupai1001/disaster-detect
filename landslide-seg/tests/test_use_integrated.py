import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from integrate_use_sources import materialize,OUT,publish
from landslide_seg.data.use_integrated import IntegratedUSEData
from landslide_seg.data.use_admission import sha256


class IntegrationTest(unittest.TestCase):
    def test_invalid_pixel_fraction_rejected(self):
        with self.assertRaises(ValueError):
            materialize('never_written',np.ones((2,4,4)),np.zeros((2,4,4),bool),np.zeros((2,3)),np.zeros((2,3),bool),{})

    def test_guard_cannot_be_loaded(self):
        with self.assertRaises(ValueError):IntegratedUSEData('missing',role='GUARD')

    def test_tampered_manifest_rejected(self):
        base=OUT/'test_tmp';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as name:
            p=Path(name)
            (p/'release.json').write_text(json.dumps(dict(ok=True,manifest_sha256='bad')))
            (p/'manifest.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):IntegratedUSEData(p)

    def test_duplicate_identity_rejected(self):
        with self.assertRaises(ValueError):publish([{'id':'same'},{'id':'same'}],'test_duplicate')

    def test_two_good_views_can_have_insufficient_joint_quality(self):
        base=OUT/'test_tmp';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as name:
            path=Path(name);rows=[]
            for i,sensor in enumerate(('s2','landsat')):
                valid=np.ones((2,10,10),bool)
                valid.reshape(2,-1)[:,0:15 if i==0 else 0]=False
                if i:valid.reshape(2,-1)[:,-15:]=False
                cache=path/(sensor+'.npz');np.savez_compressed(cache,x=np.ones((2,10,10)),pixel_valid=valid)
                rows.append(dict(id=sensor,role='TRAIN',sensor=sensor,origin='four_view_pilot',site_id='same',
                    crs='EPSG:32650',transform=[10,0,0,0,-10,0],cache_uri=str(cache),sha256=sha256(cache),date='2024-01-01'))
            publish(rows,str(path.relative_to(OUT)))
            pair=json.loads((path/'pairs.jsonl').read_text())
            self.assertAlmostEqual(pair['joint_valid_ratio'],.7)
            self.assertFalse(pair['qualified'])

    def test_identical_source_pixels_not_double_counted(self):
        base=OUT/'test_tmp';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as name:
            path=Path(name)
            row=dict(id='one',role='TRAIN',sensor='s2',sha256='same-content',crs='EPSG:32650',
                     transform=[10,0,0,0,-10,0],date='2024-01-01')
            publish([row,dict(row,id='two')],str(path.relative_to(OUT)))
            result=json.loads((path/'release.json').read_text())
            self.assertEqual(result['counts']['TRAIN/s2'],1)
            self.assertEqual(result['duplicate_aliases'],{'two':'one'})


if __name__=='__main__':unittest.main()
