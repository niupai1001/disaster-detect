"""Small synthetic cases verify diagnostic meanings, not just array shapes."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_object_errors import describe_objects


class ObjectErrorAuditTest(unittest.TestCase):
    def setUp(self):
        self.gt=np.zeros((24,24),dtype=bool)
        self.gt[8:11,8:11]=True
        self.valid=np.ones_like(self.gt)

    def test_no_response_and_subthreshold_are_distinct(self):
        for p,expected in [(.02,'no_hit_below_0.1'),(.3,'no_hit_0.1_to_0.5')]:
            row=describe_objects(np.full(self.gt.shape,p),self.gt,self.valid)[0]
            self.assertEqual(row['category'],expected)
            self.assertEqual(row['hit_px'],0)

    def test_perfect_prediction_and_incomplete_coverage(self):
        row=describe_objects(self.gt.astype(float),self.gt,self.valid)[0]
        self.assertTrue(row['matched']);self.assertEqual(row['max_iou'],1)
        prob=np.zeros(self.gt.shape);prob[9,9]=1
        row=describe_objects(prob,self.gt,self.valid)[0]
        self.assertEqual(row['category'],'overlap_but_unmatched')
        self.assertEqual(row['interior_fn'],0);self.assertEqual(row['boundary_fn'],8)

    def test_merge_does_not_mean_zero_response(self):
        self.gt[8:11,15:18]=True
        prob=np.zeros(self.gt.shape);prob[8:11,8:18]=1
        rows=describe_objects(prob,self.gt,self.valid)
        self.assertEqual(sum(r['matched'] for r in rows),1)
        self.assertTrue(all(r['merge_flag'] for r in rows))
        self.assertEqual(sum(r['category']=='overlap_but_unmatched' for r in rows),1)

    def test_fragment_and_invalid_prediction(self):
        self.gt[:]=False;self.gt[8:15,8:15]=True
        prob=np.zeros(self.gt.shape);prob[8:10,8:10]=1;prob[13:15,13:15]=1
        row=describe_objects(prob,self.gt,self.valid)[0]
        self.assertTrue(row['fragment_flag']);self.assertFalse(row['matched'])
        self.valid[8:15,8:15]=False
        row=describe_objects(prob,self.gt,self.valid)[0]
        self.assertEqual(row['category'],'no_valid_pixels');self.assertEqual(row['hit_px'],0)

    def test_nearby_response_is_not_a_hit(self):
        prob=np.zeros(self.gt.shape);prob[8:11,12:15]=1
        row=describe_objects(prob,self.gt,self.valid)[0]
        self.assertTrue(row['nearby_prediction_2px']);self.assertEqual(row['hit_px'],0)
        self.assertFalse(row['matched'])

    def test_existing_metric_excludes_one_pixel_near_edge(self):
        self.gt[:]=False;self.gt[1,8]=True
        row=describe_objects(self.gt.astype(float),self.gt,self.valid)[0]
        self.assertTrue(row['truncated']);self.assertFalse(row['matched'])


if __name__=='__main__':unittest.main()
