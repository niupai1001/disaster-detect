import unittest
from .split_audit import audit


def row(identifier,role,x,scene='scene'):
    return dict(id=identifier,role=role,sensor='S2',date='2024-01-01T00:00:00Z',acquisition_key=scene,product=scene,release_isolation_group=identifier,cache_sha256=identifier,bands=[dict(shape=[2,2],crs='EPSG:32650',transform=[10.,0.,x,0.,-10.,200.])])


class SplitAuditChecks(unittest.TestCase):
    def test_identities_are_distinct_from_spatial_overlap(self):
        rows=[row('train','TRAIN',500000.),row('overlap','VAL',500005.),row('far','VAL',501000.),row('touch','VAL',500020.)]
        result=audit(rows,[])
        self.assertEqual(len(result['shared_cross_role_identities']['acquisition_key']),1)
        self.assertEqual([(x['train_id'],x['val_id']) for x in result['train_val_overlaps']],[('train','overlap')]);self.assertEqual(result['train_val_overlaps'][0]['overlap_m2'],300.)
        self.assertEqual(result['protected_bounds_candidates'],[])

    def test_missing_protected_geometry_not_silently_safe(self):
        result=audit([row('train','TRAIN',500000.)],[dict(id='guard',effective_partition='GUARD',bbox=None)])
        self.assertEqual(result['protected_bbox_unmeasured'],['guard'])
