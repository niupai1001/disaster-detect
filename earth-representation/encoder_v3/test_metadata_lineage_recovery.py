import unittest
from .metadata_lineage_recovery import check_processing_identity,native_crop_check


class MetadataRecoveryChecks(unittest.TestCase):
    def test_processing_fields_must_match_frozen_properties(self):
        properties={'datetime':'2020-01-01T00:00:00Z','s2:processing_baseline':'05.00','earthsearch:boa_offset_applied':True}
        member=dict(product='explicit',date=properties['datetime'],calibration_revision=dict(source_baseline='05.00',source_flag=True));item=dict(id='explicit',properties=properties);canonical=dict(properties=properties)
        self.assertEqual(check_processing_identity(item,member,canonical)['s2:processing_baseline'],'05.00')
        with self.assertRaises(ValueError):check_processing_identity(dict(item,properties=dict(properties,**{'earthsearch:boa_offset_applied':False})),member,canonical)

    def test_integer_native_crop_without_resampling(self):
        asset={'proj:transform':[10,0,100,0,-10,500],'proj:shape':[100,100]}
        self.assertEqual(native_crop_check((10,0,120,0,-10,470),(4,5),asset)['source_pixel_offset_rc'],[3,2])
        for affine,shape in [((11.25,0,120,0,-11.25,470),(4,5)),((10,0,121,0,-10,470),(4,5)),((10,0,120,0,-10,470),(200,5))]:
            with self.assertRaises(ValueError):native_crop_check(affine,shape,asset)
