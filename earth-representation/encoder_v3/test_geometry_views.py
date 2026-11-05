import unittest
import torch
from pyproj import Transformer
from .data import Observation
from .geometry_views import common_metric_view,northing_shift
from .test_model import band


class GeometryViewChecks(unittest.TestCase):
    def test_utm_convention_preserves_native_measurements(self):
        source=band();source.crs='EPSG:32636';source.affine=(.37,0,482535,0,-1.21,-2163825)
        observation=Observation([source],dict(id='geometry_fixture',role='TRAIN'))
        viewed=common_metric_view(observation,'EPSG:32736');result=viewed.bands[0]
        self.assertEqual(northing_shift('EPSG:32636','EPSG:32736'),10_000_000.)
        for field in ('reflectance','valid_mask','wavelengths_nm','response','visible'):self.assertIs(getattr(source,field),getattr(result,field))
        shifted=source.cells()+source.cells().new_tensor([0,10_000_000,0,10_000_000])
        torch.testing.assert_close(result.cells(),shifted,atol=2e-9,rtol=0)
        a=Transformer.from_crs(source.crs,'EPSG:4326',always_xy=True).transform(source.affine[2],source.affine[5])
        b=Transformer.from_crs(result.crs,'EPSG:4326',always_xy=True).transform(result.affine[2],result.affine[5])
        self.assertAlmostEqual(a[0],b[0],places=10);self.assertAlmostEqual(a[1],b[1],places=10)
        self.assertFalse(viewed.metadata['geometry_view']['bands'][0]['resampled'])
        self.assertIs(common_metric_view(viewed,'EPSG:32736'),viewed)

    def test_projection_changes_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'Different native projections'):northing_shift('EPSG:32635','EPSG:32736')
        with self.assertRaisesRegex(ValueError,'Metric projected'):northing_shift('EPSG:4326','EPSG:4326')
