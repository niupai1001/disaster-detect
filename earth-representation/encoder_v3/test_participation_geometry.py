import unittest
from dataclasses import replace
import torch
from .test_participation_batch import encoder
from .test_adaptive_window import source
from .model import spatial_hide
from .spatial_index import SpatialIndex
from .participation_geometry import BulkSpatialIndex


class BulkGeometryChecks(unittest.TestCase):
    def test_same_candidate_order_and_edges_with_masks_and_noninteger_gsd(self):
        torch.set_num_threads(2)
        for gx,gy in ((.5,1.21),(10.,20.)):
            band=source(gx,gy);band.valid_mask[0,0]=False;rectangle=(band.affine[2]+gx*3,band.affine[5]-gy*7,band.affine[2]+gx*8,band.affine[5]-gy*2);visible,_=spatial_hide(band,rectangle);r=encoder()([visible],origin_xy=(99.8,180.));mask=torch.tensor(rectangle,dtype=torch.float64);targets=band.cells()[:17];a=SpatialIndex(r);b=BulkSpatialIndex(r)
            torch.testing.assert_close(a.a_tokens(r,mask),b.a_tokens(r,mask),atol=0,rtol=0)
            for x,y in zip(a.edges(r,targets,'A',mask),b.edges(r,targets,'A',mask),strict=True):torch.testing.assert_close(x,y,atol=0,rtol=0)
            torch.testing.assert_close(b.a_tokens(r,mask),a.a_tokens(r,mask),atol=0,rtol=0)
        invalid=replace(r,valid=torch.zeros_like(r.valid));self.assertEqual(len(BulkSpatialIndex(invalid).a_tokens(invalid,mask)),0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_metadata_filter_matches_reference(self):
        from .data import Observation,to_device
        torch.set_num_threads(2);band=source(.5,1.21);rectangle=(101.,190.,104.,197.);visible,_=spatial_hide(band,rectangle);gpu=to_device(Observation([visible],dict(id='fixture')),'cuda').bands;r=encoder().cuda()(gpu,origin_xy=(99.8,180.));mask=r.bounds.new_tensor(rectangle)
        torch.testing.assert_close(SpatialIndex(r).a_tokens(r,mask),BulkSpatialIndex(r).a_tokens(r,mask),atol=0,rtol=0)

    def test_A_support_planning_never_materializes_cartesian_edges(self):
        from unittest.mock import patch
        from .data import Observation
        from .tasks import spatial_task
        from .participation_encoder import prepare_participation_task
        from .participation_layout import AreaEquivalentHillRate
        torch.set_num_threads(2);band=source(.5,1.21);task=spatial_task(Observation([band],dict(id='fixture',role='TRAIN')),(101.,190.,104.,197.));curve=AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556)
        with patch.object(SpatialIndex,'edges',side_effect=AssertionError('Cartesian A edges must not be built for support counting')):
            plan=prepare_participation_task(task,rate_curve=curve,selection_seed=0,origin_xy=(99.8,180.),chunk_pixels=3,cell_indexed=True)
        mask=torch.tensor(task.rectangle,dtype=torch.float64)
        for item in plan.bands:
            q,_,_,_=plan.geometry.spatial_index.edges(plan.geometry,item.target.band.cells()[item.indices],'A',mask)
            self.assertEqual(item.supported_count,len(torch.unique(q)))
