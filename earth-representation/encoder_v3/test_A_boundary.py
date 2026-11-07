import unittest
from dataclasses import replace
import torch
from .test_adaptive_window import source
from .data import Observation
from .tasks import spatial_task
from .participation_layout import AreaEquivalentHillRate
from .participation_encoder import ParticipationEncoder,prepare_participation_task
from .model import ReflectanceDecoder
from .participation_decoder import MatrixADecoder
from .spatial_index import SpatialIndex
from .participation_geometry import BulkSpatialIndex


class BoundaryMergeChecks(unittest.TestCase):
    def fixture(self):
        band=replace(source(.5,.5),reflectance=torch.arange(16).reshape(4,4).float()/100,valid_mask=torch.ones(4,4,dtype=torch.bool),affine=(.5,0.,0.,0.,-.5,2.));task=spatial_task(Observation([band],dict(id='generated',role='TRAIN')),(.5,.5,2.5,2.5));curve=AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556)
        return task,curve

    def test_centroid_inside_does_not_remove_visible_boundary_token(self):
        torch.set_num_threads(2);task,curve=self.fixture();plan=prepare_participation_task(task,rate_curve=curve,selection_seed=0,origin_xy=(0,0),chunk_pixels=3,cell_indexed=True)
        self.assertEqual(int(task.visible[0].keep().sum()),7);self.assertEqual(plan.bands[0].supported_count,9);mask=torch.tensor(task.rectangle,dtype=torch.float64)
        for index in (SpatialIndex(plan.geometry),BulkSpatialIndex(plan.geometry)):self.assertEqual(index.a_tokens(plan.geometry,mask).tolist(),[0])
        model=ParticipationEncoder(rate_curve=curve,selection_seed=0,hyper_hidden=64);representation=model(task.visible,origin_xy=(0,0));targets=task.targets[0].band.cells()[plan.bands[0].indices];condition=model.srf(task.targets[0].band.wavelengths_nm,task.targets[0].band.response)
        a=ReflectanceDecoder(indexed=True,cell_indexed=True);b=MatrixADecoder(indexed=True,cell_indexed=True,bulk_geometry=True);b.load_state_dict(a.state_dict());x,supported=a(representation,targets,condition,kind='A',mask_bounds=mask);y,other=b(representation,targets,condition,kind='A',mask_bounds=mask)
        self.assertTrue(bool(supported.all()));self.assertTrue(bool(torch.isfinite(x).all()));torch.testing.assert_close(supported,other);torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-6)
        invalid=replace(representation,valid=torch.zeros_like(representation.valid));self.assertEqual(len(SpatialIndex(invalid).a_tokens(invalid,mask)),0)

    def test_hidden_reflectance_stays_isolated(self):
        torch.set_num_threads(2);task,curve=self.fixture();model=ParticipationEncoder(rate_curve=curve,selection_seed=0,hyper_hidden=64);band=task.visible[0];changed=replace(band,reflectance=band.reflectance.clone());changed.reflectance[~band.keep()]=float('nan');x=model([band],origin_xy=(0,0));y=model([changed],origin_xy=(0,0));torch.testing.assert_close(x.tokens,y.tokens,atol=0,rtol=0)
