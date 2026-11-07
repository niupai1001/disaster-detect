import copy,unittest
from dataclasses import replace
import torch
from .test_participation_batch import encoder
from .test_adaptive_window import source
from .model import ReflectanceDecoder
from .participation_decoder import MatrixADecoder
from .data import Observation
from .tasks import spatial_task


class MatrixADecoderChecks(unittest.TestCase):
    def test_all_adjacent_tokens_predictions_and_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(281);bands=[replace(source(.5,.5),record_id=str(i)) for i in range(3)];obs=Observation(bands,dict(id='fixture',role='TRAIN'));task=spatial_task(obs,(102.13,196.77,105.13,199.77));model=encoder();representation=model(task.visible,origin_xy=(99.8,180.));targets=task.targets[0].band.cells()[task.targets[0].mask.flatten()]
        a=ReflectanceDecoder(indexed=True,cell_indexed=True);b=MatrixADecoder(indexed=True,cell_indexed=True,bulk_geometry=True);b.load_state_dict(a.state_dict());condition=torch.randn(32,requires_grad=True);mask=torch.tensor(task.rectangle,dtype=torch.float64);reference=a(representation,targets,condition,kind='A',mask_bounds=mask);actual=b(representation,targets,condition,kind='A',mask_bounds=mask)
        torch.testing.assert_close(actual[1],reference[1]);torch.testing.assert_close(actual[0],reference[0],atol=2e-6,rtol=2e-6,equal_nan=True)
        want=torch.autograd.grad(reference[0][reference[1]].sum(),[representation.tokens,condition,*a.parameters()],retain_graph=True);got=torch.autograd.grad(actual[0][actual[1]].sum(),[representation.tokens,condition,*b.parameters()])
        for x,y in zip(got,want,strict=True):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-6)
        self.assertEqual(a.state_dict().keys(),b.state_dict().keys())

    def test_no_adjacent_support_and_non_A_reference(self):
        torch.set_num_threads(2);model=encoder();b=source(1.,1.);r=model([b],origin_xy=(99.8,180.));targets=b.cells()[:3];condition=torch.zeros(32);head=MatrixADecoder(indexed=True,cell_indexed=True);reference=ReflectanceDecoder(indexed=True,cell_indexed=True);reference.load_state_dict(head.state_dict())
        output,supported=head(r,targets,condition,kind='A',mask_bounds=torch.tensor([1000.,1000.,1001.,1001.],dtype=torch.float64));self.assertFalse(bool(supported.any()));self.assertTrue(bool(torch.isnan(output).all()))
        got=head(r,targets,condition,kind='B1');want=reference(r,targets,condition,kind='B1');torch.testing.assert_close(got[0],want[0],atol=0,rtol=0,equal_nan=True)
