import copy,unittest
import torch
from tests.encoder_v3.test_participation_batch import encoder


class PackedOutputChecks(unittest.TestCase):
    def test_exact_output_order_and_gradient_for_mixed_lengths(self):
        torch.set_num_threads(2);torch.manual_seed(351);a=encoder();b=copy.deepcopy(a);b.packed_local_output=True;a.local_batch_size=b.local_batch_size=3
        groups=[torch.randn(n,128,requires_grad=True) for n in (4,1,2,1,9,4,2,1,9,1)];clones=[x.detach().clone().requires_grad_(True) for x in groups];x=a._local_tokens(groups);y=b._local_tokens(clones)
        torch.testing.assert_close(x,y,atol=0,rtol=0);target=torch.randn_like(x);(x*target).sum().backward();(y*target).sum().backward()
        for u,v in zip(groups,clones):torch.testing.assert_close(u.grad,v.grad,atol=2e-6,rtol=2e-6)
        for u,v in zip(a.parameters(),b.parameters()):
            if u.grad is None:self.assertIsNone(v.grad)
            else:torch.testing.assert_close(u.grad,v.grad,atol=2e-6,rtol=2e-6)
        self.assertEqual(a.get_extra_state(),b.get_extra_state());self.assertEqual(a.state_dict().keys(),b.state_dict().keys())

    def test_full_encoder_and_synthetic_joint_resume(self):
        from unittest.mock import patch
        from tests.encoder_v3.test_participation_batch import ParticipationBatchChecks
        from tests.encoder_v3 import test_participation_joint as joint_tests
        a=encoder();b=copy.deepcopy(a);b.packed_local_output=True
        for bands,origin in ParticipationBatchChecks().observations():torch.testing.assert_close(a(bands,origin_xy=origin).tokens,b(bands,origin_xy=origin).tokens,atol=0,rtol=0)
        with patch.object(joint_tests.ParticipationJointChecks,'encoder',lambda self:b):
            joint_tests.ParticipationJointChecks().test_full_joint_update_ema_and_next_step_restore()
