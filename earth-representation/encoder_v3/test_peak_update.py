import copy
import json
from pathlib import Path
import random
import unittest
import numpy as np
import torch
from .peak_update_check import compare


class PeakUpdateChecks(unittest.TestCase):
    def test_limits_exposure_support_and_rng_are_enforced(self):
        limits=json.loads((Path(__file__).parent/'reports/fp32_equivalence_tolerance_review.json').read_text(encoding='utf-8'))
        first={k:torch.ones(2) for k in ('modules','optimizer','gradients','predictions')}
        first.update(scheduler={'next_update':101},support=torch.tensor([True,False]),signature='exact',result={'rows':[dict(kind='A',source='S2',observation='x',exposure={'hidden':['a']},result=dict(loss=1.,bands=[dict(band='a',status='OK',count=1,unsupported=0,nmse=2.)]))]},rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=None))
        self.assertTrue(compare(first,copy.deepcopy(first),limits)['passed'])
        for field,change in [('predictions',lambda x:x.add_(3e-6)),('support',lambda x:x.logical_not_())]:
            second=copy.deepcopy(first);change(second[field]);self.assertFalse(compare(first,second,limits)['passed'])
        second=copy.deepcopy(first);second['rng']['torch'][0]^=1
        self.assertIn('RNG',compare(first,second,limits)['failures'])
        second=copy.deepcopy(first);second['result']['rows'][0]['exposure']={}
        with self.assertRaisesRegex(ValueError,'exposure'):compare(first,second,limits)
