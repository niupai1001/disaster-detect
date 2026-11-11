import copy
import json
import numpy as np
import pytest
import torch

from encoder_v3.hierarchical_revised import model as module
from encoder_v3.hierarchical_revised.attention import IndexedAttention, Metrics
from encoder_v3.hierarchical_revised.grouped import Prepared, Factorized
from encoder_v3.hierarchical_revised.model import Config
from encoder_v3.hierarchical_revised.profile import CONFIG, synthetic
from encoder_v3.hierarchical_revised.tasks import Pretraining


def test_partitioned_softmax_exact_outputs_and_gradients():
    torch.manual_seed(51)
    q, k, v = [torch.randn(n, 4, 8, requires_grad=True) for n in (7, 11, 11)]
    graph = Prepared(np.array([0, 0, 2, 3]), np.array([1, 2, 0, 3]),
                     ((np.array([0, 1, 2]), np.array([4, 5, 6])),
                      (np.array([3, 4, 5]), np.array([4, 7, 8, 9]))), 7)
    source, destination = graph.expanded()
    actual = graph.attend(q, k, v, None, "test")
    expected, _ = IndexedAttention.apply(q, k, v, lambda a,b:(source,destination), 7, "test", None, q.device)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    gradient = torch.randn_like(actual)
    first = torch.autograd.grad(actual, (q,k,v), gradient, retain_graph=True)
    second = torch.autograd.grad(expected, (q,k,v), gradient)
    for a,b in zip(first,second):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize("task", ["A","B1"])
@pytest.mark.parametrize("bands", [12, 96])
def test_complete_model_matches_expanded_reference(task, bands, monkeypatch):
    torch.set_num_threads(2)
    cfg = Config(**json.loads(CONFIG.read_text())["model"])
    sample = synthetic(torch.device("cpu"), bands=bands)
    torch.manual_seed(24)
    fast = Pretraining(cfg)
    reference = copy.deepcopy(fast)
    metrics = Metrics()
    first = fast(sample,kind=task,seed=17,metrics=metrics)
    first.loss.backward()
    class Expanded:
        def __init__(self,*args):
            self.graph = Factorized(*args)
        def __call__(self,start,end):
            return self.graph(start,end)
    monkeypatch.setattr(module,"Factorized",Expanded)
    reference_metrics = Metrics()
    second = reference(sample,kind=task,seed=17,metrics=reference_metrics)
    second.loss.backward()
    torch.testing.assert_close(first.predictions,second.predictions,atol=2e-6,rtol=1e-5)
    for a,b in zip(fast.parameters(),reference.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,atol=5e-6,rtol=2e-4)
    for name in metrics.attention:
        assert metrics.attention[name]["pairs"] == reference_metrics.attention[name]["pairs"]
