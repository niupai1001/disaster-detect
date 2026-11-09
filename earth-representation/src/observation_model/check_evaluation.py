"""Check the fixed-QA metric and paired bootstrap on one real engineering evaluation."""
import json

import torch

from .compare import compare
from .data import ROOT, RELEASE
from .evaluate import correlation, measure, midranks


def main():
    truth=torch.tensor([0.,1.,2.],device="cuda")
    prediction=truth+1
    result=measure(prediction,truth,torch.ones(3,device="cuda",dtype=torch.bool),torch.zeros(3,device="cuda",dtype=torch.bool))
    assert result["count"]==3 and result["uncovered_target_pixels"]==3 and result["sum_squared"]==3
    ties=torch.tensor([3.,1.,1.,2.],device="cuda")
    assert torch.equal(midranks(ties),torch.tensor([3.,.5,.5,2.],device="cuda",dtype=torch.float64))
    assert abs(correlation(midranks(ties),midranks(ties))-1)<1e-12
    path=ROOT/"observation_model/outputs/phase1/evaluation"
    evaluation=json.loads((path/"ridge_cross_fixedqa.json").read_text())
    stats=json.loads((RELEASE/"statistics.json").read_text())["bands"]
    self_comparison=compare(evaluation,evaluation,"cross30",stats)
    assert not self_comparison["extension_gate_passed"]
    assert all(r["relative_rmse_improvement"]==0 and r["bootstrap95"]==[0,0] for r in self_comparison["comparisons"].values())
    model=json.loads((path/"engineering_cross_fixedqa.json").read_text())
    actual=compare(model,evaluation,"cross30",stats)
    (path/"engineering_comparison_physical_gate.json").write_text(json.dumps(actual,indent=2))
    (path/"acceptance_physical_gate.json").write_text(json.dumps(dict(status="passed",fixed_QA_counts_uncovered_predictions=True,
                     tie_aware_ranks=True,self_bootstrap_zero=True,actual_paired_comparison=actual,
                     boundary="engineering checkpoint comparison only; not a formal scientific outcome"),indent=2))
    print(json.dumps(dict(status="passed",independent_bootstrap_groups=actual["independent_bootstrap_groups"],
                          engineering_extension_gate_passed=actual["extension_gate_passed"])))


if __name__=="__main__":main()
