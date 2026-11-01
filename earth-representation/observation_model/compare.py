"""Paired acquisition-group bootstrap using the same measured pixels for both methods."""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import RELEASE


def scene_key(row):
    return (row["id"],row["direction"],row["band"],tuple(row["hidden_bands"]))


def compare(candidate, reference, mask_set, statistics, repeats=1000):
    left={scene_key(r):r for r in candidate["scenes"] if r["mask_set"]==mask_set}
    right={scene_key(r):r for r in reference["scenes"] if r["mask_set"]==mask_set}
    if not left or left.keys()!=right.keys():raise ValueError("comparison members, directions and masks must match")
    if any(left[k]["count"]!=right[k]["count"] for k in left):raise ValueError("methods did not use identical measured target QA")
    # A location's two acquisitions and any windows sharing either acquisition form one resampling unit.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    groups={}
    for row in left.values():
        acquisition=row["acquisition"];root=find(acquisition)
        if row["group"] in groups:parent[find(root)]=find(groups[row["group"]])
        else:groups[row["group"]]=root
    units=sorted({find(r["acquisition"]) for r in left.values()})
    bands=sorted({(r["direction"],r["band"]) for r in left.values()})
    count=np.zeros((len(units),len(bands)));candidate_ss=np.zeros_like(count);reference_ss=np.zeros_like(count)
    for key,row in left.items():
        i=units.index(find(row["acquisition"]));j=bands.index((row["direction"],row["band"]))
        count[i,j]+=row["count"];candidate_ss[i,j]+=row["sum_squared"];reference_ss[i,j]+=right[key]["sum_squared"]
    scales=np.array([statistics[band]["std"] for _,band in bands])
    def score(weights,sums,indices,normalized=False):
        numerator=weights@sums[:,indices];denominator=weights@count[:,indices]
        with np.errstate(divide="ignore",invalid="ignore"):
            per_band=np.sqrt(numerator/denominator)
            if normalized:per_band=per_band/scales[indices]
        directions=sorted({bands[j][0] for j in indices})
        return np.mean([np.mean(per_band[:,[k for k,j in enumerate(indices) if bands[j][0]==direction]],axis=1) for direction in directions],axis=0)
    weights=np.ones((1,len(units)))
    generator=np.random.default_rng(0)
    boot=generator.multinomial(len(units),np.full(len(units),1/len(units)),size=repeats)
    comparisons={}
    for direction in ["all"]+sorted({x[0] for x in bands}):
        indices=[j for j,(d,_) in enumerate(bands) if direction=="all" or d==direction]
        trial=float(score(weights,candidate_ss,indices)[0]);baseline=float(score(weights,reference_ss,indices)[0])
        relative=1-score(boot,candidate_ss,indices)/score(boot,reference_ss,indices)
        finite=relative[np.isfinite(relative)]
        interval=np.quantile(finite,[.025,.975]).tolist() if len(finite) else [None,None]
        comparisons[direction]=dict(candidate_score=trial,reference_score=baseline,relative_rmse_improvement=1-trial/baseline,
                                    candidate_normalized_score=float(score(weights,candidate_ss,indices,True)[0]),
                                    reference_normalized_score=float(score(weights,reference_ss,indices,True)[0]),
                                    bootstrap95=interval,finite_bootstrap_replicates=len(finite),
                                    improvement_gate_passed=bool(1-trial/baseline>=.05 and interval[0] is not None and interval[0]>0))
    hard_groups={"red_edge":{"S2:B05","S2:B06","S2:B07","S2:B8A"},"SWIR":{"S2:B11","S2:B12","L8:SR_B6","L8:SR_B7"}}
    hard={}
    for name,members in hard_groups.items():
        indices=[j for j,(_,band) in enumerate(bands) if band in members]
        if indices:
            trial=float(score(weights,candidate_ss,indices)[0]);baseline=float(score(weights,reference_ss,indices)[0])
            hard[name]=dict(candidate_score=trial,reference_score=baseline,relative_degradation=trial/baseline-1)
    primary_pairs=all(result.get("scope",{}).get("pair_set") in [None,"strict"] for result in [candidate,reference])
    return dict(mask_set=mask_set,paired_records=len(left),independent_bootstrap_groups=len(units),comparisons=comparisons,
                gate_metric="sensor/direction-balanced mean per-band physical reflectance RMSE; normalized score reported separately for scheduling",
                hard_band_groups=hard,primary_pairs_for_extension_gate=primary_pairs,extension_gate_passed=bool(primary_pairs and all(x["improvement_gate_passed"] for x in comparisons.values()) and all(x["relative_degradation"]<=.02 for x in hard.values())),
                bootstrap=dict(repeats=repeats,seed=0,unit="connected original acquisitions and same-location groups",interval="percentile95"),
                boundary="single training seed; small numbers of independent groups limit geographic conclusions")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--candidate",type=Path,required=True);parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--mask-set",default="singleton");parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    result=compare(json.loads(args.candidate.read_text()),json.loads(args.reference.read_text()),args.mask_set,json.loads((RELEASE/"statistics.json").read_text())["bands"])
    args.out.parent.mkdir(exist_ok=True,parents=True);args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=="__main__":main()
