"""Research positive/negative controls on repaired TRAIN/VAL, no model or TEST."""
import json
from pathlib import Path

import torch

from .data import ROOT,RELEASE,NativeSet
from .paired import CROSS_RELEASE,NativePairs


def correlation(a,b,valid):
    x=a[valid].double();y=b[valid].double();x=x-x.mean();y=y-y.mean()
    denominator=x.norm()*y.norm()
    return float((x*y).sum()/denominator) if denominator>1e-10 else None


def main():
    assert RELEASE.name=="release_local_v2" and CROSS_RELEASE.name=="cross_local_v2","select the repaired releases explicitly"
    records=[]
    for split in ["train","val"]:
        data=NativeSet(split);pairs=NativePairs(data)
        names={s:[b["name"] for b in data.metadata[s]] for s in ["S2","L8"]}
        red=pairs.target["S2"][:,names["S2"].index("S2:B04")];nir=pairs.target["S2"][:,names["S2"].index("S2:B8A")]
        ls_red=pairs.target["L8"][:,names["L8"].index("L8:SR_B4")];ls_nir=pairs.target["L8"][:,names["L8"].index("L8:SR_B5")]
        for i,row in enumerate(pairs.records):
            other=next((j for j,r in enumerate(pairs.records) if r["group"]!=row["group"]),None)
            if other is None:raise ValueError("no independent geographic group for a meaningful wrong-pair control")
            for label,j in [("matched",i),("wrong_geographic_group",other)]:
                # All targets already share40x40 local30m supports. Wrong pairs deliberately ignore their geographic origin.
                valid=pairs.joint[i]&pairs.joint[other]
                a=(nir[i]-red[i])/(nir[i]+red[i]+1e-6);b=(ls_nir[j]-ls_red[j])/(ls_nir[j]+ls_red[j]+1e-6)
                records.append(dict(split=split,kind=label,s2_site=row["site_id"],landsat_site=pairs.records[j]["site_id"],valid_pixels=int(valid.sum()),
                    ndvi_correlation=correlation(a,b,valid),ndvi_mae=float((a[valid]-b[valid]).abs().mean()),
                    red_structure_correlation=correlation(red[i],ls_red[j],valid)))
        del pairs,data
    summary={}
    for split in ["train","val"]:
        summary[split]={}
        for kind in ["matched","wrong_geographic_group"]:
            selected=[r for r in records if r["split"]==split and r["kind"]==kind]
            summary[split][kind]={key:float(torch.tensor([r[key] for r in selected if r[key] is not None]).quantile(.5)) for key in ["ndvi_correlation","ndvi_mae","red_structure_correlation"]}
    result=dict(status="MEASURED_POSITIVE_NEGATIVE_CONTROLS",device="cuda",optimizer_updates=0,test_opened=False,summary=summary,records=records,
        data_release=str(RELEASE),pair_release=str(CROSS_RELEASE),control_mask="same intersection of anchor and wrong-partner joint QA for both controls",
        boundary="all admitted strict TRAIN/VAL pairs; deterministic wrong partner is first different geographic group within same split; local40x40 arrays intentionally compared despite different origins; original phases retained, not a registration test; matched NDVI was an admission filter, so its separation is not independent model evidence; red texture also reported; very few independent scenes")
    path=ROOT/"observation_model/outputs/repair_v2/diagnosis/pair_controls.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2));print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
