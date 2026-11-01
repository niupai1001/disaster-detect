"""Match ordinary blur/resize to low-pass/area sampling on frozen TRAIN only."""
import json
from pathlib import Path
import numpy as np
import torch
from src.data.landslide_patches import _read_patch
from src.augment import spatial_degradation as sd

def stats(x,y):
    error=(x-y).square().mean((1,2,3)) / x.square().mean((1,2,3)).clamp_min(1e-8)
    edge=(y[:,:,:,1:]-y[:,:,:,:-1]).square().mean((1,2,3)) / (x[:,:,:,1:]-x[:,:,:,:-1]).square().mean((1,2,3)).clamp_min(1e-8)
    return torch.stack([error.mean(),edge.mean()])

def main():
    out=Path("outputs/verification")
    members=json.loads((out/"runs/verified_s2_standard.provenance.json").read_text())["members"]
    chosen=members[::max(1,len(members)//16)][:16]
    x=torch.from_numpy(np.stack([_read_patch(Path(p))[0][:,:128,:128] for p in chosen])).cuda()
    rows=[]
    for factor in [1.5,2.,3.]:
        for sigma in [.6,1.,1.5]:
            target=stats(x,sd.spatial_degrade(x,sigma,factor))
            candidates=[]
            for generic_sigma in [round(float(v),2) for v in np.linspace(.1,2.5,49)]:
                score=stats(x,sd.generic_resize(sd.lowpass(x,float(generic_sigma)),factor))
                relative=(score-target).abs()/target.abs().clamp_min(1e-8)
                candidates.append((float(relative.max()),float(generic_sigma),score.cpu().tolist()))
            error,gs,score=min(candidates)
            rows.append(dict(factor=factor,physical_sigma=sigma,generic_sigma=gs,target=target.cpu().tolist(),generic=score,max_relative_error=error))
    result=dict(status="accepted" if max(r["max_relative_error"] for r in rows)<=.1 else "not_matched",tolerance=.1,metrics=["relative squared distortion","horizontal gradient energy retention"],train_members=chosen,mapping=rows)
    (out/"augmentation_calibration.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=="__main__":main()
