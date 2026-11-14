"""Fixed-panel phase and visualization checks on float predictions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def phase_residual(prediction,truth,mask,affine,origin,step,phase):
    gx,_,x0,_,ey,y0 = affine
    counts, sums = np.zeros(2,dtype=np.int64),np.zeros(2)
    for axis,start,spacing,base in ((0,y0,ey,origin[1]),(1,x0,gx,origin[0])):
        positions = start+(np.arange(mask.shape[axis])+.5)*spacing
        edges = np.diff(np.floor((positions-base)/step-phase))!=0
        if axis==0:
            selected = mask[1:]&mask[:-1]
            boundary = edges[:,None]
        else:
            selected = mask[:,1:]&mask[:,:-1]
            boundary = edges[None,:]
        residual = abs(np.diff(prediction-truth,axis=axis))
        for i,flag in enumerate((~boundary,boundary)):
            x = residual[selected&flag]
            sums[i] += x.sum(dtype=np.float64)
            counts[i] += len(x)
    means = [float(s/n) if n else None for s,n in zip(sums,counts)]
    return dict(interior_pairs=int(counts[0]),boundary_pairs=int(counts[1]),interior_mae=means[0],
                boundary_mae=means[1],ratio=means[1]/means[0] if all(m is not None for m in means) and means[0]>0 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign",type=Path)
    args = parser.parse_args()
    campaign = args.campaign
    stage = campaign/"s6_component_trials"
    figure_root = campaign/"summary/figures"
    scene_geometry = json.loads((campaign/"fixed_scene_geometry.json").read_bytes())
    audits = []
    with np.load(stage/"B0/val_233/raw_predictions.npz") as base:
        for arm in ("BP","N0","N1","N2","C"):
            parent = dict(BP="B0",N0="BP",N1="N0",N2="N1",C="N2")[arm]
            path = stage/arm/"val_233/raw_predictions.npz"
            if not path.exists():
                continue
            with np.load(path) as current,np.load(stage/parent/"val_233/raw_predictions.npz") as previous:
                for task in ("A","B1"):
                    groups = {}
                    for field in sorted(k for k in base.files if k.endswith("_target_mask") and f"_{task}_" in k):
                        prefix = field.removesuffix("target_mask")
                        observation = prefix.split(f"_{task}_")[0]
                        groups.setdefault(observation,[]).append(prefix)
                    for observation,group in groups.items():
                        for prefix in group:
                            truth = base[prefix+"truth"]
                            target = base[prefix+"target_mask"]
                            affine = base[prefix+"affine"]
                            assert np.array_equal(target,current[prefix+"target_mask"])
                            origin = scene_geometry[observation]["origin"]
                            for name,values in ((parent,previous),(arm,current)):
                                p = values[prefix+"main_prediction"]
                                for step in (40,80,160,320):
                                    for phase in (0.,.5):
                                        audits.append(dict(candidate=arm,arm=name,id=observation,task=task,band=prefix,
                                            metres=step,phase=phase,origin=list(origin),
                                            origin_scope="complete frozen observation",
                                            **phase_residual(p,truth,target,affine,origin,step,phase)))
                        prefix = max(group,key=lambda g:int(base[g+"target_mask"].sum()))
                        truth,mask,qa,keep = [base[prefix+k] for k in ("truth","target_mask","qa","visible_mask")]
                        limit = base[prefix+"display_scale"]
                        m1,m2 = (x[prefix+"main_prediction"] for x in (previous,current))
                        p1,p2 = (x[prefix+"total_prediction"] for x in (previous,current))
                        error = np.concatenate([abs(p[mask]-truth[mask]) for p in (p1,p2)])
                        err_scale = (0,max(float(np.quantile(error,.98)),1e-8))
                        shown = np.where(keep,truth,np.nan)
                        shown_label,shown_limit = "Visible input",limit
                        if task=="B1":
                            refs = sorted(k for k in base.files if k.startswith(f"{observation}_B1_visible_reference_"))
                            if refs:
                                shown = base[refs[0]]
                                shown_limit = np.nanquantile(shown,[.02,.98])
                                shown_label = "Visible "+refs[0].split("reference_")[1]
                        figure,axes = plt.subplots(2,5,figsize=(12,5.5),layout="constrained")
                        panels = [(np.where(qa,truth,np.nan),"Truth",limit),(shown,shown_label,shown_limit),
                                  (mask,"Target mask",(0,1)),(m1,f"{parent} main",limit),(m2,f"{arm} main",limit),
                                  (p1,f"{parent} total",limit),(p2,f"{arm} total",limit),
                                  (abs(p1-truth),f"{parent} error",err_scale),(abs(p2-truth),f"{arm} error",err_scale),
                                  (abs(np.diff(p2-truth,axis=1)),"Candidate gradient error",err_scale)]
                        for i,(ax,(arr,name,scale)) in enumerate(zip(axes.flat,panels,strict=True)):
                            ax.imshow(arr,vmin=scale[0],vmax=scale[1],cmap="gray" if i<7 else "magma",
                                      interpolation="none")
                            ax.set_title(name,fontsize=9)
                            ax.axis("off")
                        figure.suptitle(f"V3-FR · Step 233 · {prefix.removesuffix('_')}")
                        figure.savefig(figure_root/f"s6_{arm}_{observation}_{task}.png",dpi=135)
                        plt.close(figure)
    (stage/"fixed_panel_phase_audit.json").write_text(json.dumps(audits,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
