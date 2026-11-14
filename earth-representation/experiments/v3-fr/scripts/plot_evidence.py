"""Redraw float predictions and S5 curves without a model or Viewer."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(sys.argv[1])
figures = Path(sys.argv[2]) if len(sys.argv)>2 else root/"summary/figures"
names = json.loads((root/"experiment_display_names.json").read_bytes())
figures.mkdir(parents=True,exist_ok=True)
fit = root/"s5_capacity_fit"
plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"]})
fig,axes = plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
for arm in ("AE-L","AE-F","OR-D0","OR-D1"):
    rows = json.loads((fit/arm/"evaluation.json").read_bytes())
    for j,ax in enumerate(axes):
        id_ = next(r["id"] for r in rows if r["role"]=="fit" and r["id"].endswith("_L8")==bool(j))
        points = [(r["step"],r["mse"]) for r in rows if r["id"]==id_]
        ax.plot(*zip(*points),label=names[arm]["name"])
for ax,title in zip(axes,("S2 fixed TRAIN scene","L8 fixed TRAIN scene")):
    ax.set(title=title,xlabel="Optimizer updates",ylabel="Native MSE (z²)",yscale="log")
    ax.grid(alpha=.24)
axes[1].legend(frameon=False,fontsize=8)
fig.suptitle("V3-FR shared-path and free-latent fits · one S2 and one L8 observation per update")
fig.savefig(figures/"s5_fit.png",dpi=170)
plt.close(fig)

baseline = root/"s1_baseline"
display = json.loads((baseline/"display_audit.json").read_bytes())
with np.load(baseline/"raw_predictions.npz") as raw:
    for row in display:
        prefix = row["id"]+"_"+row["task"]+"_"
        truth = raw[prefix+"truth"]
        qa = raw[prefix+"qa"]
        target = raw[prefix+"target_mask"]
        keep = raw[prefix+"visible_mask"]
        p,m,d = (raw[prefix+k] for k in ("total_prediction","main_prediction","detail_contribution"))
        same = row["display_low"],row["display_high"]
        err = abs(p-truth)
        grad = np.full_like(p,np.nan)
        grad[:,:-1] = abs(np.diff(p-truth,axis=1))
        error_scale = max(float(np.nanquantile(err,.98)),1e-8)
        panels = [(np.where(qa,truth,np.nan),"Native truth",same,"gray"),
                  (np.where(keep,truth,np.nan),"Visible target-band values",same,"gray"),
                  (m,"Shared main prediction",same,"gray"),(p,"Total prediction",same,"gray"),
                  (target,"Target selection",(0,1),"gray"),(err,"Absolute error",(0,error_scale),"magma"),
                  (grad,"Signed gradient error magnitude",(0,error_scale),"magma")]
        detail_scale = max(float(np.nanquantile(abs(d),.98)),1e-8)
        panels.append((d,"Legacy detail contribution",(-detail_scale,detail_scale),"coolwarm"))
        if prefix+"visible_reference" in raw and row["task"]=="B1":
            ref = raw[prefix+"visible_reference"]
            panels[1] = (ref,"Visible reference "+row["reference_band"],np.nanquantile(ref,[.02,.98]),"gray")
        fig,axes = plt.subplots(2,4,figsize=(12,7),layout="constrained")
        for ax,(image,title,scale,cmap) in zip(axes.flat,panels,strict=True):
            ax.imshow(image,cmap=cmap,vmin=scale[0],vmax=scale[1],interpolation="none")
            ax.set_title(title,fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{names['S1-REPLAY']['name']} · {row['id']} · {row['task']} · {row['band']}\n"
                     f"Original prediction below/above fixed display range: {row['clipped_low']:.1%} / "
                     f"{row['clipped_high']:.1%}    Target variance ratio: {row['variance_ratio']:.3g}")
        fig.savefig(figures/(prefix+"historical.png"),dpi=135)
        plt.close(fig)
