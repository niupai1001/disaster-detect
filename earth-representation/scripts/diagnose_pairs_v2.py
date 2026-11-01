"""Near-time pair controls on common 30m support, not a same-acquisition claim."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.triples import load_triples

def corr(a,b,mask):
    x=a[mask];y=b[mask]
    if len(x)<30 or x.std()<1e-8 or y.std()<1e-8:return None
    return float(np.corrcoef(x,y)[0,1])

def main():
    items=load_triples(min_valid=.8)
    # Fixed first 40 items; separate from data inclusion or model selection.
    items=items[:40]
    fields=[]
    for it in items:
        f={}
        for key,red,nir in [("s2",2,7),("landsat",3,4),("hls",2,7)]:
            x=torch.from_numpy(it[key]["image"]).cuda()[None]
            v=torch.from_numpy(it[key]["valid_mask"].astype("float32")).cuda()[None,None]
            a=F.avg_pool2d(x*v,3,3)/F.avg_pool2d(v,3,3).clamp_min(1e-8)
            nd=(a[:,nir]-a[:,red])/(a[:,nir]+a[:,red]).clamp_min(1e-6)
            f[key]=(nd[0].cpu().numpy(),(F.avg_pool2d(v,3,3)[0,0]>=.999).cpu().numpy())
        fields.append(f)
    rows=[]
    for i,(it,f) in enumerate(zip(items,fields)):
        a,va=f["s2"];b,vb=f["landsat"]
        wrong,vw=fields[(i+17)%len(items)]["landsat"]
        rows.append(dict(sample=it["sample_id"],same_place=corr(a,b,va&vb),shift1=corr(a,np.roll(b,1,1),va&np.roll(vb,1,1)),shift3=corr(a,np.roll(b,3,1),va&np.roll(vb,3,1)),transpose=corr(a,b.T,va&vb.T),wrong_place=corr(a,wrong,va&vw)))
    out=Path("outputs/verification")
    (out/"pair_controls.json").write_text(json.dumps(dict(status="near_time_diagnostic_only",same_acquisition_provenance="unverified",support="3x3 area aggregation on common 10m grid; 42x42 at 30m",rows=rows),indent=2))
    fig,ax=plt.subplots(figsize=(8,4));keys=["same_place","shift1","shift3","transpose","wrong_place"]
    ax.boxplot([[r[k] for r in rows if r[k] is not None] for k in keys],tick_labels=keys);ax.set_ylabel("NDVI correlation on jointly valid pixels");fig.tight_layout();fig.savefig(out/"pair_controls.png",dpi=140)
    print(json.dumps({k:float(np.median([r[k] for r in rows if r[k] is not None])) for k in keys}))
if __name__=="__main__":main()
