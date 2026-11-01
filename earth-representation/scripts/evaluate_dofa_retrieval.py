"""Read-only official DOFA reference on the identical frozen regional gallery."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.frozen_segmentation_probe import encoder
from scripts.cross_sensor_eval import retrieval_stats
from src.data.triples import load_triples


def main():
    out=Path("outputs/verification")
    frozen=json.loads((out/"existing_retrieval.json").read_text())
    items=load_triples(min_valid=.8,allowed_ids=set(frozen["sample_ids"]))
    model,_,info=encoder("dofa")
    features={};seconds={}
    with torch.no_grad():
        for key in ["s2","landsat","hls"]:
            torch.cuda.synchronize();start=time.perf_counter();rows=[]
            waves=(items[0][key]["centre"]/1000).tolist()
            for i in range(0,len(items),16):
                x=torch.from_numpy(np.stack([it[key]["image"] for it in items[i:i+16]])).cuda()
                x=F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False)
                result=model(x,waves)[-1]
                rows.append(F.normalize(result.mean((2,3)),dim=1))
            features[key]=torch.cat(rows)
            torch.cuda.synchronize();seconds[key]=time.perf_counter()-start
    pairs={}
    for a,b in [("s2","landsat"),("s2","hls"),("landsat","hls")]:
        pairs[a+"->"+b]=retrieval_stats(features[a],features[b]);pairs[b+"->"+a]=retrieval_stats(features[b],features[a])
    result=dict(status="external_regional_reference_only",official=info,n=len(items),sample_ids=[x["sample_id"] for x in items],
        input="surface reflectance, resize same footprint to224, nominal wavelengths in micrometers",pooling="last dense level spatial mean and L2 normalization",seconds=seconds,pairs=pairs,
        cost_boundary="shared GPU, includes transfer and resize; external pretraining amount and architecture differ; not an equal-compute causal comparison")
    (out/"dofa_retrieval.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=="__main__":main()
