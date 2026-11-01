"""Frozen checkpoints on corrected data; no optimizer updates."""
import json,os
from pathlib import Path
import numpy as np
import torch
from scripts.cross_sensor_eval import build,features,retrieval_stats
from src.data.triples import load_triples

def main():
    torch.manual_seed(0)
    items=load_triples(min_valid=.8)
    if len(items)<20:raise ValueError("insufficient common QA-valid triples")
    device="cuda"
    results=[]
    paths=[None,Path("outputs/dino_official/off3k.pt"),Path("outputs/dino_official/mixed2k.pt")]
    paths += [Path(f"outputs/e1_official/E1-{i}.pt") for i in [0,1,2]]
    paths += [Path(f"outputs/e2_official/E2-{i}.pt") for i in [0,3,4]]
    for path in paths:
        model=build().to(device).eval()
        if path:model.load_state_dict(torch.load(path,map_location=device,weights_only=False)["teacher"])
        fs={k:features(model,items,k,device,batch=32) for k in ["s2","landsat","hls"]}
        row={"checkpoint":str(path) if path else "random_seed0","n":len(items),"pairs":{}}
        for a,b in [("s2","landsat"),("s2","hls"),("landsat","hls")]:
            for x,y in [(a,b),(b,a)]:row["pairs"][x+"->"+y]=retrieval_stats(fs[x],fs[y])
        results.append(row)
        print(json.dumps(row),flush=True)
        Path("outputs/verification/existing_retrieval.json").write_text(json.dumps({"status":"DIAGNOSTIC_ONLY","same_acquisition_provenance":"not_verified","regions":sorted(set(x["region"] for x in items)),"sample_ids":[x["sample_id"] for x in items],"results":results},indent=2))
        del model,fs

if __name__=="__main__":main()
