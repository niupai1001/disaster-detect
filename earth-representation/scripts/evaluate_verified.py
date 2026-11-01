import json
from pathlib import Path
import numpy as np
import torch
from scripts.cross_sensor_eval import build,features,retrieval_stats
from src.data.triples import load_triples

def main():
 out=Path("outputs/verification")
 frozen=json.loads((out/"existing_retrieval.json").read_text())
 items=load_triples(min_valid=.8,allowed_ids=set(frozen["sample_ids"]))
 results=[]
 for suffix in [".best_unlabeled",""]:
  p=out/"runs"/("verified_s2_standard"+suffix+".pt")
  state=torch.load(p,map_location="cuda",weights_only=False)
  model=build().cuda().eval();model.load_state_dict(state["teacher"])
  fs={k:features(model,items,k,"cuda",batch=32) for k in ["s2","landsat","hls"]}
  singular=torch.linalg.svdvals(fs["s2"]-fs["s2"].mean(0));prob=singular/singular.sum().clamp_min(1e-12)
  rank=float(torch.exp(-(prob*prob.clamp_min(1e-12).log()).sum()))
  row=dict(checkpoint=str(p),step=state["step"],epoch=state["epoch"],n=len(items),centered_feature_effective_rank=rank,mean_feature_std=float(fs["s2"].std(0).mean()),pairs={})
  for a,b in [("s2","landsat"),("s2","hls"),("landsat","hls")]:
   row["pairs"][a+"->"+b]=retrieval_stats(fs[a],fs[b])
   row["pairs"][b+"->"+a]=retrieval_stats(fs[b],fs[a])
  results.append(row);print(json.dumps(row),flush=True)
  del model,state,fs
 (out/"verified_baseline_evaluation.json").write_text(json.dumps(dict(status="regional_diagnostic_only",results=results),indent=2))
if __name__=="__main__":main()
