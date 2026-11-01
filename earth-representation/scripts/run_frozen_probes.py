"""Frozen semantic auxiliary diagnostics, one head seed and shared data."""
import json
from pathlib import Path
from scripts.frozen_segmentation_probe import data,run,ROOT

def main():
    rows,x,y,v,sha=data()
    jobs=[("random_frozen","random"),("baseline_verified","outputs/verification/runs/verified_s2_standard.best_unlabeled.pt"),
        ("ordinary_verified","outputs/verification/runs/verified_s2_generic.best_unlabeled.pt"),
        ("physical_verified","outputs/verification/runs/verified_s2_physical.best_unlabeled.pt"),
        ("legacy_off3k","outputs/dino_official/off3k.pt"),
        ("legacy_mixed2k","outputs/dino_official/mixed2k.pt")]
    jobs += [("legacy_e1_"+str(i),f"outputs/e1_official/E1-{i}.pt") for i in [0,1,2]]
    jobs += [("legacy_e2_"+str(i),f"outputs/e2_official/E2-{i}.pt") for i in [0,3,4]]
    jobs += [("dofa_external","dofa")]
    results=[]
    status=ROOT/"outputs/verification/probe_queue_status.json"
    for tag,checkpoint in jobs:
        status.write_text(json.dumps(dict(status="running",current=tag,completed=[r["tag"] for r in results]),indent=2))
        try:results.append(run(checkpoint,tag,rows,x,y,v,sha,40))
        except Exception as exc:
            status.write_text(json.dumps(dict(status="failed",current=tag,error=str(exc),completed=[r["tag"] for r in results]),indent=2));raise
    status.write_text(json.dumps(dict(status="completed_auxiliary_only",completed=[r["tag"] for r in results]),indent=2))
    (ROOT/"outputs/verification/probe_comparison.json").write_text(json.dumps(results,indent=2))
if __name__=="__main__":main()
