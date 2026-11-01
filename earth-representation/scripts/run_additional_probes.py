"""Append completed scale or sensor-route checkpoints to the same frozen probe."""
import argparse,json
from pathlib import Path

def main():
    import torch
    torch.cuda.init()
    from scripts.frozen_segmentation_probe import data,run,ROOT
    ap=argparse.ArgumentParser();ap.add_argument("--group",choices=["scale","sensor"],required=True);args=ap.parse_args()
    out=ROOT/"outputs/verification"
    tags=["corrective_scale_"+x for x in ["uniform","random","selective"]] if args.group=="scale" else ["sensor_"+x for x in ["single","mixed","paired"]]
    rows,x,y,v,sha=data()
    existing=json.loads((out/"probe_comparison.json").read_text())
    for tag in tags:
        checkpoint=str(out/"runs"/(tag+".best_unlabeled.pt"))
        result=run(checkpoint,tag,rows,x,y,v,sha,40)
        existing=[r for r in existing if r["tag"]!=tag]+[result]
        (out/"probe_comparison.json").write_text(json.dumps(existing,indent=2))
    (out/(args.group+"_probe_status.json")).write_text(json.dumps(dict(status="completed_auxiliary_only",tags=tags),indent=2))
if __name__=="__main__":main()
