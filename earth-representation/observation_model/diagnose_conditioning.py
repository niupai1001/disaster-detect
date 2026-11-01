"""Zero-update fixed-input target-query and gradient comparison on frozen TRAIN cases."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import NativeSet,RELEASE
from .model import make_model,checkpoint_coordinate_mode
from .check_precision import tile_values


def digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--variant",choices=["transformer","cnn"],default="transformer");args=p.parse_args()
    torch.set_num_threads(1)
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    statistics=json.loads((RELEASE/"statistics.json").read_text())["bands"]
    assert state["statistics"]==statistics
    model=make_model(args.variant,coordinate_mode=checkpoint_coordinate_mode(state));model.load_state_dict(state["model"]);model.eval()
    before=digest(model);data=NativeSet("train",device="cpu");records=[]
    parameters=[v for v in model.parameters() if v.requires_grad]
    tiles=torch.linspace(0,399,16).round().long()
    for sensor,targets in [("S2",[0,3]),("L8",[1,4])]:
        for index in [0,1]:
            ids=torch.tensor([index]);hidden=torch.zeros(1,len(data.metadata[sensor]),dtype=torch.bool);hidden[:,targets]=True
            representation=model.encode(data.bands(sensor,ids,statistics,hidden))
            gradients=[];predictions=[];truths=[];valids=[];cases=[]
            for target in targets:
                query=data.query(sensor,target,ids);prediction,coverage=model.decoder(representation,query,tiles)
                truth=tile_values(data.value[sensor][query.band][ids],tiles,query.grid_gsd_m)
                valid=tile_values(data.valid[sensor][query.band][ids],tiles,query.grid_gsd_m)&coverage
                loss=F.huber_loss(prediction[valid],truth[valid],delta=.02)
                grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
                gradients.append(torch.cat([g.flatten() if g is not None else torch.zeros_like(v).flatten() for g,v in zip(grads,parameters)]))
                predictions.append(prediction.detach());truths.append(truth);valids.append(valid)
                cases.append(dict(band=query.band,loss=float(loss.detach()),target_mean=float(truth[valid].mean()),prediction_mean=float(prediction[valid].detach().mean()),bias=float((prediction[valid]-truth[valid]).mean().detach())))
            common=valids[0]&valids[1];a,b=gradients
            records.append(dict(sensor=sensor,observation=data.records[sensor][index]["observation_id"],cases=cases,
                gradient_cosine=float(F.cosine_similarity(a[None],b[None])),
                average_gradient_norm=float(((a+b)/2).norm()),individual_gradient_norms=[float(a.norm()),float(b.norm())],
                target_spectral_difference=float((truths[1]-truths[0])[common].mean()),
                predicted_query_difference=float((predictions[1]-predictions[0])[common].mean())))
    after=digest(model);assert before==after
    report=dict(status="ZERO_UPDATE_TRAIN_DIAGNOSTIC",checkpoint=str(args.checkpoint.resolve()),step=state["step"],checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                model_sha256_before=before,model_sha256_after=after,device="cpu",records=records,
                boundary="first2 frozen TRAIN observations per sensor; both target bands hidden simultaneously, identical encoder input and16 fixed target tiles; gradients compared at one checkpoint; not a causal training intervention or validation ranking; no optimizer updates")
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":main()
