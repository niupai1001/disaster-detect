"""One real-data FP32/BF16 loss and gradient comparison; no tolerance sweep."""
import contextlib
import json
import os
import time
from pathlib import Path
import torch
from torch.nn import functional as F

if __name__=="__main__":torch.cuda.init()

from .data import NativeSet,RELEASE
from .model import make_model


def tile_values(value,indices,gsd):
    b,h,w=value.shape;p=60//gsd;n=h//p
    return value.reshape(b,n,p,n,p).permute(0,1,3,2,4).reshape(b,n*n,p*p)[:,indices]


def main():
    stats=json.loads((RELEASE/"statistics.json").read_text())["bands"];data=NativeSet("train")
    results={};gradients={}
    for precision in ["float32","bfloat16"]:
        model=make_model(coordinate_mode=os.environ.get("OBSERVATION_COORDINATE_MODE","legacy_local")).cuda();model.train();torch.cuda.synchronize();start=time.perf_counter();loss_total=0.
        for sensor,target in [("S2",8),("L8",3)]:
            c=10 if sensor=="S2" else 7;ids=torch.tensor([0,1],device="cuda")
            hidden=torch.zeros(2,c,dtype=torch.bool,device="cuda");hidden[:,target]=True;hidden[:,(target+1)%c]=True
            bands=data.bands(sensor,ids,stats,hidden);query=data.query(sensor,target,ids);tiles=torch.arange(16,device="cuda")
            manager=torch.autocast("cuda",dtype=torch.bfloat16) if precision=="bfloat16" else contextlib.nullcontext()
            with manager:prediction,coverage=model.decoder(model.encode(bands),query,tiles)
            name=query.band;truth=tile_values(data.value[sensor][name][ids],tiles,query.grid_gsd_m)
            valid=tile_values(data.valid[sensor][name][ids],tiles,query.grid_gsd_m)&coverage
            loss=F.huber_loss(prediction.float()[valid],truth[valid],delta=.02)/2
            loss.backward();loss_total+=float(loss.detach())
        torch.cuda.synchronize()
        gradients[precision]=torch.cat([p.grad.flatten().float() for p in model.parameters() if p.grad is not None])
        results[precision]=dict(loss=loss_total,seconds=time.perf_counter()-start,finite=bool(torch.isfinite(gradients[precision]).all()))
        del model
    a,b=gradients["float32"],gradients["bfloat16"]
    relative_loss=abs(results["float32"]["loss"]-results["bfloat16"]["loss"])/max(abs(results["float32"]["loss"]),1e-9)
    relative_gradient=float((a-b).norm()/a.norm().clamp_min(1e-9));cosine=float(F.cosine_similarity(a[None],b[None]))
    passed=relative_loss<=.02 and relative_gradient<=.05 and cosine>=.99 and all(r["finite"] for r in results.values())
    result=dict(status="passed" if passed else "not_adopted",results=results,relative_loss_error=relative_loss,relative_gradient_error=relative_gradient,gradient_cosine=cosine,thresholds=dict(loss=.02,gradient=.05,cosine=.99),adopted_precision="bfloat16_autocast_fp32_parameters_and_loss" if passed else "float32",scope="one actual native S2 and Landsat combined comparison; engineering only; zero optimizer updates")
    path=Path(os.environ.get("OBSERVATION_CHECK_DIR",Path(__file__).parent/"outputs/phase1"))/"precision_check.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=="__main__":main()
