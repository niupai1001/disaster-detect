"""One real full-footprint update measures native GPU feasibility."""
import json
import time
from pathlib import Path
import torch

if __name__=="__main__":torch.cuda.init()

from .data import NativeSet,RELEASE
from .model import make_model


def main():
    stats=json.loads((RELEASE/"statistics.json").read_text())["bands"]
    data=NativeSet("train")
    model=make_model().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4)
    ids=torch.tensor([0,1],device="cuda");hidden=torch.zeros(2,10,device="cuda",dtype=torch.bool);hidden[:,0]=True
    bands=data.bands("S2",ids,stats,hidden);query=data.query("S2",0,ids)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    representation=model.encode(bands)
    pred,valid=model.decoder(representation,query,torch.arange(16,device="cuda"))
    loss=pred.square().mean();loss.backward();optimizer.step();torch.cuda.synchronize()
    result=dict(kind="engineering_real_native_update_not_formal",batch=2,footprint_m=1200,source_bands=10,encoded_tokens_per_observation=16000,target_tiles=16,loss=float(loss.detach()),seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
    path=Path(__file__).parent/"outputs/phase1/native_profile.json";path.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=="__main__":main()
