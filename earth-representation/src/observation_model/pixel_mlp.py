"""A small conditional per-pixel baseline, with no spatial encoder or hidden input."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __name__=="__main__":torch.cuda.init()

from .data import ROOT,RELEASE,NativeSet,BalancedSampler,MaskSampler
from .paired import NativePairs
from .baselines import features,OUT
from .train_masked import evaluation
from .train_cross import evaluate_cross,cross_query
from src.ssl.tvlp_writer import TvlpRun


class PixelMLP(nn.Module):
    def __init__(self,statistics):
        super().__init__();self.statistics=statistics;self.names=sorted(statistics)
        self.network=nn.Sequential(nn.Linear(2*len(self.names)+4,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,1))
    def encode(self,bands):return bands
    def input_features(self,bands,query):
        value,valid=features(bands,query);shape=value.shape[:-1]
        x=torch.zeros(*shape,len(self.names),device=value.device);mask=torch.zeros_like(x)
        for i,band in enumerate(bands):
            index=self.names.index(band.name);available=valid[...,i] if band.name!=query.band else torch.zeros_like(valid[...,i])
            x[...,index]=torch.where(available,value[...,i],0);mask[...,index]=available.float()
        condition=torch.tensor([query.wavelength_nm/2500.,query.native_gsd_m/60.,query.grid_gsd_m/60.,float(query.sensor=="S2")],device=value.device)
        return torch.cat([x,mask,condition.expand(*shape,4)],-1)
    def decoder(self,bands,query):
        x=self.input_features(bands,query);flat=x.reshape(-1,x.shape[-1])
        result=torch.cat([self.network(part) for part in flat.split(65536)]).reshape(x.shape[:-1])
        return result,torch.ones_like(result,dtype=torch.bool)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["masked","cross"],default="masked");args=parser.parse_args()
    torch.manual_seed(0);statistics=json.loads((RELEASE/"statistics.json").read_text())["bands"]
    output=OUT/(args.mode+"_mlp");output.mkdir(exist_ok=True,parents=True)
    if (output/"training_result.json").exists():print("completed MLP reused");return
    model=PixelMLP(statistics).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=14,threshold=.001,threshold_mode="rel",min_lr=1e-6)
    train=NativeSet("train");validation=NativeSet("val");pairset=NativePairs(train) if args.mode=="cross" else None;val_pairs=NativePairs(validation) if pairset else None
    lengths={s:len(v) for s,v in train.records.items()} if pairset is None else {"pairs":len(pairset.records)}
    sampler=BalancedSampler(lengths);masker=MaskSampler(json.loads((RELEASE/"masking.json").read_text()));rng=np.random.default_rng(0);pixels=torch.Generator(device="cuda").manual_seed(0)
    step=0;best=float("inf");history=[]
    last=output/"last.pt"
    if last.exists():
        state=torch.load(last,map_location="cuda",weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);sampler.load_state_dict(state["sampler"])
        rng.bit_generator.state=state["rng"];masker.generator.bit_generator.state=state["mask_rng"];pixels.set_state(state["pixel_rng"].cpu());step=state["step"];best=state["best"];history=state["history"]
    tv=TvlpRun(Path(os.environ.get("OBSERVATION_TV_ROOT",ROOT/"observation_model/outputs/tvlp")),project="conditional-observation",name="pixel_mlp_"+args.mode,run_id=os.environ.get("OBSERVATION_RUN_PREFIX","obs_")+"mlp_"+args.mode,resume=last.exists(),seed=0,config=dict(mode=args.mode,release=str(RELEASE),parameters=sum(p.numel() for p in model.parameters()),max_steps=10000,pixel_batch=8192,native_observations_per_update=8,statistics="TRAIN-only",masking="target never included; geometry-aware source sampling",selection="same fixed VAL score as corresponding neural experiment",optimizer="AdamW lr.001 WD.0001, independent plateau schedule"),metric_definitions={"train/loss":{"direction":"min"},"val/score":{"direction":"min"}})
    start=time.perf_counter()
    def save(path):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),sampler=sampler.state_dict(),rng=rng.bit_generator.state,mask_rng=masker.generator.bit_generator.state,pixel_rng=pixels.get_state(),step=step,best=best,history=history,statistics=statistics,mode=args.mode)
        temporary=path.with_suffix(".tmp");torch.save(state,temporary);temporary.replace(path)
    try:
        while step<10000:
            model.train();optimizer.zero_grad(set_to_none=True);total=0.
            for source in ["S2","L8"]:
                target=source if args.mode=="masked" else "L8" if source=="S2" else "S2"
                j=int(rng.integers(len(train.metadata[target])))
                if pairset is None:
                    ids=torch.tensor(sampler.take(source,4),device="cuda");hidden=masker.sample(source,j,4,"cuda");query=train.query(source,j,ids)
                    bands=train.bands(source,ids,statistics,hidden);truth=train.value[source][query.band][ids];valid=train.valid[source][query.band][ids]
                else:
                    positions=torch.tensor(sampler.take("pairs",4),device="cuda");ids=pairset.indices[source][positions];query=cross_query(train,target,j,pairset.phase[positions]);bands=train.bands(source,ids,statistics)
                    truth=pairset.target[target][positions,j];valid=pairset.valid[target][positions,j]
                x=model.input_features(bands,query);xs=[];ys=[]
                for i in range(len(ids)):
                    candidates=valid[i].flatten().nonzero().flatten()
                    if not len(candidates):raise ValueError("empty real target mask")
                    positions=candidates[torch.randint(len(candidates),(1024,),device="cuda",generator=pixels)]
                    xs.append(x[i].reshape(-1,x.shape[-1])[positions]);ys.append(truth[i].flatten()[positions])
                prediction=model.network(torch.cat(xs)).squeeze(-1);loss=F.huber_loss(prediction,torch.cat(ys),delta=.02)/2
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite pixel regression")
                loss.backward();total+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True);optimizer.step();step+=1
            tv.log("train",dict(loss=total,lr=optimizer.param_groups[0]["lr"],pixel_exposure=8192*step,observation_exposure=8*step),global_step=step);tv.heartbeat(step)
            if step%100==0:print(json.dumps(dict(mode=args.mode,step=step,loss=total,seconds=time.perf_counter()-start)),flush=True)
            if step%250==0:
                result=evaluation(model,validation,statistics,"float32") if pairset is None else evaluate_cross(model,val_pairs,statistics,"float32")
                score=result["score"];improved=score<best;best=min(best,score);scheduler.step(score);history.append(dict(step=step,score=score))
                (output/f"validation_{step}.json").write_text(json.dumps(result,indent=2));tv.log("val",dict(score=score),global_step=step);save(last)
                if improved:save(output/"best.pt")
        save(last);tv.finish();(output/"training_result.json").write_text(json.dumps(dict(status="completed_pending_TEST",step=step,best_score=best,last_score=history[-1]["score"],seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),mode=args.mode,stop_reason="fixed_baseline_cap_not_confirmed_convergence"),indent=2))
    except BaseException as exc:tv.finish("failed",str(exc));raise


if __name__=="__main__":main()
