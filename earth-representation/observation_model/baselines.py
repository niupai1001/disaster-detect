"""Strong local spectral controls with the same physical target grids and masks."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

if __name__=="__main__":torch.cuda.init()

from .data import ROOT,RELEASE,NativeSet
from .operators import resample_band
from .paired import NativePairs
from .train_masked import evaluation
from .train_cross import evaluate_cross

OUT=Path(os.environ.get("OBSERVATION_BASELINES",ROOT/"observation_model/outputs/baselines"))


def features(bands,query):
    values=[];available=[]
    for band in bands:
        x,coverage=resample_band(band.value,band.valid,band.native_gsd_m,band.phase_xy_m,query.grid_gsd_m,query.phase_xy_m)
        values.append(x);available.append(coverage>=.8)
    return torch.stack(values,-1),torch.stack(available,-1)


class SpectralControl:
    def __init__(self,statistics,kind="mean",fits=None,lambdas=None,mode="masked"):
        self.statistics=statistics;self.kind=kind;self.fits=fits or {};self.lambdas=lambdas or {};self.mode=mode;self.cache={}
    def eval(self):return self
    def train(self):return self
    def encode(self,bands):return bands
    def key(self,bands,query):return self.mode+":"+bands[0].name.split(":")[0]+":"+query.band
    def decoder(self,bands,query):
        x,valid=features(bands,query);shape=x.shape[:-1];c=x.shape[-1]
        # Never condition a same-sensor estimate on its own observed values.
        for index,band in enumerate(bands):
            if band.name==query.band:valid[...,index]=False
        target=self.statistics[query.band];mean=target["mean"];std=target["std"]
        if self.kind=="mean":return torch.full(shape,mean,device=x.device),torch.ones(shape,device=x.device,dtype=torch.bool)
        if self.kind=="interpolation":
            raw=torch.stack([x[...,i]*self.statistics[b.name]["std"]+self.statistics[b.name]["mean"] for i,b in enumerate(bands)],-1)
            waves=torch.tensor([b.wavelength_nm for b in bands],device=x.device);target_wave=query.wavelength_nm
            lower=waves.clone();lower[lower>target_wave]=-float("inf")
            upper=waves.clone();upper[upper<target_wave]=float("inf")
            lo=torch.where(valid,lower,-float("inf")).argmax(-1);hi=torch.where(valid,upper,float("inf")).argmin(-1)
            has_lo=(valid&(waves<=target_wave)).any(-1);has_hi=(valid&(waves>=target_wave)).any(-1)
            nearest=torch.where(valid,(waves-target_wave).abs(),float("inf")).argmin(-1)
            lo=torch.where(has_lo,lo,nearest);hi=torch.where(has_hi,hi,nearest)
            low=raw.gather(-1,lo[...,None]).squeeze(-1);high=raw.gather(-1,hi[...,None]).squeeze(-1)
            weight=torch.where(lo==hi,torch.zeros_like(low),(target_wave-waves[lo])/(waves[hi]-waves[lo]).clamp_min(1e-6))
            prediction=low+(high-low)*weight
            return torch.where(valid.any(-1),prediction,mean),torch.ones(shape,device=x.device,dtype=torch.bool)
        key=self.key(bands,query);fit=self.fits[key];flat=x.reshape(-1,c).double();available=valid.reshape(-1,c)
        bits=(available.long()*(2**torch.arange(c,device=x.device))).sum(1)
        result=torch.empty(len(flat),device=x.device,dtype=torch.float64)
        regularization=self.lambdas[key] if isinstance(self.lambdas,dict) else self.lambdas
        for mask in bits.unique().tolist():
            selected=[i for i in range(c) if mask&(1<<i)];subset=bits==mask
            if not selected:result[subset]=fit["mean"][-1];continue
            cache_key=(key,mask,float(regularization))
            if cache_key not in self.cache:
                idx=torch.tensor(selected,device=x.device)
                covariance=fit["covariance"][idx[:,None],idx[None,:]]
                coefficient=torch.linalg.solve(covariance+float(regularization)*torch.eye(len(idx),device=x.device,dtype=torch.float64),fit["covariance"][idx,-1])
                self.cache[cache_key]=(idx,coefficient)
            idx,coefficient=self.cache[cache_key]
            result[subset]=(flat[subset][:,idx]-fit["mean"][idx])@coefficient+fit["mean"][-1]
        return (result.reshape(shape)*std+mean).float(),torch.ones(shape,device=x.device,dtype=torch.bool)


def fit_ridge(data,statistics,mode):
    generator=torch.Generator(device=data.device).manual_seed(0);fits={}
    pairset=NativePairs(data) if mode=="cross" else None
    for sensor in ["S2","L8"]:
        targets=data.metadata[sensor] if mode=="masked" else data.metadata["L8" if sensor=="S2" else "S2"]
        c=len(data.metadata[sensor]);target_sensor=sensor if mode=="masked" else "L8" if sensor=="S2" else "S2"
        for j,band in enumerate(targets):
            n=0;total=torch.zeros(c+1,device=data.device,dtype=torch.float64);outer=torch.zeros(c+1,c+1,device=data.device,dtype=torch.float64)
            limit=len(data.records[sensor]) if mode=="masked" else len(pairset.records)
            for start in range(0,limit,8):
                positions=torch.arange(start,min(start+8,limit),device=data.device)
                ids=positions if mode=="masked" else pairset.indices[sensor][positions]
                bands=data.bands(sensor,ids,statistics)
                if mode=="masked":
                    query=data.query(sensor,j,ids);truth=data.value[sensor][query.band][ids];truth_valid=data.valid[sensor][query.band][ids]
                else:
                    from .train_cross import cross_query
                    query=cross_query(data,target_sensor,j,pairset.phase[positions]);truth=pairset.target[target_sensor][positions,j];truth_valid=pairset.valid[target_sensor][positions,j]
                x,valid=features(bands,query);selected=[]
                y=(truth-statistics[query.band]["mean"])/statistics[query.band]["std"]
                for index in range(len(ids)):
                    mask=valid[index].all(-1)&truth_valid[index]
                    candidates=mask.flatten().nonzero().flatten()
                    take=candidates[torch.randperm(len(candidates),device=data.device,generator=generator)[:512]]
                    if len(take):selected.append(torch.cat([x[index].reshape(-1,c)[take],y[index].reshape(-1,1)[take]],1).double())
                if not selected:continue
                values=torch.cat(selected);n+=len(values);total+=values.sum(0);outer+=values.T@values
            if n<32:raise ValueError("insufficient legitimate TRAIN spectral regression pixels")
            mean=total/n;covariance=outer/n-mean[:,None]*mean[None,:]
            key=mode+":"+sensor+":"+band["name"];fits[key]=dict(mean=mean,covariance=covariance,count=n,target=band["name"],source=sensor)
            print(json.dumps(dict(key=key,train_pixels=n)),flush=True)
    return fits


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["masked","cross"],default="masked");args=parser.parse_args()
    statistics=json.loads((RELEASE/"statistics.json").read_text())["bands"];OUT.mkdir(exist_ok=True)
    file=OUT/(args.mode+"_ridge_fits.pt")
    if file.exists():fits=torch.load(file,map_location="cuda",weights_only=False)
    else:
        fits=fit_ridge(NativeSet("train"),statistics,args.mode);torch.save(fits,file)
    validation=NativeSet("val");pairset=NativePairs(validation) if args.mode=="cross" else None
    def evaluate(model):return evaluation(model,validation,statistics,"float32") if args.mode=="masked" else evaluate_cross(model,pairset,statistics,"float32")
    results={}
    for kind in ["mean","interpolation"]:
        results[kind]=evaluate(SpectralControl(statistics,kind,mode=args.mode))
    trials={}
    for regularization in [.0001,.01,1.]:
        result=evaluate(SpectralControl(statistics,"ridge",fits,regularization,mode=args.mode));trials[str(regularization)]=result
    lambdas={}
    for key,fit in fits.items():
        source=fit["source"];name=fit["target"]
        if args.mode=="masked":score=lambda r:r["bands"][name]["rmse"]
        else:
            direction=source+"_to_"+("L8" if source=="S2" else "S2");score=lambda r:r["directions"][direction][name]["rmse"]
        lambdas[key]=float(min(trials,key=lambda k:score(trials[k])))
    results["ridge"]=evaluate(SpectralControl(statistics,"ridge",fits,lambdas,mode=args.mode))
    (OUT/(args.mode+"_ridge_lambdas.json")).write_text(json.dumps(lambdas,indent=2))
    (OUT/(args.mode+"_validation.json")).write_text(json.dumps(dict(mode=args.mode,results=results,lambda_trials=trials,selection="three fixed regularizers chosen per band using VAL; no TEST used",boundary="TRAIN covariance fitted from native-grid spectral samples; inference conditions only on visible bands; no hidden target pixel passed as input"),indent=2))
    print(json.dumps({k:v["score"] for k,v in results.items()}))


if __name__=="__main__":main()
