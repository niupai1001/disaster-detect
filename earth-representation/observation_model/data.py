"""Frozen native arrays, TRAIN-only scalar statistics and balanced observation sampling."""
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch

from .model import BandRaster,TargetQuery

ROOT=Path(os.environ.get("OBSERVATION_PROJECT_ROOT",Path(__file__).resolve().parents[1]))
RELEASE=Path(os.environ.get("OBSERVATION_RELEASE",ROOT/"observation_model/outputs/phase0/release_local_v1"))


def rows(split=None):
    result=[json.loads(s) for s in (RELEASE/"single_masked.jsonl").read_text().splitlines()]
    return [r for r in result if split is None or r["split"]==split]


def fit_statistics():
    stats={}
    for row in rows("train"):
        with np.load(row["cache"]) as z:
            for band in row["bands"]:
                values=z[band["name"]][z[band["name"]+"_valid"]].astype(np.float64)
                s=stats.setdefault(band["name"],dict(count=0,sum=0.,sum_sq=0.))
                s["count"]+=len(values);s["sum"]+=float(values.sum());s["sum_sq"]+=float((values*values).sum())
    for s in stats.values():
        s["mean"]=s["sum"]/s["count"];s["std"]=max(np.sqrt(max(s["sum_sq"]/s["count"]-s["mean"]**2,0)),1e-3)
    (RELEASE/"statistics.json").write_text(json.dumps(dict(fitted_role="train_only",quantity="native_surface_reflectance",bands=stats),indent=2))
    return stats


def masking_contract():
    result={}
    for sensor,c in [("S2",10),("L8",7)]:
        generator=np.random.default_rng(0)
        combinations=[list(k) for n in range(2,c//2+1) for k in itertools.combinations(range(c),n)]
        held=[combinations[i] for i in generator.choice(len(combinations),12,replace=False)]
        result[sensor]=dict(unseen_combinations=held,training="uniform target band, uniform legal hidden count1..floor(C/2), conditional subset containing target; each observation independent",validation="all singleton targets; unseen combinations evaluated separately",seed=0)
    (RELEASE/"masking.json").write_text(json.dumps(result,indent=2))
    return result


class NativeSet:
    def __init__(self,split,device="cuda"):
        self.split=split;self.device=device;self.records={};self.value={};self.valid={};self.phase={};self.metadata={}
        for sensor in ["S2","L8"]:
            records=[r for r in rows(split) if r["sensor"]==sensor]
            if not records:raise ValueError(f"no {split}/{sensor} native observations")
            self.records[sensor]=records;self.metadata[sensor]=records[0]["bands"]
            self.value[sensor]={};self.valid[sensor]={};self.phase[sensor]={}
            cache=[np.load(r["cache"]) for r in records]
            try:
                for i,band in enumerate(self.metadata[sensor]):
                    name=band["name"]
                    self.value[sensor][name]=torch.from_numpy(np.stack([z[name] for z in cache])).to(device)
                    self.valid[sensor][name]=torch.from_numpy(np.stack([z[name+"_valid"] for z in cache])).to(device)
                    self.phase[sensor][name]=torch.tensor([r["bands"][i]["phase_xy_m"] for r in records],device=device,dtype=torch.float32)
            finally:
                for z in cache:z.close()

    def bands(self,sensor,indices,statistics,hidden=None):
        result=[]
        for i,band in enumerate(self.metadata[sensor]):
            name=band["name"];s=statistics[name]
            value=(self.value[sensor][name][indices]-s["mean"])/s["std"]
            valid=self.valid[sensor][name][indices]
            if hidden is not None:valid=valid&~hidden[:,i,None,None]
            result.append(BandRaster(name,value,valid,band["wavelength_nm"],band["bandwidth_nm"],band["native_gsd_m"],self.phase[sensor][name][indices]))
        return result

    def query(self,sensor,index,ids):
        band=self.metadata[sensor][index]
        return TargetQuery(sensor,band["name"],band["wavelength_nm"],band["bandwidth_nm"],band["native_gsd_m"],band["native_gsd_m"],self.phase[sensor][band["name"]][ids])


class BalancedSampler:
    def __init__(self,lengths,seed=0):
        self.generator=np.random.default_rng(seed);self.lengths=lengths
        self.orders={s:self.generator.permutation(n).tolist() for s,n in lengths.items()}
        self.cursors={s:0 for s in lengths};self.exposure={s:[0]*n for s,n in lengths.items()}

    def take(self,sensor,count):
        result=[]
        while len(result)<count:
            cursor=self.cursors[sensor];n=self.lengths[sensor]
            take=min(count-len(result),n-cursor);part=self.orders[sensor][cursor:cursor+take]
            result.extend(part)
            for index in part:self.exposure[sensor][index]+=1
            cursor+=take
            if cursor==n:self.orders[sensor]=self.generator.permutation(n).tolist();cursor=0
            self.cursors[sensor]=cursor
        return result

    def state_dict(self):return dict(orders=self.orders,cursors=self.cursors,exposure=self.exposure,rng=self.generator.bit_generator.state)
    def load_state_dict(self,state):
        self.orders=state["orders"];self.cursors=state["cursors"];self.exposure=state["exposure"];self.generator.bit_generator.state=state["rng"]


class MaskSampler:
    def __init__(self,contract,seed=0):
        self.generator=np.random.default_rng(seed);self.choices={}
        for sensor,c in [("S2",10),("L8",7)]:
            reserved={tuple(x) for x in contract[sensor]["unseen_combinations"]}
            self.choices[sensor]={}
            for target in range(c):
                self.choices[sensor][target]={k:[part for part in itertools.combinations(range(c),k) if target in part and part not in reserved] for k in range(1,c//2+1)}

    def sample(self,sensor,target,count,device):
        c=10 if sensor=="S2" else 7;hidden=np.zeros((count,c),bool)
        choices=self.choices[sensor][target]
        for i in range(count):
            k=int(self.generator.integers(1,c//2+1));available=choices[k]
            part=available[int(self.generator.integers(len(available)))];hidden[i,list(part)]=True
        return torch.from_numpy(hidden).to(device)
