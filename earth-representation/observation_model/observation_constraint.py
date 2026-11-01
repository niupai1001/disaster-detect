"""Approximate OLI observation operator;10m output is an estimate, never target truth."""
import contextlib

import numpy as np
import torch

from .model import TargetQuery
from .operators import observable_10_to_30


def observe_fine_prediction(model,representation,query):
    fine_query=TargetQuery(query.sensor,query.band,query.wavelength_nm,query.bandwidth_nm,
                           query.native_gsd_m,10,(5.,5.))
    fine,fine_valid=model.decoder(representation,fine_query)
    phase=torch.as_tensor(query.phase_xy_m,device=fine.device,dtype=torch.float32)
    if phase.ndim==1:phase=phase.expand(len(fine),2)
    source_phase=torch.full_like(phase,5.)
    with torch.autocast("cuda",enabled=False):
        coarse,coverage=observable_10_to_30(fine.float(),fine_valid,source_phase,phase)
    return fine,fine_valid,coarse,coverage


@torch.no_grad()
def evaluate_observation(model,pairs,statistics,precision):
    from .train_cross import evaluate_cross,cross_query
    result=evaluate_cross(model,pairs,statistics,precision);model.eval();metrics={};scenes=[]
    for i,row in enumerate(pairs.records):
        inputs=pairs.data.bands("S2",pairs.indices["S2"][i:i+1],statistics)
        manager=torch.autocast("cuda",dtype=torch.bfloat16) if precision!="float32" else contextlib.nullcontext()
        with manager:
            representation=model.encode(inputs)
            for j,band in enumerate(pairs.data.metadata["L8"]):
                query=cross_query(pairs.data,"L8",j,pairs.phase[i:i+1])
                _,_,coarse,coverage=observe_fine_prediction(model,representation,query)
                valid=pairs.valid["L8"][i:i+1,j]
                error=(coarse-pairs.target["L8"][i:i+1,j]).double()[valid]
                totals=np.array([len(error),float(error.abs().sum()),float(error.square().sum()),float(error.sum())])
                metrics[band["name"]]=metrics.get(band["name"],np.zeros(4))+totals
                scenes.append(dict(site_id=row["site_id"],band=band["name"],count=len(error),sum_squared=float(error.square().sum()),
                                   low_operator_coverage_pixels=int((valid&(coverage<.8)).sum())))
    bands={}
    for name,totals in metrics.items():
        count=max(totals[0],1);rmse=float(np.sqrt(totals[2]/count))
        bands[name]=dict(count=int(totals[0]),rmse=rmse,mae=float(totals[1]/count),bias=float(totals[3]/count),normalized_rmse=rmse/statistics[name]["std"])
    result["observed_fine30"]=dict(score=float(np.mean([r["normalized_rmse"] for r in bands.values()])),bands=bands,scenes=scenes,
                                  boundary="10m model estimate degraded by approximate operator versus real30m Landsat; no Landsat10m truth RMSE")
    result["selection"]="unchanged direct bidirectional cross30 validation score; observed fine30 diagnostic reported separately"
    model.train();return result
