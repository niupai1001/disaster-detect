"""Compare the capped joint pilot with existing controls on frozen VAL only."""
import json
import os
from pathlib import Path
import numpy as np
import torch

if __name__=='__main__':torch.cuda.init()

from .data import NativeSet, RELEASE, ROOT
from .train_joint import joint_model, balanced_masks, masked_loss
from .train_masked import evaluation
from .model import make_model
from .baselines import SpectralControl
from .check_precision import tile_values


def main():
    torch.set_num_threads(4)
    out=ROOT/'observation_model/outputs/joint_v3';folder=out/'pilot1000';old=ROOT/'observation_model/outputs/repair_v2'
    stats=json.loads((RELEASE/'statistics.json').read_text())['bands'];contract=json.loads((RELEASE/'masking.json').read_text());data=NativeSet('val')
    plans={};rng=np.random.default_rng(1907)
    for sensor in ['S2','L8']:
        c=len(data.metadata[sensor]);n=len(data.records[sensor]);reserved={tuple(x) for x in contract[sensor]['unseen_combinations']}
        plans[sensor]=(list(range(n)),balanced_masks(c,n,reserved,rng),torch.arange(400,device='cuda').expand(n,-1))
    results={}
    for kind in ['new_best','new_last','old_best','ridge']:
        if kind.startswith('new_'):
            state=torch.load(folder/('best.pt' if kind=='new_best' else 'last.pt'),map_location='cuda',weights_only=False)
            assert state['statistics']==stats;model=joint_model(stats).cuda();model.load_state_dict(state['model'])
        elif kind=='old_best':
            state=torch.load(old/'runs/masked_transformer/best_unlabeled.pt',map_location='cuda',weights_only=False)
            assert state['statistics']==stats;model=make_model(coordinate_mode='explicit_global').cuda();model.load_state_dict(state['model'])
        else:
            control=SpectralControl(stats,'ridge',torch.load(old/'baselines/masked_ridge_fits.pt',map_location='cuda',weights_only=False),json.loads((old/'baselines/masked_ridge_lambdas.json').read_text()))
            class Adapter:
                def train(self):return self
                def eval(self):return self
                def encode(self,x):return control.encode(x)
                def decoder(self,x,q,t=None):
                    pred,valid=control.decoder(x,q)
                    return (pred,valid) if t is None else (tile_values(pred,t,q.grid_gsd_m),tile_values(valid,t,q.grid_gsd_m))
            model=Adapter()
        model.eval()
        with torch.no_grad():
            single=evaluation(model,data,stats,'float32');model.eval();multi,bands,_,_=masked_loss(model,data,stats,plans)
        smse=sum(sum(b['normalized_rmse']**2 for b in single['bands'].values() if b['sensor']==s)/len(data.metadata[s]) for s in ['S2','L8'])/2
        results[kind]=dict(score=.5*(smse+multi),singleton_mse=smse,multiple_mse=multi,singleton_nrmse=single['score'],singleton=single,multiple_bands=bands)
        print(json.dumps(dict(kind=kind,score=results[kind]['score'],singleton_nrmse=single['score'])),flush=True)
        del model
    result=dict(status='PILOT_VAL_COMPARISON_COMPLETE',results=results,data_release=str(RELEASE),split='val',selection='new best by equal singleton/multiple normalized MSE; no TEST',
        boundary='1000-step candidate cap; changed training strategy package, not single-factor causal ablation; old weights trained longer; no convergence or significance claim')
    (out/'pilot_comparison.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
