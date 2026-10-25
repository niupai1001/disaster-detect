"""Score saved spatial reconstruction, without updating USE or temporary heads."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from landslide_seg.models.frozen_use import state_sha256
from pretrain_use_independent import OUT, initialize, inputs, spatial_batch, masked_mae
from audit_use_independent_data import read, sha


def error_totals(prediction,target,valid):
    if prediction.shape!=target.shape or valid.shape!=target.shape or valid.dtype!=torch.bool:
        raise ValueError('Spatial error shape/mask mismatch')
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Nonfinite valid spatial prediction or target')
    delta=torch.where(valid,prediction.double()-target.double(),0)
    result=dict(absolute=delta.abs().sum((-1,-2)),squared=delta.square().sum((-1,-2)),pixels=valid.sum((-1,-2)))
    edge_absolute=torch.zeros_like(result['absolute']);edge_squared=edge_absolute.clone();edges=torch.zeros_like(result['pixels'])
    for axis in (-1,-2):
        a=[slice(None)]*4;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
        a,b=tuple(a),tuple(b)
        usable=valid[a]&valid[b]
        error=torch.where(usable,(prediction[a].double()-prediction[b].double())-(target[a].double()-target[b].double()),0)
        edge_absolute+=error.abs().sum((-1,-2));edge_squared+=error.square().sum((-1,-2));edges+=usable.sum((-1,-2))
    result.update(edge_absolute=edge_absolute,edge_squared=edge_squared,edges=edges)
    return result


@torch.no_grad()
def evaluate(model,split,statistics,device):
    model.eval().requires_grad_(False)
    before=state_sha256(model)
    metrics={}
    for sensor,rows in split.items():
        rng=np.random.default_rng(1236)
        loss_sum=0.;bands={}
        for start in range(0,len(rows),8):
            selected=rows[start:start+8]
            obs,target,valid,metadata,known=spatial_batch(selected,rng,statistics,device)
            prediction=model(obs,metadata,known,spatial=True)
            loss_sum+=float(masked_mae(prediction,target,valid))*len(selected)
            totals={key:value.cpu().tolist() for key,value in error_totals(prediction,target,valid).items()}
            for i,row in enumerate(selected):
                for j,name in enumerate(row['band_ids']):
                    current=bands.setdefault(name,dict(absolute=0.,squared=0.,pixels=0,edge_absolute=0.,edge_squared=0.,edges=0,
                                                       std=statistics[sensor]['std'][j]))
                    if current['std']!=statistics[sensor]['std'][j]:raise ValueError('Band normalization differs inside sensor')
                    for key in totals:current[key]+=totals[key][i][j]
        result={}
        for name,value in bands.items():
            pixels,edges=value['pixels'],value['edges'];sd=value['std']
            if pixels==0:raise ValueError('No valid hidden pixels for band')
            mae=value['absolute']/pixels;rmse=math.sqrt(value['squared']/pixels)
            edge_mae=value['edge_absolute']/edges if edges else None
            edge_rmse=math.sqrt(value['edge_squared']/edges) if edges else None
            result[name]=dict(hidden_valid_pixels=pixels,hidden_valid_neighbor_pairs=edges,
                standardized_mae=mae,standardized_rmse=rmse,reflectance_mae=mae*sd,reflectance_rmse=rmse*sd,
                standardized_neighbor_difference_mae=edge_mae,standardized_neighbor_difference_rmse=edge_rmse,
                reflectance_neighbor_difference_mae=edge_mae*sd if edge_mae is not None else None,
                reflectance_neighbor_difference_rmse=edge_rmse*sd if edge_rmse is not None else None)
        metrics[sensor]=dict(sample_band_balanced_standardized_mae=loss_sum/len(rows),bands=result,
            interpretation='Neighbor differences use adjacent hidden-and-originally-valid pixels on each declared grid, not derivatives per metre')
    if state_sha256(model)!=before:raise ValueError('Spatial evaluation mutated model state')
    return dict(full_model_sha256=before,metrics=metrics)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:1');args=parser.parse_args()
    folder=OUT/'pretraining/spatial';target_path=folder/'spatial_reconstruction_diagnostics.json'
    if target_path.exists():raise FileExistsError('Preserve spatial diagnostics')
    accepted=read(folder/'acceptance.json');config=read(folder/'resolved_config.json')
    if accepted.get('passed') is not True or sha(folder/'best.pt')!=accepted['checkpoints']['best']['file_sha256']:
        raise ValueError('Accepted selected spatial checkpoint required')
    frozen=read(OUT/'frozen_contracts_acceptance.json')['records']['spatial']
    if sha(folder/'initialization.pt')!=frozen['initialization_file_sha256']:
        raise ValueError('Frozen initialization changed')
    splits,statistics,_,_=inputs('spatial')
    if statistics!=config['statistics']:raise ValueError('Frozen statistics changed')
    result={}
    for name,file in (('initial','initialization.pt'),('best','best.pt')):
        saved=torch.load(folder/file,map_location='cpu',weights_only=False)
        model=initialize('spatial',args.device);model.load_state_dict(saved['model'],strict=True)
        if name=='initial' and state_sha256(model)!=config['initialization_sha256']:
            raise ValueError('Initial full model differs')
        if name=='best' and state_sha256(model.use)!=accepted['checkpoints']['best']['use_sha256']:
            raise ValueError('Best USE differs')
        result[name]=evaluate(model,splits['PRETRAIN_VAL'],statistics,args.device)
        result[name]['checkpoint_sha256']=sha(folder/file)
        print(name,'spatial evaluation complete',flush=True)
        del model,saved
    selected=next(row for row in read(folder/'history.json') if row['step']==accepted['checkpoints']['best']['step'])
    for sensor,value in result['best']['metrics'].items():
        if abs(value['sample_band_balanced_standardized_mae']-selected['metrics'][sensor]['standardized_spatial_mae'])>1e-6:
            raise ValueError('Spatial reevaluation differs from saved training validation')
    proof=dict(passed=True,results=result,training_updates_added=0,selection_changed=False,
        validation_files={sensor:[row['file'] for row in rows] for sensor,rows in splits['PRETRAIN_VAL'].items()},
        mask_seed=1236,mask_rule=config['spatial_objective'],auditor_sha256=sha(Path(__file__)),
        limitations=['Includes temporary reconstruction decoder, which is not transferred downstream',
                     'Neighbor difference error is a defined local structure diagnostic, not semantic or downstream success',
                     'Each sensor has its own grid/resolution; physical error per grid step is not a common per-metre derivative',
                     'Initial-versus-best comparison is descriptive; no extra optimization or checkpoint selection'])
    target_path.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Spatial reconstruction diagnostics accepted',flush=True)


if __name__=='__main__':main()
