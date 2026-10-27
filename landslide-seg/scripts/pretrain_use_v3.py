"""USE v3 three-view objectives. No segmentation labels enter this module."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from pretrain_use_v2 import (inputs,load_rows,sample_rows,task_view,deterministic,lr,
                            checkpoint,restore,atomic_save)
from landslide_seg.models.use_v3 import USEv3
from landslide_seg.models.use_v3_reconstruction import Reconstruction
from evaluate_use_v2 import baseline


def initialize(device='cpu',task=None,encoder=None):
    deterministic(); torch.manual_seed(1234)
    encoder=USEv3() if encoder is None else encoder
    torch.manual_seed(1235)
    return Reconstruction(encoder,task).to(device)


def mean_error(pred,target,mask):
    count=mask.sum((-1,-2)); usable=count>0
    if not usable.any(): raise ValueError('No valid targets')
    error=torch.where(mask,pred-target,0).abs().sum((-1,-2))/count.clamp_min(1)
    return error[usable].mean()


def spatial_error(pred,target,hidden,valid):
    values=[mean_error(pred,target,hidden)]
    for scale in (2,4):
        count=F.avg_pool2d(valid.float(),scale,ceil_mode=True)
        p=F.avg_pool2d(torch.where(valid,pred,0),scale,ceil_mode=True)/count.clamp_min(1e-8)
        t=F.avg_pool2d(torch.where(valid,target,0),scale,ceil_mode=True)/count.clamp_min(1e-8)
        support=(F.max_pool2d(hidden.float(),scale,ceil_mode=True)>0)&(count>0)
        values.append(mean_error(p,t,support))
    return torch.stack(values).mean()


def gradient_error(pred,target,hidden,valid):
    values=[]; counts=0
    for axis in (-1,-2):
        n=pred.shape[axis]-1
        a=lambda t:t.narrow(axis,0,n)
        b=lambda t:t.narrow(axis,1,n)
        mask=a(valid)&b(valid)&(a(hidden)|b(hidden))
        if mask.any():
            values.append(mean_error(b(pred)-a(pred),b(target)-a(target),mask)); counts+=int(mask.sum())
    if not values: raise ValueError('No valid gradient pairs')
    return torch.stack(values).mean(),counts


def consistency(a,b):
    mask=a['valid']['main']&b['valid']['main']
    if not mask.any(): raise ValueError('No common Context support')
    x=F.normalize(a['main'],dim=1); y=F.normalize(b['main'],dim=1)
    return (x-y).square().sum(1)[mask].mean(),int(mask.sum())


def views(model,original,seed,block,statistics=None,baseline_only=False):
    predictions=[]; features=[]; raw={}; counts={}; gradients=[]
    valid=original['pixel_valid']&original['band_valid'][:,:,None,None]
    for i,task in enumerate(('spectral','fine','context')):
        view,hidden,summary=task_view(original,task,seed+i,block=block,metadata_mode='known')
        if baseline_only:
            pred=baseline(view,statistics)[0 if task=='spectral' else 1]
        else:
            feature=model.features(view); features.append(feature)
            pred=model.predict(feature,view['band_metadata'],view['known_mask'],summary)
        raw[task]=spatial_error(pred,original['x'],hidden,valid) if task=='context' else mean_error(pred,original['x'],hidden)
        counts[task]=int(hidden.sum())
        g,n=gradient_error(pred,original['x'],hidden,valid); gradients.append(g)
        counts['gradient_'+task]=n
        predictions.append(pred)
    raw['gradient']=torch.stack(gradients).mean()
    if not baseline_only: raw['consistency'],counts['consistency']=consistency(features[0],features[1])
    return raw,counts


def calibrate(model,splits,stats,device):
    result={}; members={}
    for sensor,pool in splits['TRAIN'].items():
        chosen=np.random.default_rng(1234).permutation(len(pool))[:32]
        members[sensor]=[pool[i]['file'] for i in chosen]
        values={key:[] for key in ('spectral','fine','context','gradient','consistency')}
        for j,i in enumerate(chosen):
            original=load_rows([pool[i]],stats,device)
            with torch.no_grad():
                raw,_=views(model,original,700000+j*3,(16,32,64)[j%3],stats[sensor],True)
                initial,_=views(model,original,700000+j*3,(16,32,64)[j%3])
            raw['consistency']=initial['consistency']
            for key,v in raw.items(): values[key].append(float(v))
        result[sensor]={key:float(np.mean(v)) for key,v in values.items()}
        if any(not np.isfinite(v) or v<=1e-8 for v in result[sensor].values()):
            raise ValueError('Degenerate calibration denominator')
    return dict(coefficients=result,members=members,seed=1234,role='TRAIN',count_per_sensor=32)


def update(model,optimizer,split,stats,step,calibration,device):
    rows=sample_rows(split,step); optimizer.zero_grad(set_to_none=True)
    records=[]
    for micro in range(4):
        current=rows[micro*2:micro*2+2]
        original=load_rows(current,stats,device,1234+step*8+micro*2)
        # Sensor batches are homogeneous: six S2 followed by two Landsat.
        raw,counts=views(model,original,1234+step*100+micro*3,(16,32,64)[(step-1)%3])
        denom=calibration['coefficients'][current[0]['sensor']]
        normalized={k:v/denom[k] for k,v in raw.items()}
        weighted={k:v*(.1 if k in ('gradient','consistency') else 1)/3.2 for k,v in normalized.items()}
        (sum(weighted.values())/4).backward()
        records.append(dict(raw={k:float(v.detach()) for k,v in raw.items()},
                            normalized={k:float(v.detach()) for k,v in normalized.items()},
                            weighted={k:float(v.detach()) for k,v in weighted.items()},counts=counts))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm): raise ValueError('Nonfinite gradients')
    optimizer.step()
    return dict(step=step,gradient_norm=float(norm),microbatches=records,files=[r['file'] for r in rows],
                sensors=[r['sensor'] for r in rows])
