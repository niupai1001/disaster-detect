"""TRAIN-only conditional Ridge/residual controls, evaluated on all383 fixed VAL.

No encoder optimization, image download, test access, or validation tuning.
"""
from collections import defaultdict,Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .expanded_dataset import ExpandedOpticalDataset
from .encoder_v2 import axis_overlap
from .encoder_v2_runtime import BASE
from .expansion_data import OUT,read,save,digest

DEST=BASE/'pre_G5/ridge_audit_v2'


def family(instrument):return 'S2' if instrument.startswith('S2') else 'L8'


def normalization():
    source=read(OUT/'statistics.json')['bands'];result={}
    for fam,instruments in [('S2',['S2A','S2B']),('L8',['LC08'])]:
        bands={k.split(':')[1] for k in source if k.split(':')[0] in instruments}
        for band in bands:
            stats=[source[i+':'+band] for i in instruments if i+':'+band in source]
            n=sum(s['valid_pixels'] for s in stats)
            mean=sum(s['valid_pixels']*s['mean'] for s in stats)/n
            second=sum(s['valid_pixels']*(s['std']**2+s['mean']**2) for s in stats)/n
            result[fam+':'+band]=(mean,max(math.sqrt(max(second-mean*mean,0)),1e-5))
    return result


def tensors(item,device):
    return {b['band_key']:dict(b,x=torch.as_tensor(b['reflectance'],device=device),q=torch.as_tensor(b['pixel_valid'],device=device)) for b in item['bands']}


def aligned(bands,target,cache):
    key=(tuple(target['shape']),tuple(target['transform']))
    if key in cache:return cache[key]
    gx,_,cx,_,e,cy=target['transform'];gy=-e;h,w=target['shape'];device=target['x'].device
    ex=torch.arange(w+1,device=device)*gx;ey=torch.arange(h+1,device=device)*gy
    result={}
    for name,source in bands.items():
        if source['shape']==target['shape'] and source['transform']==target['transform']:
            result[name]=(source['x'],source['q'],torch.zeros_like(source['x']));continue
        sx,_,left,_,se,top=source['transform'];sy=-se;hh,ww=source['shape']
        px=torch.arange(ww,device=device)*sx+(left-cx);py=torch.arange(hh,device=device)*sy+(cy-top)
        wx=axis_overlap(ex,px,px+sx);wy=axis_overlap(ey,py,py+sy)
        area=(wy@source['q'].float())@wx.T
        x=torch.where(source['q'],source['x'],0)
        mean=((wy@x)@wx.T)/area.clamp_min(1e-10)
        second=((wy@x.square())@wx.T)/area.clamp_min(1e-10)
        result[name]=(mean,area>=.8*gx*gy,(second-mean.square()).clamp_min(0))
    cache[key]=result;return result


def contexts(bands,target):
    names=sorted(bands);others=[n for n in names if n!=target]
    result=[('single',others,[])]
    swir=['B11','B12'] if target.startswith('B') else ['SR_B6','SR_B7']
    rededge=['B05','B06','B07']
    if target in swir and all(n in bands for n in swir):result.append(('SWIR_pair',[n for n in names if n not in swir],[]))
    if target in rededge and all(n in bands for n in rededge):result.append(('rededge_group',[n for n in names if n not in rededge],[]))
    reserved=['B04','B08'] if target.startswith('B') else ['SR_B3','SR_B4','SR_B5']
    if target in reserved and all(n in bands for n in reserved):result.append(('visible_NIR_group',[n for n in names if n not in reserved],[]))
    rgb=['B02','B03','B04'] if target.startswith('B') else ['SR_B2','SR_B3','SR_B4']
    if target not in rgb and all(n in bands for n in rgb) and rgb!=others:result.append(('RGB_only',rgb,[]))
    fine=[n for n in others if bands[n]['transform'][0]<bands[target]['transform'][0]]
    if fine:result.append(('single_with_texture',others,fine))
    return result


def feature_matrix(alignment,visible,texture,fam,norm):
    values=[];valid=[];labels=[]
    for name in visible:
        mean,std=norm[fam+':'+name];x,q,_=alignment[name]
        values.append((x-mean)/std);valid.append(q);labels.append(name)
    for name in texture:
        _,std=norm[fam+':'+name];_,q,v=alignment[name]
        values.append(v/(std**2));valid.append(q);labels.append(name+'@within_target_variance')
    return torch.stack(values,-1),torch.stack(valid,-1),labels


def model_key(fam,target,labels):return fam+'|'+target+'|'+','.join(labels)


def new_accumulator(key,labels,device):
    d=len(labels);r=64;seed=int(hashlib.sha256(('0:'+key).encode()).hexdigest()[:14],16)
    gen=torch.Generator(device=device).manual_seed(seed)
    w=torch.randn(d,r,generator=gen,device=device)/math.sqrt(d);b=torch.rand(r,generator=gen,device=device)*2*math.pi
    dim=d+r+1
    return dict(labels=labels,w=w,b=b,total=torch.zeros(dim,dim,device=device,dtype=torch.float64),
        rhs=torch.zeros(dim,device=device,dtype=torch.float64),count=0,observations=set(),pixels=0)


def phi(z,fit):return torch.cos(z@fit['w']+fit['b'])*math.sqrt(2/64)


def solve(accumulators):
    result={}
    for key,a in accumulators.items():
        d=len(a['labels']);n=a['count'];m=a['total']/n;y=a['rhs']/n
        reg=torch.eye(d+1,device=m.device,dtype=m.dtype)*1e-3;reg[0,0]=0
        linear=torch.linalg.solve(m[:d+1,:d+1]+reg,y[:d+1])
        pp=m[d+1:,d+1:]-m[d+1:,0,None]*m[None,0,d+1:]
        residual_rhs=y[d+1:]-m[d+1:,:d+1]@linear
        residual=torch.linalg.solve(pp+1e-3*torch.eye(64,device=m.device,dtype=m.dtype),residual_rhs)
        result[key]=dict(labels=a['labels'],w=a['w'],b=a['b'],linear=linear,residual=residual,
            phi_mean=m[0,d+1:],observations=len(a['observations']),pixels=a['pixels'],sufficient_moment_matrix=m,sufficient_rhs=y)
    return result


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'summary.json').exists():raise FileExistsError('Preserve completed audit')
    start=time.perf_counter();torch.set_num_threads(4);device='cuda:1';torch.cuda.set_device(device)
    norm=normalization();train=ExpandedOpticalDataset(OUT,'TRAIN');val=ExpandedOpticalDataset(OUT,'VAL')
    indices=[i for i,r in enumerate(train.rows) if r['instrument'] in ['S2A','S2B','LC08']]
    save(DEST/'config.json',dict(seed=0,TRAIN_instruments=['S2A','S2B','LC08'],train_observations=len(indices),
        val_observations=len(val),response_holdout='S2C VAL only; excluded from fitting and G5 selection',
        regularization=1e-3,training_samples_per_observation_target=256,RFF_features=64,
        nonlinear_control='TRAIN-only RFF ridge fitted to linear residual; fixed feature basis/regularization; no validation selection',
        contexts=['single','SWIR_pair','rededge_group','visible_NIR_group','RGB_only','single_with_texture'],
        actual_native_targets=True,optimizer_updates=0,data_manifest_sha256=read(OUT/'release.json')['manifest_sha256']))
    accum={}
    with torch.inference_mode():
        for number,index in enumerate(indices):
            row=train.rows[index];bands=tensors(train[index],device);cache={};fam=family(row['instrument'])
            for target,band in bands.items():
                alignment=aligned(bands,band,cache)
                for context,visible,texture in contexts(bands,target):
                    z,q,labels=feature_matrix(alignment,visible,texture,fam,norm)
                    valid=q.all(-1)&band['q'];z=z[valid];truth=band['x'][valid]
                    if len(truth)<16:continue
                    sample=torch.linspace(0,len(truth)-1,min(256,len(truth)),device=device).long();z=z[sample];truth=truth[sample]
                    key=model_key(fam,target,labels)
                    if key not in accum:accum[key]=new_accumulator(key,labels,device)
                    a=accum[key];mean,std=norm[fam+':'+target]
                    augmented=torch.cat([torch.ones(len(z),1,device=device),z,phi(z,a)],-1).double()
                    y=((truth-mean)/std).double()
                    # Each observation contributes equal mass, rather than favoring large windows.
                    a['total']+=augmented.T@augmented/len(z);a['rhs']+=augmented.T@y/len(z)
                    a['count']+=1;a['observations'].add(row['id']);a['pixels']+=len(z)
            if number%100==0:print('RIDGE_FIT',number,len(indices),len(accum),flush=True)
        fits=solve(accum);torch.save(fits,DEST/'TRAIN_fits.pt');del accum
        records=[]
        for number in range(len(val)):
            row=val.rows[number];bands=tensors(val[number],device);cache={};fam=family(row['instrument'])
            cohort='unseen_S2C_response' if row['instrument']=='S2C' else 'known_instruments'
            for target,band in bands.items():
                alignment=aligned(bands,band,cache)
                for context,visible,texture in contexts(bands,target):
                    z,q,labels=feature_matrix(alignment,visible,texture,fam,norm);key=model_key(fam,target,labels)
                    if key not in fits:raise ValueError('No legal TRAIN fit for '+key)
                    fit=fits[key]
                    z=torch.where(q,z,0).reshape(-1,len(labels));aug=torch.cat([torch.ones(len(z),1,device=device),z],-1).double()
                    linear=aug@fit['linear'];nonlinear=linear+(phi(z,fit).double()-fit['phi_mean'])@fit['residual']
                    mean,std=norm[fam+':'+target];truth=band['x'];valid=band['q'];yy=truth[valid].double()
                    metrics={}
                    for name,pred in [('ridge',linear),('ridge_plus_nonlinear_residual',nonlinear)]:
                        pp=(pred*std+mean).reshape(truth.shape)[valid];error=pp-yy
                        sse=float(error.square().sum());sst=float((yy-yy.mean()).square().sum())
                        metrics[name]=dict(mae=float(error.abs().mean()),rmse=float(error.square().mean().sqrt()),
                            nmse=float((error/std).square().mean()),R2=1-sse/sst if sst>1e-12 else None,
                            p95_abs=float(torch.quantile(error.abs(),.95)))
                    records.append(dict(member=row['id'],instrument=row['instrument'],family=fam,cohort=cohort,
                        isolation_group=row['release_isolation_group'],acquisition_key=row['acquisition_key'],
                        context=context,target=target,visible=visible,texture=texture,valid_pixels=int(valid.sum()),
                        target_std=float(yy.std(unbiased=False)),source_missing_fraction=float((~q[valid]).float().mean()),metrics=metrics))
            if number%25==0:print('RIDGE_VAL',number,len(val),flush=True)
        path=DEST/'per_observation.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
        groups=defaultdict(list)
        for r in records:groups[(r['cohort'],r['family'],r['context'],r['target'])].append(r)
        summary=[]
        for key,rows in sorted(groups.items()):
            scores={name:{metric:float(np.mean([r['metrics'][name][metric] for r in rows if r['metrics'][name][metric] is not None]))
                for metric in ['mae','rmse','nmse','R2','p95_abs']} for name in ['ridge','ridge_plus_nonlinear_residual']}
            before=np.array([r['metrics']['ridge']['nmse'] for r in rows]);after=np.array([r['metrics']['ridge_plus_nonlinear_residual']['nmse'] for r in rows])
            clusters=defaultdict(list)
            for index,row in enumerate(rows):clusters[row['isolation_group']].append(index)
            clusters=list(clusters.values());gen=np.random.default_rng(0);bootstrap=[]
            for _ in range(500):
                selected=gen.integers(0,len(clusters),len(clusters));ids=np.concatenate([clusters[i] for i in selected])
                bootstrap.append(float((before[ids]-after[ids]).mean()))
            summary.append(dict(cohort=key[0],family=key[1],context=key[2],target=key[3],observations=len(rows),scores=scores,
                isolation_groups=len(clusters),R2_median={name:float(np.median([r['metrics'][name]['R2'] for r in rows if r['metrics'][name]['R2'] is not None])) for name in scores},
                low_variance_observations=sum(r['target_std']<.01 for r in rows),
                nonlinear_nmse_improvement=float((before-after).mean()),bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist()))
        save(DEST/'summary.json',dict(status='COMPLETE',groups=summary,records=len(records),
            train_observations=len(indices),val_observations=len(val),optimizer_updates=0,seconds=time.perf_counter()-start,
            limitations='Bootstrap resamples frozen spatial/acquisition isolation components, not individual pixels. Residual control is not an encoder transfer result.'))
        print('RIDGE_AUDIT_COMPLETE',len(records),time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
