"""TRAIN-only tuned ridge probes for native hidden-band texture before/after compression."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_pd_here_r3_texture_study import BASE, Study, initial_model, batches
from pd_here_geotoken_dualvit_r3.data import write_json

LAYERS=('spectral_pre','spectral_post','spatial_pre','spatial_post','joint_post','fusion_post','global_input','global1_after_attention','global1','global2_after_attention','global2','encoder')
LAMBDAS=(1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1.,100.)


def inner_groups(rows):
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    seen={}
    for i,row in enumerate(rows):
        for key in ('isolation_group','canonical_acquisition'):
            marker=(key,row[key])
            if marker in seen:parent[find(i)]=find(seen[marker])
            seen[marker]=i
        a=row['bbox']
        for j in range(i):
            b=rows[j]['bbox']
            if a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]:
                parent[find(i)]=find(j)
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[find(i)].append(row['id'])
    ordered=sorted(groups.values(),key=lambda g:min(g))
    random.Random(17).shuffle(ordered)
    tune={i for g in ordered[round(.8*len(ordered)):] for i in g}
    return tune


def prepare_fit(x,y,weights):
    weights=weights/weights.sum()
    mx=(x*weights[:,None]).sum(0)
    my=(y*weights[:,None]).sum(0)
    xc=x-mx
    scale=(xc.square().sum(-1)*weights).sum().sqrt().clamp_min(1e-12)
    xc=xc/scale
    root=weights.sqrt()[:,None]
    design=xc*root
    response=(y-my)*root
    return mx,my,scale,design,response


def ridge_fit(prepared,regularization):
    mx,my,scale,design,response=prepared
    gram=design@design.T
    alpha=torch.linalg.solve(gram+regularization*torch.eye(len(gram),device=gram.device,dtype=gram.dtype),response)
    return dict(mean_x=mx,mean_y=my,scale=scale,weight=design.T@alpha)


def predict(model,x):
    return (x-model['mean_x'])/model['scale']@model['weight']+model['mean_y']


def weights_for(parents,device):
    counts={parent:parents.count(parent) for parent in set(parents)}
    return torch.tensor([1/counts[p] for p in parents],device=device,dtype=torch.float64)


def score(pred,y,parents,shape):
    error=(pred-y).double()
    h,w=shape
    px=pred.reshape(-1,h,w);tx=y.reshape(-1,h,w)
    dx=torch.cat(((px[:,:,1:]-px[:,:,:-1]).flatten(1),(px[:,1:]-px[:,:-1]).flatten(1)),-1)
    dy=torch.cat(((tx[:,:,1:]-tx[:,:,:-1]).flatten(1),(tx[:,1:]-tx[:,:-1]).flatten(1)),-1)
    metrics=dict(texture_mse=error.square().mean(-1),gradient_mse=(dx-dy).square().mean(-1),
                 target_texture_energy=y.double().square().mean(-1))
    gradient_correlation=torch.full((len(y),),float('nan'),device=y.device,dtype=torch.float64)
    ax=dx-dx.mean(-1,keepdim=True);ay=dy-dy.mean(-1,keepdim=True)
    valid=(ax.square().mean(-1).sqrt()>=1e-6)&(ay.square().mean(-1).sqrt()>=1e-6)
    gradient_correlation[valid]=(ax[valid]*ay[valid]).sum(-1)/(ax[valid].norm(dim=-1)*ay[valid].norm(dim=-1))
    metrics['gradient_correlation']=gradient_correlation
    records=[]
    for p in sorted(set(parents)):
        ids=torch.tensor([i for i,v in enumerate(parents) if v==p],device=y.device)
        record=dict(parent=p)
        for key,value in metrics.items():
            vv=value[ids]
            vv=vv[torch.isfinite(vv)]
            if len(vv):record[key]=float(vv.mean())
        records.append(record)
    result={key:float(np.mean([r[key] for r in records if key in r]))
            for key in metrics if any(key in r for r in records)}
    return result,records


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cuda:0'
    study=Study(BASE/'texture_study',device)
    variant='fusion256_global2_zerohead'
    state=torch.load(study.release/variant/'last.pt',map_location='cpu',weights_only=True)
    model,_=initial_model(study.layout,variant,state['identity']['statistics'],device)
    model.load_state_dict(state['model']);model.eval()
    features={}
    def pre(name):
        def capture(module,args):features[name]=args[0].detach()
        return capture
    def post(name):
        def capture(module,args,result):features[name]=result.detach()
        return capture
    hooks=[
        model.spectral_reduce.register_forward_pre_hook(pre('spectral_pre')),
        model.spatial_reduce.register_forward_pre_hook(pre('spatial_pre')),
        model.spectral_reduce.register_forward_hook(post('spectral_post')),
        model.spatial_reduce.register_forward_hook(post('spatial_post')),
        model.encoder_norm.register_forward_hook(post('encoder')),
        model.fuse.register_forward_hook(post('fusion_post')),
        model.global_block.register_forward_pre_hook(pre('global_input')),
        model.global_block.register_forward_hook(post('global1')),
        model.global_block.norm2.register_forward_pre_hook(pre('global1_after_attention')),
        model.extra_global_blocks[0].norm2.register_forward_pre_hook(pre('global2_after_attention')),
        model.extra_global_blocks[0].register_forward_hook(post('global2'))]
    data={role:defaultdict(list) for role in study.rows}
    offsets=np.cumsum([0,*[h*w for h,w in study.layout.native_shapes]])
    for role,rows in study.rows.items():
        for start,batch_rows in batches(rows):
            values,qa,views=study.inputs(batch_rows,0)
            keep,bands=views[1]
            output=model(values,qa,keep,bands)
            features['joint_post']=torch.cat((features['spatial_post'],features['spectral_post']),-1)
            for i,row in enumerate(batch_rows):
                hidden=int((~bands[i]).nonzero().item())
                # Common full QA support for all probes and targets, with at least two patches.
                legal=keep[i].clone()
                for q in qa:legal &= q[i].flatten(1).all(-1)
                ids=legal.nonzero().flatten().tolist()
                if len(ids)<2:continue
                if role=='train' and len(ids)>4:
                    ids=sorted(random.Random(f"17:{row['id']}:probe").sample(ids,4))
                positions=output['ids_keep'][i].tolist()
                packed=[positions.index(j) for j in ids]
                truth=values[hidden][i,ids].flatten(1)
                texture=truth-truth.mean(-1,keepdim=True)
                predicted=output['prediction'][i,ids,offsets[hidden]:offsets[hidden+1]]
                predicted=predicted-predicted.mean(-1,keepdim=True)
                raw=torch.cat([torch.where(bands[i,j],v[i,ids],0.).flatten(1) for j,v in enumerate(values)],-1)
                entry=dict(parent=row['id'],patches=ids,target=texture.cpu(),
                           model_prediction=predicted.cpu(),raw_visible=raw.cpu())
                for name in LAYERS:entry[name]=features[name][i,packed].cpu()
                data[role][hidden].append(entry)
            if start%160==0:print('features',role,start,flush=True)
    for hook in hooks:hook.remove()
    tune=inner_groups(study.rows['train'])
    out=study.release/'compression_probe'
    out.mkdir(exist_ok=True)
    report=dict(checkpoint=str(study.release/variant/'last.pt'),task='B1 exposure0',
                probes=['raw_visible',*LAYERS],target='hidden-band native texture after subtracting target patch mean',
                input_target_mean=False,
                scope='700 TRAIN /300 development VAL pool; same full QA patches for all probes; max4 patches/TRAIN parent, all legal VAL patches',
                fitting='parent-equal ridge, centered features and scalar unit feature-energy normalization; lambda chosen in TRAIN-only geographic-group holdout',
                lambda_grid=LAMBDAS,geometry='gradient uses native horizontal/vertical neighbours within each patch only',
                limitations=['Different input dimensions imply different probe parameter counts; cross-validation and evaluation shared',
                             'A hidden geographic patch has no local pre-compression feature and is not tested here',
                             'This diagnoses linear accessibility of texture, not total mutual information'],
                bands={})
    for band in sorted(data['train']):
        train_entries=data['train'][band];val_entries=data['val'][band]
        fit_entries=[r for r in train_entries if r['parent'] not in tune]
        tune_entries=[r for r in train_entries if r['parent'] in tune]
        if not fit_entries or not tune_entries or not val_entries:
            report['bands'][study.layout.band_keys[band]]=dict(status='INSUFFICIENT_GROUPS')
            continue
        def concatenate(entries,key):
            return torch.cat([r[key] for r in entries]).to(device=device,dtype=torch.float64)
        def parent_ids(entries):
            return [r['parent'] for r in entries for _ in r['patches']]
        yt=concatenate(train_entries,'target');yv=concatenate(val_entries,'target')
        yf=concatenate(fit_entries,'target');yh=concatenate(tune_entries,'target')
        pt=parent_ids(train_entries);pv=parent_ids(val_entries)
        pf=parent_ids(fit_entries);ph=parent_ids(tune_entries)
        wt=weights_for(pt,device);wf=weights_for(pf,device);wh=weights_for(ph,device)
        band_report=dict(status='PASS',train_parents=len(train_entries),val_parents=len(val_entries),
                         fit_parents=len(fit_entries),tune_parents=len(tune_entries),train_patches=len(pt),val_patches=len(pv),probes={})
        for name in ('raw_visible',*LAYERS):
            xf=concatenate(fit_entries,name);xh=concatenate(tune_entries,name)
            prepared=prepare_fit(xf,yf,wf)
            scores=[]
            for regularization in LAMBDAS:
                fitted=ridge_fit(prepared,regularization)
                err=(predict(fitted,xh)-yh).square().mean(-1)
                scores.append(float((err*wh).sum()/wh.sum()))
            chosen=LAMBDAS[int(np.argmin(scores))]
            xt=concatenate(train_entries,name);xv=concatenate(val_entries,name)
            fitted=ridge_fit(prepare_fit(xt,yt,wt),chosen)
            train_score,_=score(predict(fitted,xt),yt,pt,study.layout.native_shapes[band])
            val_score,per_parent=score(predict(fitted,xv),yv,pv,study.layout.native_shapes[band])
            band_report['probes'][name]=dict(input_dimensions=xt.shape[-1],lambda_selected=chosen,
                holdout_scores=scores,train=train_score,val=val_score,val_parents=per_parent)
            print(study.layout.band_keys[band],name,chosen,val_score['texture_mse'],flush=True)
        for name,estimate in [('constant_texture',torch.zeros_like(yv)),('current_decoder',concatenate(val_entries,'model_prediction'))]:
            val_score,per_parent=score(estimate,yv,pv,study.layout.native_shapes[band])
            band_report['probes'][name]=dict(val=val_score,val_parents=per_parent)
        report['bands'][study.layout.band_keys[band]]=band_report
        write_json(out/'report.json',report)
    # Equal parent within each hidden band, then equal hidden-band macro.
    names=next(v['probes'].keys() for v in report['bands'].values() if v['status']=='PASS')
    report['macro']={name:{key:float(np.mean([v['probes'][name]['val'][key] for v in report['bands'].values()
                      if v['status']=='PASS' and key in v['probes'][name]['val']]))
                         for key in ('texture_mse','gradient_mse','target_texture_energy')}
                     for name in names}
    write_json(out/'report.json',report)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,key in zip(axes,('texture_mse','gradient_mse')):
        ax.bar(range(len(names)),[report['macro'][n][key] for n in names],color='#467896')
        ax.set_xticks(range(len(names)),names,rotation=45,ha='right')
        ax.set_ylabel('Development VAL '+key)
        ax.set_yscale('log');ax.grid(axis='y',alpha=.25)
    fig.suptitle('Frozen Encoder | TRAIN-tuned linear probes | Hidden-band texture')
    fig.savefig(out/'probe_comparison.png',dpi=160)
    print(json.dumps(report['macro'],indent=2))


if __name__=='__main__':main()
