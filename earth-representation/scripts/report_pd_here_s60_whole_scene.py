"""Report the frozen 2,000-parent whole-image/512-output experiment."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_pd_here_s60_whole_scene import (
    ROOT, FROZEN, TASKS, ExpandedTrainingData, WholeSceneViT, masks, read, write_json,
)
from pd_here_s60.whole_scene import native_axis, pack_scene, tile_bounds, tiles
from pd_here_s60.whole_scene_dual import WholeSceneDualViT
from pd_here_s60.whole_scene_dual_axis import WholeSceneDualAxisViT, MODEL_ID as DUAL_AXIS_ID


def native_indices(band, scene, offset):
    _,ph,pw = next(item for item in scene.layout if item[0]==band.key)
    left,top = scene.origin
    dx,_,x,_,dy,y = band.grid.affine
    h,w = band.grid.shape
    iy,sy = native_axis(h,-dy,top-y,ph,scene.patch_m)
    ix,sx = native_axis(w,dx,x-left,pw,scene.patch_m)
    return iy[:,None]*scene.grid[1]+ix,offset+sy[:,None]*pw+sx,iy,ix


@torch.no_grad()
def predict(model, scene, keep, bands, grids):
    result=torch.zeros_like(scene.values)
    for tile,visible,index in tiles(scene,keep):
        structural=tile.qa.any(-1)
        if structural.any():
            packed=torch.zeros_like(tile.values)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                packed[structural]=model(tile,visible,bands).float()
            result[index]=packed
    output=[]
    start=0
    for band,(_,ph,pw) in zip(grids,scene.layout,strict=True):
        patch,slot,iy,ix = native_indices(band,scene,start)
        output.append((result[patch.to(result.device),slot.to(result.device)].cpu(),patch,slot,iy,ix))
        start+=ph*pw
    return output


def gradient_metrics(prediction, truth, valid):
    dx=[];dy=[]
    for dim,legal in ((0,valid[1:]&valid[:-1]),(1,valid[:,1:]&valid[:,:-1])):
        dx.append(prediction.diff(dim=dim)[legal])
        dy.append(truth.diff(dim=dim)[legal])
    px,py=torch.cat(dx).double(),torch.cat(dy).double()
    if not len(px):return {}
    result=dict(gradient_mse=float((px-py).square().mean()),gradient_pairs=len(px))
    x,y=px-px.mean(),py-py.mean()
    threshold=1e-6*len(px)**.5
    if min(x.norm(),y.norm())>=threshold:
        result['gradient_correlation']=float((x@y)/(x.norm()*y.norm()))
    return result


def mean(records,key):
    defined=[x[key] for x in records if key in x]
    return dict(mean=sum(defined)/len(defined),count=len(defined)) if defined else dict(mean=None,count=0)


def curves(run,history):
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for i,task in enumerate(TASKS):
        for j,metric in enumerate(('mse','within_patch_mse','template_fraction')):
            ax=axes[i,j]
            ax.plot([r['epoch'] for r in history],
                    [r['fixed_train']['aggregate'][task]['source_macro'].get(metric) for r in history],
                    marker='o',label='Fixed TRAIN panel, source macro')
            if metric=='mse':
                ax.plot([r['epoch'] for r in history],[r['train'][task] for r in history],label='TRAIN updates')
            ax.set(title=f'{task} / {metric}',xlabel='Epoch')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(run/'curves.png',dpi=150);plt.close(fig)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    run=args.run
    status=read(run/'status.json')
    history=read(run/'history.json')
    if status['status']!='COMPLETED' or len(history)!=10 or history[-1]['updates']!=1250:
        raise ValueError('Expected the complete ten-epoch run')
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    curves(run,history)
    identity=read(run/'identity.json')
    layouts={tuple(tuple(y) for y in x['bands']):x['train_means'] for x in identity['layouts']}
    model_type=(WholeSceneDualViT if identity['patch_m']==120 else
                WholeSceneDualAxisViT if identity['model_id']==DUAL_AXIS_ID else WholeSceneViT)
    model=model_type(layouts).to(args.device).eval()
    state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    assert state['epoch']==10 and json.loads(json.dumps(state['identity']))==identity
    model.load_state_dict(state['model'])
    del state
    subset=read(FROZEN/'subset_expanded_2000_seed17.json')
    by_id={r['id']:r for r in subset['parents']}
    data=ExpandedTrainingData(ROOT)
    members=read(run/'members.json')
    big=max(members,key=lambda r:len(r['tiles']))
    examples={big['id']}
    for source in ('Core S2','Core L8','FLAIR-HUB S2'):
        examples.add(next(x['id'] for x in members if x['source']==source and x['id'] in identity['panel']))
    records=[]; images=[]; same_targets=[]
    for key in dict.fromkeys([*identity['panel'],*sorted(examples)]):
        row=by_id[key]
        scale=subset['dn_scales'][row['domain']] if row['encoding']=='stored_dn' else 1.
        obs=data.load(row,'cpu')
        scene=pack_scene(obs,scale,identity['patch_m']).to(args.device)
        analysis=pack_scene(obs,scale) if scene.patch_m==120 else scene
        comparison_views=masks(analysis,key,0)
        for task,(keep,bands) in zip(TASKS,masks(scene,key,0),strict=True):
            p=predict(model,scene,keep,bands,obs.bands)
            scores=[]
            past_keep,past_bands=comparison_views[TASKS.index(task)]
            past_keep=past_keep.cpu()
            offset=0
            for j,((pred,patch,slot,iy,ix),band) in enumerate(zip(p,obs.bands,strict=True)):
                _,ph,pw=analysis.layout[j]
                analysis_patch,analysis_slot,_,_=native_indices(band,analysis,offset)
                analysis_slot-=offset
                offset+=ph*pw
                if task=='B1' and bands[j]:continue
                keep_cpu=keep.cpu()
                native_keep=keep_cpu[patch] if keep_cpu.ndim==1 else keep_cpu[patch,slot]
                valid=band.valid & (~native_keep if task=='A' else True)
                reference_support=band.valid & (~past_keep[analysis_patch] if task=='A' else ~past_bands[j].cpu())
                same_targets.append(dict(id=key,task=task,band=band.key,target_pixels=int(valid.sum()),
                                         matches_spatial_60=torch.equal(valid,reference_support)))
                truth=band.values/scale
                record=dict(band=band.key,**gradient_metrics(pred,truth,valid))
                if valid.any():
                    err=(pred-truth)[valid].double()
                    ids=analysis_patch[valid]
                    sums=torch.zeros(len(analysis.values),dtype=err.dtype).scatter_add_(0,ids,err)
                    counts=torch.bincount(ids,minlength=len(sums))
                    record.update(mse=float(err.square().mean()),
                                  common60_within_patch_mse=float((err-(sums/counts.clamp_min(1))[ids]).square().mean()))
                    complete=counts==ph*pw
                    if int(complete.sum())>=2:
                        use=valid & complete[analysis_patch]
                        ids=complete.long().cumsum(0)[analysis_patch[use]]-1
                        patch_values=torch.zeros(int(complete.sum()),ph*pw)
                        patch_values[ids,analysis_slot[use]]=pred[use]
                        texture=patch_values-patch_values.mean(-1,keepdim=True)
                        energy=texture.square().mean()
                        if energy>1e-20:
                            record['common60_template_fraction']=float(texture.mean(0).square().mean()/energy)
                scores.append(record)
                if task=='B1' and key in examples:
                    # Show one non-overlapping native output tile per example.
                    bounds=[b for b in tile_bounds(scene) if ((iy>=b[0])&(iy<b[1])).any() and ((ix>=b[2])&(ix<b[3])).any()]
                    def support(bounds):
                        r0,r1,c0,c1=bounds
                        return ((iy>=r0)&(iy<r1))[:,None]&((ix>=c0)&(ix<c1))[None]
                    sl=max((support(b) for b in bounds),key=lambda m:int((m&valid).sum()))
                    yy,xx=sl.nonzero(as_tuple=True)
                    box=(slice(int(yy.min()),int(yy.max())+1),slice(int(xx.min()),int(xx.max())+1))
                    assert max(pred[box].shape)<=512
                    images.append((row['source'],band.key,key[-8:],truth[box],pred[box],valid[box]))
            if key in identity['panel']:
                records.append(dict(id=key,source=row['source'],task=task,bands=scores,
                                    metrics={k:v['mean'] for k in ('mse','common60_within_patch_mse','common60_template_fraction','gradient_mse','gradient_correlation')
                                             if (v:=mean(scores,k))['count']}))
    aggregate={}
    metric_keys=('mse','common60_within_patch_mse','common60_template_fraction','gradient_mse','gradient_correlation')
    for task in TASKS:
        sources={source:{k:mean([r['metrics'] for r in records if r['task']==task and r['source']==source],k)
                         for k in metric_keys} for source in identity['sources']}
        aggregate[task]=dict(by_source=sources,source_macro={k:mean([
            {k:source[k]['mean']} for source in sources.values() if source[k]['count']],k)
            for k in metric_keys})
    write_json(run/'native_gradients_last.json',dict(aggregate=aggregate,parents=records,
        native_support_checks=same_targets,scope='fixed TRAIN panel, original native adjacent QA-valid hidden pixels; shared 60 m analysis grid'))
    fig,axes=plt.subplots(len(images),3,figsize=(10,3*len(images)),squeeze=False,layout='constrained')
    for ax,title in zip(axes[0],('Native truth','Last prediction','Signed error')):ax.set_title(title)
    for axs,(source,band,key,y,p,q) in zip(axes,images,strict=True):
        lo,hi=float(y[q].min()),float(y[q].max())
        cap=max(float((p-y)[q].abs().quantile(.98)),1e-5)
        for ax,z,cmap,vmin,vmax in zip(axs,(y,p,p-y),('gray','gray','RdBu_r'),(lo,lo,-cap),(max(hi,lo+1e-5),max(hi,lo+1e-5),cap),strict=True):
            ax.imshow(np.ma.array(z.numpy(),mask=~q.numpy()),cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
        axs[0].set_ylabel(f'{source} / {band}\n{key} / {y.shape[0]}×{y.shape[1]}')
    fig.suptitle('B1 fixed TRAIN examples | epoch 10 last')
    fig.savefig(run/'B1_predictions.png',dpi=150);plt.close(fig)
    print('Finished whole-scene report',flush=True)


if __name__=='__main__':main()
