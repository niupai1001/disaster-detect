import sys
from pathlib import Path
sys.path.insert(0,str(Path('scripts').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from report_pd_here_s60_whole_scene import ROOT,FROZEN,ExpandedTrainingData,MODELS,pack_scene,masks,predict,tile_bounds,read,write_json

root=ROOT/'outputs/pd_here_s60'
names=('native512_legacy_dual2000_v2_seed17','native512_legacy_global1_2000_seed17','native512_legacy_spectral_only_2000_seed17')
runs=[root/name for name in names]
device='cuda:1'
torch.set_num_threads(4)
torch.backends.mha.set_fastpath_enabled(False)
torch.backends.cuda.matmul.allow_tf32=False
models=[]
for path in runs:
    state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
    identity=state['identity']
    assert state['epoch']==10 and state['history'][-1]['updates']==1250
    layouts={tuple(tuple(b) for b in x['bands']):x['train_means'] for x in identity['layouts']}
    model=MODELS[identity['model_id']](layouts).to(device)
    model.load_state_dict(state['model']);model.eval()
    models.append(model)
    del state
subset=read(FROZEN/'subset_expanded_2000_seed17.json')
members={r['id']:r for r in subset['parents']}
data=ExpandedTrainingData(ROOT)
panel=identity['panel']
large=max(read(runs[0]/'members.json'),key=lambda r:len(r['tiles']))
selection=[next(members[k] for k in panel if members[k]['source']==source)
           for source in ('Core L8','Core S2','FLAIR-HUB S2')]
selection.append(members[large['id']])
titles=('Visible input','Native truth','Dual / 2 global','Dual / 1 global','Spectral / 2 global')
with torch.no_grad():
 for task_index,task in enumerate(('A','B1')):
    fig,axes=plt.subplots(len(selection),len(titles),figsize=(14,2.8*len(selection)),squeeze=False,layout='constrained')
    for ax,title in zip(axes[0],titles):ax.set_title(title,fontsize=10)
    records=[]
    for i,row in enumerate(selection):
        scale=subset['dn_scales'][row['domain']] if row['encoding']=='stored_dn' else 1.
        obs=data.load(row,'cpu')
        scene=pack_scene(obs,scale,120).to(device)
        keep,available=masks(scene,row['id'],0)[task_index]
        key='SR_B4' if row['source']=='Core L8' else 'B04'
        j=(next(j for j,b in enumerate(obs.bands) if b.key==key) if task=='A'
           else (~available).nonzero().item())
        band=obs.bands[j]
        preds=[]
        for model in models:
            pred,patch,slot,iy,ix=predict(model,scene,keep,available,obs.bands)[j]
            preds.append(pred)
        native_keep=keep.cpu()[patch] if keep.ndim==1 else keep.cpu()[patch,slot]
        hidden=band.valid & (~native_keep if task=='A' else True)
        bounds=tile_bounds(scene)
        if len(bounds)==1:
            box=(slice(None),slice(None))
        else:
            def view_box(bound):
                r0,r1,c0,c1=bound
                y=((iy>=r0)&(iy<r1)).nonzero().flatten()
                x=((ix>=c0)&(ix<c1)).nonzero().flatten()
                return slice(int(y[0]),int(y[-1])+1),slice(int(x[0]),int(x[-1])+1)
            box=max(map(view_box,bounds),key=lambda sl:int(hidden[sl].sum()))
        y=band.values[box]/scale
        valid=band.valid[box];q=hidden[box]
        preds=[p[box] for p in preds]
        assert q.any() and max(y.shape)<=512
        if task=='A':
            input_image,input_valid=y,valid&~q
        else:
            v=obs.bands[available.nonzero()[0,0].item()]
            input_image,input_valid=v.values/scale,v.valid
            if len(bounds)>1 and v.grid.shape==band.grid.shape:
                input_image,input_valid=input_image[box],input_valid[box]
            elif len(bounds)>1:
                a,_,x,_,e,z=v.grid.affine
                dx,_,ox,_,dy,oy=band.grid.affine
                r0=max(0,int((z-(oy+dy*box[0].start))/-e))
                c0=max(0,int((ox+dx*box[1].start-x)/a))
                r1=min(v.grid.shape[0],int((z-(oy+dy*box[0].stop))/-e+.999999))
                c1=min(v.grid.shape[1],int((ox+dx*box[1].stop-x)/a+.999999))
                input_image,input_valid=input_image[r0:r1,c0:c1],input_valid[r0:r1,c0:c1]
            axes[i,0].set_xlabel(v.key)
        lo=min(float(y[valid].min()),*(float(p[q].min()) for p in preds))
        hi=max(float(y[valid].max()),*(float(p[q].max()) for p in preds),lo+1e-6)
        for col,(img,qa) in enumerate(zip((input_image,y,*preds),(input_valid,valid,q,q,q),strict=True)):
            axes[i,col].imshow(np.ma.array(img.numpy(),mask=~qa.numpy()),interpolation='nearest',cmap='gray',vmin=lo,vmax=hi)
            axes[i,col].set_xticks([]);axes[i,col].set_yticks([])
        axes[i,0].set_ylabel(f'{row["source"]} / {band.key}\n{row["id"][-8:]} / {y.shape[0]}×{y.shape[1]}')
        records.append(dict(id=row['id'],source=row['source'],band=band.key,display_shape=list(y.shape),
            hidden_pixels=int(q.sum()),mse={name:float((p-y)[q].square().mean()) for name,p in zip(names,preds)}))
    fig.suptitle(f'{task} native targets | 120 m architectures | epoch 10 last')
    fig.savefig(runs[1]/f'{task}_predictions_comparison.png',dpi=150);plt.close(fig)
    write_json(runs[1]/f'{task}_predictions_comparison.json',dict(examples=records,train_only=True))
    print(task,len(records),flush=True)
