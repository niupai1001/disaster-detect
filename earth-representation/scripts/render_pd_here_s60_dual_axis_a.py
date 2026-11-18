import sys, json
from pathlib import Path
sys.path.insert(0,str(Path('scripts').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from report_pd_here_s60_whole_scene import (ROOT,FROZEN,ExpandedTrainingData,
    WholeSceneViT,WholeSceneDualAxisViT,pack_scene,masks,predict,tile_bounds,read,write_json)

root=ROOT/'outputs/pd_here_s60'
reference=root/'native512_expanded2000_seed17'
run=root/'native512_dual_axis2000_seed17'
device='cuda:1'
torch.set_num_threads(4)
torch.backends.mha.set_fastpath_enabled(False)
torch.backends.cuda.matmul.allow_tf32=False
models=[]
for path,cls in [(reference,WholeSceneViT),(run,WholeSceneDualAxisViT)]:
    state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
    identity=state['identity']
    assert state['epoch']==10 and state['history'][-1]['updates']==1250
    layouts={tuple(tuple(b) for b in x['bands']):x['train_means'] for x in identity['layouts']}
    model=cls(layouts).to(device)
    model.load_state_dict(state['model']);model.eval()
    models.append((model,identity['patch_m']))
    del state
subset=read(FROZEN/'subset_expanded_2000_seed17.json')
members={r['id']:r for r in subset['parents']}
d=ExpandedTrainingData(ROOT)
panel=identity['panel']
big=max(read(run/'members.json'),key=lambda r:len(r['tiles']))
selection=[(next(members[k] for k in panel if members[k]['source']==source),band)
           for source,band in [('Core L8','SR_B4'),('Core S2','B04'),('FLAIR-HUB S2','B04')]]
selection.append((members[big['id']],'B04'))
titles=('Visible A input','Native truth','60 m Spatial ViT','60 m Dual-Axis','Spatial ViT error','Dual ViT error')
fig,axes=plt.subplots(len(selection),len(titles),figsize=(16,2.8*len(selection)),squeeze=False,layout='constrained')
for ax,title in zip(axes[0],titles):ax.set_title(title,fontsize=10)
records=[]
with torch.no_grad():
    for i,(row,key) in enumerate(selection):
        scale=subset['dn_scales'][row['domain']] if row['encoding']=='stored_dn' else 1.
        obs=d.load(row,'cpu')
        j=next(j for j,b in enumerate(obs.bands) if b.key==key)
        band=obs.bands[j]
        preds=[]; supports=[]
        for model,patch_m in models:
            scene=pack_scene(obs,scale,patch_m).to(device)
            keep,available=masks(scene,row['id'],0)[0]
            prediction,patch,slot,iy,ix=predict(model,scene,keep,available,obs.bands)[j]
            visible=keep.cpu()
            native_keep=visible[patch] if visible.ndim==1 else visible[patch,slot]
            supports.append(band.valid & ~native_keep)
            preds.append(prediction)
            if model is models[-1][0]:
                bounds=tile_bounds(scene)
        hidden=supports[0]
        assert torch.equal(hidden,supports[1])
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
        legal=band.valid[box]
        q=hidden[box]
        p0,p1=[p[box] for p in preds]
        assert q.any() and max(y.shape)<=512
        visible=legal & ~q
        e0,e1=p0-y,p1-y
        lo=min(float(y[legal].min()),float(p0[q].min()),float(p1[q].min()))
        hi=max(float(y[legal].max()),float(p0[q].max()),float(p1[q].max()),lo+1e-6)
        cap=max(float(torch.cat((e0[q].abs(),e1[q].abs())).quantile(.98)),1e-5)
        for j2,(z,mask) in enumerate(zip((y,y,p0,p1,e0,e1),(visible,legal,q,q,q,q),strict=True)):
            error=j2>=4
            im=axes[i,j2].imshow(np.ma.array(z.numpy(),mask=~mask.numpy()),interpolation='nearest',
                cmap='RdBu_r' if error else 'gray',vmin=-cap if error else lo,vmax=cap if error else hi)
            axes[i,j2].set_xticks([]);axes[i,j2].set_yticks([])
        fig.colorbar(im,ax=axes[i,4:].tolist(),shrink=.7)
        axes[i,0].set_ylabel(f'{row["source"]} / {key}\n{row["id"][-8:]} / {y.shape[0]}×{y.shape[1]}')
        records.append(dict(id=row['id'],source=row['source'],band=key,display_shape=list(y.shape),
            hidden_pixels=int(q.sum()),same_native_hidden_pixels=True,
            display_mse=dict(spatial=float(e0[q].square().mean()),dual=float(e1[q].square().mean())),
            subset='largest scene native output tile' if len(bounds)>1 else 'fixed TRAIN panel, whole scene'))
fig.suptitle('A spatial masking | identical hidden pixels | epoch 10 last')
fig.savefig(run/'A_predictions_comparison.png',dpi=150);plt.close(fig)
write_json(run/'A_predictions_comparison.json',dict(examples=records,train_only=True,reference=str(reference)))
print(json.dumps(records,ensure_ascii=False))

