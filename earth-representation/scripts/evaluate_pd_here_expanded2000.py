from pathlib import Path
import hashlib,sys,json,math
sys.path.insert(0,str(Path('scripts').resolve()))
import torch
from report_pd_here_s60_whole_scene import ROOT,FROZEN,ExpandedTrainingData,MODELS,pack_scene,masks,predict,read,write_json
from pd_here_s60.whole_scene import native_axis

ROOT_RUN=ROOT/'outputs/pd_here_s60'
OUT=ROOT/'outputs/reports/pd_here_s60/expanded2000_comparison'
NAMES={
 'S60 spatial':'native512_expanded2000_seed17',
 'S60 Dual-Axis':'native512_dual_axis2000_seed17',
 'R3 dual / 2 global':'native512_legacy_dual2000_v2_seed17',
 'R3 dual / 1 global':'native512_legacy_global1_2000_seed17',
 'R3 spectral / 2 global':'native512_legacy_spectral_only_2000_seed17',
}

def scalar(z):return float(z)
def corr(a,b,eps=1e-6):
    a,b=a-a.mean(),b-b.mean()
    aa,bb=a.square().mean().sqrt(),b.square().mean().sqrt()
    return scalar((a*b).mean()/(aa*bb)) if min(aa,bb)>=eps else None

def metrics(y,p,q,band,origin):
    a,b=y[q].double(),p[q].double()
    error=b-a
    mse=error.square().mean()
    variance=(a-a.mean()).square().mean()
    result=dict(mse=scalar(mse),rmse=scalar(mse.sqrt()),mae=scalar(error.abs().mean()),
                signed_bias=scalar(error.mean()),bias_squared=scalar(error.mean().square()),
                centered_mse=scalar((error-error.mean()).square().mean()),
                r2=scalar(1-mse/variance) if variance>=1e-12 else None,
                pixel_correlation=corr(a,b))
    dx,_,x,_,dy,top=band.grid.affine
    patch_axes={}
    for meters in (60,120):
        iy,ry=native_axis(y.shape[0],-dy,origin[1]-top,math.ceil(meters/-dy-1e-8),meters)
        ix,rx=native_axis(y.shape[1],dx,x-origin[0],math.ceil(meters/dx-1e-8),meters)
        patch_axes[meters]=(iy,ix)
        width=int(ix.max())+1
        ids=(iy[:,None]*width+ix)[q]
        count=torch.bincount(ids)
        legal=count.clamp_min(1)
        def centered(v):
            sums=torch.zeros(len(count),dtype=torch.double).scatter_add_(0,ids,v)
            return v-(sums/legal)[ids]
        ac,bc=centered(a),centered(b)
        result[f'patch{meters}_centered_mse']=scalar((bc-ac).square().mean())
        px,tx=bc.square().mean().sqrt(),ac.square().mean().sqrt()
        result[f'patch{meters}_pred_texture_rms']=scalar(px)
        result[f'patch{meters}_truth_texture_rms']=scalar(tx)
        result[f'patch{meters}_texture_amplitude_ratio']=scalar(px/tx) if tx>=1e-6 else None
        result[f'patch{meters}_texture_correlation']=corr(bc,ac)
        ph,pw=math.ceil(meters/-dy-1e-8),math.ceil(meters/dx-1e-8)
        complete=count==ph*pw
        if int(complete.sum())>=2:
            use=complete[ids]
            index=complete.long().cumsum(0)[ids[use]]-1
            slot=(ry[:,None]*pw+rx)[q][use]
            for label,val in [('prediction',b),('truth',a)]:
                z=torch.zeros(int(complete.sum()),ph*pw,dtype=torch.double)
                z[index,slot]=val[use]
                z=z-z.mean(1,keepdim=True)
                energy=z.square().mean()
                result[f'patch{meters}_{label}_template']=scalar(z.mean(0).square().mean()/energy) if energy>1e-20 else None
    predicted=[]; target=[]; seams={60:[],120:[]}
    for dim,mask in ((0,q[1:]&q[:-1]),(1,q[:,1:]&q[:,:-1])):
        predicted.append(p.diff(dim=dim)[mask].double())
        target.append(y.diff(dim=dim)[mask].double())
        for meters,(iy,ix) in patch_axes.items():
            crossed=(iy.diff()!=0)[:,None] if dim==0 else (ix.diff()!=0)[None]
            seams[meters].append((mask&crossed)[mask])
    u,v=torch.cat(predicted),torch.cat(target)
    counts=dict(pixels=int(q.sum()),gradient_pairs=len(u))
    if len(u):
        et=(u-v).square()
        tx=v.square().mean().sqrt()
        result.update(gradient_mse=scalar(et.mean()),gradient_correlation=corr(u,v),
                      gradient_amplitude_ratio=scalar(u.square().mean().sqrt()/tx) if tx>=1e-6 else None)
        # Fixed per-target top quartile based on truth magnitude; ties are retained.
        high=v.abs()>=v.abs().quantile(.75)
        counts['high_texture_pairs']=int(high.sum())
        result.update(high_texture_gradient_mse=scalar(et[high].mean()),
                      high_texture_gradient_correlation=corr(u[high],v[high]))
        for meters,parts in seams.items():
            seam=torch.cat(parts)
            counts[f'patch{meters}_boundary_pairs']=int(seam.sum())
            for name,sub in [('boundary',seam),('interior',~seam)]:
                if sub.any():result[f'patch{meters}_{name}_gradient_mse']=scalar(et[sub].mean())
    return result,counts

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    d=ExpandedTrainingData(ROOT)
    subset=read(FROZEN/'subset_expanded_2000_seed17.json')
    rows={r['id']:r for r in subset['parents']}
    reference=None; all_records=[]
    for name,runname in NAMES.items():
        run=ROOT_RUN/runname
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        ident=state['identity']
        assert state['epoch']==10 and state['history'][-1]['updates']==1250
        model=MODELS[ident['model_id']]({tuple(tuple(x) for x in p['bands']):p['train_means']
                                      for p in ident['layouts']}).to('cuda:1').eval()
        model.load_state_dict(state['model'])
        del state
        run_masks={}
        for key in ident['panel']:
            row=rows[key]; obs=d.load(row,'cpu')
            scale=subset['dn_scales'][row['domain']] if row['encoding']=='stored_dn' else 1.
            scene=pack_scene(obs,scale,ident['patch_m']).to('cuda:1')
            for task,(keep,bands) in zip(('A','B1'),masks(scene,key,0),strict=True):
                predictions=predict(model,scene,keep,bands,obs.bands)
                keep_cpu=keep.cpu()
                for j,((p,patch,slot,_,_),band) in enumerate(zip(predictions,obs.bands,strict=True)):
                    if task=='B1' and bands[j]:continue
                    vis=keep_cpu[patch] if keep_cpu.ndim==1 else keep_cpu[patch,slot]
                    hidden=band.valid & (~vis if task=='A' else True)
                    if not hidden.any():continue
                    run_masks[(key,task,band.key)]=hashlib.sha256(hidden.numpy().tobytes()).hexdigest()
                    result,counts=metrics(band.values/scale,p,hidden,band,scene.origin)
                    all_records.append(dict(model=name,run=runname,parent=key,task=task,source=row['source'],
                                            domain=row['domain'],band=band.key,metrics=result,counts=counts))
        if reference is None:reference=run_masks
        else:assert run_masks==reference, name
        print(name,'target bands',len(run_masks),'matched',flush=True)
        del model
    write_json(OUT/'native_metrics_by_parent_band.json',dict(scope='five completed checkpoints; 22 fixed TRAIN parents; same native hidden support',records=all_records))

if __name__=='__main__':main()
