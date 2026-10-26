"""Fixed-view per-sensor reconstruction diagnostics; never select with task labels."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from pretrain_use_v2 import ROOT,inputs,load_rows,task_view,initialize,masked_mae
from landslide_seg.models.use_v2 import canonical_context


def baseline(observation,statistics=None):
    x=observation['x']; v=observation['pixel_valid']
    if statistics is None:
        physical=x; mean=0; std=1
    else:
        mean=x.new_tensor(statistics['mean'])[None,:,None,None]
        std=x.new_tensor(statistics['std'])[None,:,None,None]
        physical=torch.where(v,x*std+mean,0)
    spectral=physical.sum(1,keepdim=True)/v.sum(1,keepdim=True).clamp_min(1)
    # Normalized local interpolation uses only visible observations.
    local=F.avg_pool2d(x,9,1,4)/F.avg_pool2d(v.float(),9,1,4).clamp_min(1/81)
    return (spectral.expand_as(x)-mean)/std,local


def calibrate(splits,stats,device='cpu'):
    coefficients={}; regressions={}; members={}
    for sensor,pool in splits['TRAIN'].items():
        indices=np.random.default_rng(1234).permutation(len(pool))[:32]
        rows=[pool[i] for i in indices]; members[sensor]=[r['file'] for r in rows]
        sums={task:[] for task in ('spectral','fine','context')}
        xy={task:[] for task in sums}
        for index,row in enumerate(rows):
            original=load_rows([row],stats,device)
            for ti,task in enumerate(sums):
                o,mask,_=task_view(original,task,700000+index*3+ti,metadata_mode='known')
                if not mask.any(): continue
                pred=baseline(o,stats[sensor])[0 if task=='spectral' else 1]
                sums[task].append(float(masked_mae(pred,original['x'],mask)))
                mu=pred.new_tensor(stats[sensor]['mean'])[None,:,None,None]
                sd=pred.new_tensor(stats[sensor]['std'])[None,:,None,None]
                left=(pred*sd+mu)[mask].cpu().numpy()[::31]
                right=(original['x']*sd+mu)[mask].cpu().numpy()[::31]
                xy[task].append((left,right))
        coefficients[sensor]={task:max(float(np.mean(values)),1e-6) for task,values in sums.items()}
        regressions[sensor]={}
        for task,values in xy.items():
            left=np.concatenate([v[0] for v in values]); right=np.concatenate([v[1] for v in values])
            design=np.stack((left,np.ones_like(left)),1).astype(np.float64)
            weights=np.linalg.solve(design.T@design+np.diag([1e-4,1e-8]),design.T@right)
            regressions[sensor][task]=weights.tolist()
    return dict(coefficients=coefficients,linear=regressions,members=members,seed=1234,
        fit_role='TRAIN',simple_baseline='spectral: visible physical reflectance mean; spatial: normalized 9x9 local interpolation',
        linear_rule='TRAIN-only affine regression on the simple prediction in physical reflectance; no band IDs',
        target_summary='visible-only standardized per-band mean and standard deviation; shared by unknown-target readout',
        loss_rule='sample/band-balanced standardized MAE divided by TRAIN baseline MAE, three tasks equally weighted')


def errors(pred,target,mask):
    count=mask.sum((-1,-2)); usable=count>0
    if not usable.any(): return None
    delta=torch.where(mask,pred-target,0)
    mae=(delta.abs().sum((-1,-2))/count.clamp_min(1))[usable].mean()
    mse=(delta.square().sum((-1,-2))/count.clamp_min(1))[usable].mean()
    diffs=[]
    for axis in (-1,-2):
        neighbor=mask.narrow(axis,1,mask.shape[axis]-1)&mask.narrow(axis,0,mask.shape[axis]-1)
        diff=pred.diff(dim=axis)-target.diff(dim=axis)
        if neighbor.any(): diffs.append(diff[neighbor].abs().mean())
    return dict(mae=float(mae),rmse=float(mse.sqrt()),
                adjacent_difference_mae=float(torch.stack(diffs).mean()) if diffs else None,
                hidden_pixels=int(mask.sum()))


def edge_error(pred,target,mask):
    dx=F.pad(target.diff(dim=-1).abs(),(0,1,0,0))
    dy=F.pad(target.diff(dim=-2).abs(),(0,0,0,1))
    gradient=dx+dy
    values=gradient[mask]
    if not len(values): return None
    edge=mask & (gradient>=torch.quantile(values,.9))
    return float((pred-target).abs()[edge].mean()) if edge.any() else None


def selection(result,calibration,tasks=('spectral','fine','context')):
    return float(np.mean([np.mean([result[s]['aggregate'][t]['mae']/calibration['coefficients'][s][t]
                                  for t in tasks]) for s in ('s2','landsat')]))


@torch.no_grad()
def evaluate(model,split,stats,device,limit=None,calibration=None,metadata_mode='known',tasks=('spectral','fine','context'),block=32):
    was_training=model.training; model.eval()
    result={}
    for sensor,rows in split.items():
        records=[]
        for index,row in enumerate(rows if limit is None else rows[:limit]):
            original=load_rows([row],stats,device)
            for task in tasks:
                ti=('spectral','fine','context').index(task)
                o,hidden,summary=task_view(original,task,900000+index*3+ti,block=block,metadata_mode=metadata_mode)
                if not hidden.any(): continue
                pred=model(o,o['band_metadata'],o['known_mask'],summary)
                simple=baseline(o,stats[sensor])[0 if task=='spectral' else 1]
                sd=pred.new_tensor(stats[sensor]['std'])[None,:,None,None]
                mu=pred.new_tensor(stats[sensor]['mean'])[None,:,None,None]
                record=dict(file=row['file'],task=task,model=errors(pred,original['x'],hidden),
                            physical=errors(pred*sd,original['x']*sd,hidden),baseline=errors(simple,original['x'],hidden),
                            edge_mae=edge_error(pred,original['x'],hidden),
                            bands={row['band_ids'][j]:errors(pred[:,j:j+1]*sd[:,j:j+1],original['x'][:,j:j+1]*sd[:,j:j+1],hidden[:,j:j+1])
                                   for j in range(len(row['band_ids'])) if hidden[:,j].any()})
                if calibration:
                    slope,intercept=calibration['linear'][sensor][task]
                    linear=((simple*sd+mu)*slope+intercept-mu)/sd
                    record['linear']=errors(linear,original['x'],hidden)
                records.append(record)
        aggregate={}
        for task in tasks:
            subset=[r for r in records if r['task']==task]
            if subset:
                aggregate[task]={key:sum(r[source]['mae'] for r in subset)/len(subset)
                                 for key,source in [('mae','model'),('baseline_mae','baseline')]}
        result[sensor]=dict(records=records,aggregate=aggregate)
    model.train(was_training)
    return result


@torch.no_grad()
def representations(encoder,split,stats,device,limit=None):
    encoder.eval(); result={}
    for sensor,pool in split.items():
        rows=pool if limit is None else pool[:limit]
        left=[]; right=[]; mean_left=[]; mean_right=[]
        for row in rows:
            batch=load_rows([row],stats,device)
            c=batch['x'].shape[1]
            for ids,collector,base in ((list(range(0,c,2)),left,mean_left),(list(range(1,c,2)),right,mean_right)):
                view={k:v[:,ids] if torch.is_tensor(v) and v.ndim>=2 and v.shape[1]==c else v for k,v in batch.items()}
                output=encoder(view)
                vectors=[]
                for feature,valid in zip(output['context'],output['valid']['context']):
                    feature=canonical_context(feature)
                    vectors.append((feature*valid[:,None]).sum((-1,-2))/valid.sum((-1,-2)).clamp_min(1)[:,None])
                collector.append(torch.cat(vectors,1).cpu())
                visible=view['pixel_valid']; x=view['x']
                # Scalar mean/std are the same capacity for both band views.
                vals=x[visible]; base.append(torch.stack((vals.mean(),vals.std(unbiased=False)))[None].cpu())
        def metrics(a,b):
            a=torch.cat(a).double(); b=torch.cat(b).double()
            sim=F.normalize(a,dim=1)@F.normalize(b,dim=1).T
            singular=torch.linalg.svdvals(torch.cat((a-a.mean(0),b-b.mean(0)),0))
            p=singular/singular.sum().clamp_min(1e-15)
            n=len(a)
            return dict(locations=n,recall1=float((sim.argmax(1)==torch.arange(n)).double().mean()),
                recall5=float((sim.topk(min(5,n),dim=1).indices==torch.arange(n)[:,None]).any(1).double().mean()),
                positive_cosine=float(sim.diag().mean()),negative_cosine=float(sim[torch.arange(n),torch.arange(n).roll(1)].mean()),
                variance=float(torch.cat((a,b)).var(0,unbiased=False).mean()),
                effective_rank=float(torch.exp(-(p*p.clamp_min(1e-15).log()).sum())),chance_recall1=1/n,
                chance_recall5=min(5,n)/n)
        result[sensor]=dict(encoder=metrics(left,right),mean_pooling=metrics(mean_left,mean_right),
                            files=[r['file'] for r in rows],pairing='two band subsets of the same observation; not cross-sensor pairing')
    return result


def area30(row,statistics,device='cpu'):
    """Area-resample a real georeferenced cache to a local metric 30m grid."""
    import rasterio
    from rasterio.warp import reproject,calculate_default_transform,Resampling,transform_bounds
    from rasterio.transform import Affine,array_bounds
    from landslide_seg.data.optical_observation import OpticalObservation,collate_optical
    from audit_use_independent_data import EXTERNAL,sha
    path=EXTERNAL/row['file']
    if sha(path)!=row['sha256']: raise ValueError('Cache changed')
    with np.load(path,allow_pickle=False) as data:
        x,v,m,k=(data[key].copy() for key in ('x','pixel_valid','metadata','known'))
    crs=row['crs']; source=Affine(*row['transform']); bounds=array_bounds(*x.shape[-2:],source)
    lon0,lat0,lon1,lat1=transform_bounds(crs,'EPSG:4326',*bounds)
    lon,lat=(lon0+lon1)/2,(lat0+lat1)/2
    if not -80<=lat<=84: raise ValueError('30m UTM diagnostic outside supported latitude')
    utm=f'EPSG:{(32600 if lat>=0 else 32700)+min(60,max(1,int((lon+180)//6)+1))}'
    dest,w,h=calculate_default_transform(crs,utm,x.shape[-1],x.shape[-2],*bounds,resolution=30)
    coverage=footprint_coverage(source,x.shape[-2:],crs,dest,(h,w),utm)
    result=np.zeros((len(x),h,w),np.float32); fraction=np.zeros_like(result)
    for band in range(len(x)):
        for original,target in ((np.where(v[band],x[band],0),result[band]),(v[band].astype(np.float32),fraction[band])):
            reproject(original,target,src_transform=source,src_crs=crs,dst_transform=dest,dst_crs=utm,
                      resampling=Resampling.average,init_dest_nodata=True)
    valid=fraction*coverage[None]>=.8
    result=np.where(valid,result/np.maximum(fraction,1e-8),0)
    mean,std=(np.asarray(statistics[key],dtype=np.float32)[:,None,None] for key in ('mean','std'))
    result=np.where(valid,(result-mean)/std,0).astype(np.float32)
    provenance=dict(crs=utm,transform=list(dest)[:6],source_crs=crs,source_transform=row['transform'],
                    source_bounds=list(bounds),target_gsd_m=30,minimum_valid_area=.8,
                    valid_fraction_min=float((fraction*coverage[None]).min()),valid_fraction_mean=float((fraction*coverage[None]).mean()),
                    footprint_area_m2=float(coverage.sum()*900),edge_densification_segments=64,
                    note='Projected envelope of source physical footprint; no resizing by array shape')
    return collate_optical([OpticalObservation(torch.from_numpy(result),tuple(row['band_ids']),torch.from_numpy(m),
                    torch.from_numpy(k),torch.ones(len(x),dtype=torch.bool),torch.from_numpy(valid),provenance)]).to(device),provenance


def footprint_coverage(source,source_shape,source_crs,target,target_shape,target_crs):
    """Fraction of each target cell inside the densified source footprint."""
    import shapely
    from pyproj import Transformer
    h,w=source_shape; th,tw=target_shape; t=np.linspace(0,1,65)
    pixels=np.concatenate((np.stack((w*t,np.zeros_like(t)),1),np.stack((np.full_like(t,w),h*t),1),
                           np.stack((w*(1-t),np.full_like(t,h)),1),np.stack((np.zeros_like(t),h*(1-t)),1)))
    world=np.array([source@tuple(point) for point in pixels])
    transform=Transformer.from_crs(source_crs,target_crs,always_xy=True)
    x,y=transform.transform(world[:,0],world[:,1]); polygon=shapely.Polygon(np.stack((x,y),1))
    yy,xx=np.meshgrid(np.arange(th),np.arange(tw),indexing='ij')
    left=target.c+xx*target.a; top=target.f+yy*target.e
    cells=shapely.box(left,top+target.e,left+target.a,top)
    return np.clip(shapely.area(shapely.intersection(cells,polygon))/abs(target.a*target.e),0,1)


def paired_bootstrap(left,right,repeats=2000):
    """Paired cache-level differences, never claim a geographic population CI."""
    values=[]
    for sensor in left:
        l={(r['file'],r['task']):r['model']['mae'] for r in left[sensor]['records']}
        r={(r['file'],r['task']):r['model']['mae'] for r in right[sensor]['records']}
        if set(l)!=set(r): raise ValueError('Paired evaluation membership differs')
        files=sorted({key[0] for key in l})
        differences=np.array([np.mean([l[k]-r[k] for k in l if k[0]==name]) for name in files])
        rng=np.random.default_rng(1234)
        means=[float(rng.choice(differences,len(differences),replace=True).mean()) for _ in range(repeats)]
        values.append(dict(sensor=sensor,mean=float(differences.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                           unit='paired cached observation',limitation='Not an independent-geography or multi-seed confidence interval'))
    return values


@torch.no_grad()
def full_diagnostics(model,split,stats,device,calibration,limit=None):
    base=evaluate(model,split,stats,device,limit,calibration,metadata_mode='known')
    result=dict(reconstruction=base,representations=representations(model.use,split,stats,device,limit),
        metadata={mode:evaluate(model,split,stats,device,limit,calibration,metadata_mode=mode)
                  for mode in ('partial','unknown')},
        block_sizes={str(size):evaluate(model,split,stats,device,limit,calibration,metadata_mode='known',tasks=('context',),block=size)
                     for size in (16,64)},
        interpretation='Unknown spectral targets omitted; all unknown metadata still receives spatial reconstruction. No real cross-sensor pairing.')
    result['area30']={}
    model.eval()
    for sensor,rows in split.items():
        records=[]
        for i,row in enumerate(rows if limit is None else rows[:limit]):
            try:
                original,provenance=area30(row,stats[sensor],device)
                observation,hidden,summary=task_view(original,'fine',950000+i,metadata_mode='known')
                prediction=model(observation,observation['band_metadata'],observation['known_mask'],summary)
                records.append(dict(file=row['file'],provenance=provenance,metrics=errors(prediction,original['x'],hidden),status='PASS'))
            except (ValueError,RuntimeError) as error:
                records.append(dict(file=row['file'],status='BLOCKED',reason=str(error)))
        result['area30'][sensor]=records
    return result


def efficiency(model,device='cpu',channels=(1,3,4,6,10,16),iterations=10):
    import time
    from torch import nn
    result=[]; model.eval()
    def synchronize():
        if torch.device(device).type=='cuda': torch.cuda.synchronize(device)
    for c in channels:
        torch.manual_seed(1234)
        x=torch.randn(1,c,128,128,device=device)
        o=dict(x=x,pixel_valid=torch.ones_like(x,dtype=torch.bool),band_valid=torch.ones(1,c,dtype=torch.bool,device=device),
               band_metadata=torch.zeros(1,c,3,device=device),known_mask=torch.zeros(1,c,3,dtype=torch.bool,device=device))
        summary=torch.stack((x.mean((-1,-2)),x.std((-1,-2))),-1)
        calls={'use':lambda:model.use(o),'end_to_end':lambda:model(o,o['band_metadata'],o['known_mask'],summary)}
        record=dict(channels=c,batch=1,shape=[128,128],precision='FP32',device=device,
                    use_parameters=sum(p.numel() for p in model.use.parameters()),
                    total_parameters=sum(p.numel() for p in model.parameters()))
        for name,call in calls.items():
            costs=[]; handles=[]
            def hook(module,arguments,output):
                if isinstance(module,nn.Linear): costs.append(output.numel()*module.in_features)
                elif isinstance(module,nn.Conv2d): costs.append(output.numel()*module.in_channels//module.groups*np.prod(module.kernel_size))
                else:
                    q,k=arguments[:2]; b,nq,d=q.shape; nk=k.shape[1]
                    costs.append(b*((2*nq+2*nk)*d*d+2*nq*nk*d))
            scope=model.use if name=='use' else model
            for module in scope.modules():
                if isinstance(module,(nn.Linear,nn.Conv2d,nn.MultiheadAttention)): handles.append(module.register_forward_hook(hook))
            with torch.no_grad(): call()
            for handle in handles: handle.remove()
            if torch.device(device).type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                for _ in range(5): call()
                synchronize(); times=[]
                for _ in range(iterations):
                    start=time.perf_counter(); call(); synchronize(); times.append((time.perf_counter()-start)*1000)
            record[name]=dict(forward_mac=int(sum(costs)),median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),
                peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None,
                mac_scope='Linear/Conv and explicit MHA projection/QK/AV products; excludes normalization, softmax, activation, masking and interpolation')
        # Operator profiler counts actual forward/backward matrix products it supports.
        model.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True,acc_events=True) as prof:
            prediction=model(o,o['band_metadata'],o['known_mask'],summary)
            prediction.square().mean().backward()
        record['training_profiled_mac']=sum(event.flops for event in prof.key_averages())//2
        record['training_profile_scope']='Actual forward+backward supported profiler operators only; fused attention may be uncounted, not a full training MAC total'
        result.append(record)
    return result


def plot_features(model,row,stats,out,device='cpu'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model.eval(); o=load_rows([row],stats,device)
    with torch.no_grad(): output=model.use(o)
    x=o['x'][0].cpu().numpy(); fine=output['fine'][0].cpu().numpy()
    figure,axes=plt.subplots(2,4,figsize=(16,8))
    rgb=x[[2,1,0]].transpose(1,2,0); lo,hi=np.percentile(rgb,[2,98],axis=(0,1)); rgb=np.clip((rgb-lo)/(hi-lo+1e-6),0,1)
    axes[0,0].imshow(rgb); axes[0,0].set_title('Original RGB (display stretch)')
    axes[0,1].imshow(output['valid']['fine'][0].cpu()); axes[0,1].set_title('Original-grid support')
    for i in (0,1):
        axes[0,i+2].imshow(fine[i]); axes[0,i+2].set_title(f'Fine channel {i}')
    for i,feature in enumerate(output['context']):
        axes[1,i].imshow(feature[0].float().norm(dim=0).cpu()); axes[1,i].set_title(f'Context scale 1/{4*2**i} norm')
    for ax in axes.flat: ax.axis('off')
    figure.suptitle('Top: imagery, validity, two Fine channels. Bottom: four Context scales. Feature intensity is not a semantic class.')
    figure.tight_layout(); figure.savefig(out,dpi=140); plt.close(figure)


def geographic_bootstrap(left,right,repeats=2000):
    from pyproj import Transformer
    l={r['sample_id']:r for r in left['tiles']}; r={v['sample_id']:v for v in right['tiles']}
    if set(l)!=set(r): raise ValueError('Downstream paired membership differs')
    groups={}
    for sid,entry in l.items():
        bbox=entry['bbox']; transform=Transformer.from_crs(entry['crs'],'EPSG:6933',always_xy=True)
        x,y=transform.transform((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)
        key=(entry['event_id'],int(np.floor(x/10000)),int(np.floor(y/10000)))
        groups.setdefault(key,[]).append(sid)
    events=sorted({key[0] for key in groups}); rng=np.random.default_rng(1234); differences=[]
    def iou(entries,ids):
        tp=sum(entries[sid]['tp'] for sid in ids); fp=sum(entries[sid]['fp'] for sid in ids); fn=sum(entries[sid]['fn'] for sid in ids)
        return tp/(tp+fp+fn) if tp+fp+fn else 0.
    for _ in range(repeats):
        delta=[]
        for event in events:
            keys=[key for key in groups if key[0]==event]
            ids=[sid for index in rng.integers(len(keys),size=len(keys)) for sid in groups[keys[index]]]
            delta.append(iou(l,ids)-iou(r,ids))
        differences.append(np.mean(delta))
    return dict(delta_event_mean_iou=left['event_mean_iou']-right['event_mean_iou'],
        ci95=np.quantile(differences,[.025,.975]).tolist(),replicates=repeats,seed=1234,groups=len(groups),
        rule='Paired 10km EPSG:6933 centroid groups resampled within each DEV event, then events equally weighted',
        limitation='Conditional on two reused development events and one seed, not a cross-region population significance claim')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--variant',required=True,choices=('mean','attention'))
    p.add_argument('--checkpoint',required=True); p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu'); p.add_argument('--limit',type=int)
    a=p.parse_args(); torch.set_num_threads(4)
    splits,stats,_,_=inputs('multisensor'); model=initialize(a.variant,a.device)
    checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    model.load_state_dict(checkpoint.get('model',checkpoint),strict=True)
    result=evaluate(model,splits['PRETRAIN_VAL'],stats,a.device,a.limit)
    path=Path(a.out)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__': main()
