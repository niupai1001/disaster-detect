"""CPU-only legal linear spatial baselines on the exact fixed A validation mask."""
import ssl,os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]/'earth-representation'
RELEASE=(ROOT/'outputs/encoder_v3/hierarchical/release_no_enmap_v1').resolve()
sys.path.insert(0,str(RELEASE/'runtime/src'))
from encoder_v3.hierarchical.train import load_data,build_model,sample_from_dataset,identity
from encoder_v3.hierarchical.geometry import partitions,area_map


def main():
    out=ROOT/'outputs/encoder_v3/hierarchical/reconstruction_research/native_baselines_v1';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);config=json.loads((RELEASE/'formal_run/config.json').read_text());sets,resolver=load_data(config)
    binding=json.loads((RELEASE/'formal_run/identity.json').read_text());assert identity(config,sets,resolver)==binding
    model=build_model(config);ds=sets['VAL'];official=json.loads((RELEASE/'formal_run/validation_232.json').read_text());records={r['id']:r for r in official['members']}
    names=('visible_mean','nearest','idw8','local_affine8');members=[];start=time.perf_counter()
    for index,row in enumerate(ds.rows):
        sample=sample_from_dataset(ds,index,resolver,'cpu');geo,parts,_=partitions(tuple(c.grid for c in sample.channels),config['patch_metres'])
        maps={c.grid:area_map(c.grid,geo,parts[2],'cpu') for c in sample.channels}
        keeps,targets,_=model.masks(sample,(geo,parts,[(None,None,maps[c.grid]) for c in sample.channels],[],[]),'A',config['validation_seed'],0)
        metrics={name:[] for name in names};distances=[];fallback=0
        for c,keep,mask in zip(sample.channels,keeps,targets):
            if not bool(mask.any()):continue
            values=c.values.numpy().ravel().astype(np.float64);visible=keep.numpy().ravel();hidden=mask.numpy().ravel();truth=values[hidden]
            if not visible.any():
                prediction={name:np.zeros_like(truth) for name in names};fallback+=1
            else:
                h,w=c.grid.shape;yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');a,_,ox,_,e,oy=c.grid.affine
                # Scene-relative physical pixel centers; target values never enter predictions.
                points=np.stack(((xx.ravel()+.5)*a,(yy.ravel()+.5)*e),-1)
                source=points[visible];query=points[hidden];observed=values[visible]
                k=min(8,len(source));distance,ids=cKDTree(source).query(query,k=k,workers=1)
                if k==1:distance=distance[:,None];ids=ids[:,None]
                weights=1/np.maximum(distance,1e-12)**2;weights/=weights.sum(1,keepdims=True);neighbors=observed[ids]
                delta=(source[ids]-query[:,None])/min(c.grid.gsd_xy)
                design=np.concatenate((np.ones((*delta.shape[:2],1)),delta),axis=-1)
                gram=np.einsum('nki,nk,nkj->nij',design,weights,design)
                gram+=np.diag([1e-8,1e-2,1e-2])[None]
                rhs=np.einsum('nki,nk,nk->ni',design,weights,neighbors)
                local=np.linalg.solve(gram,rhs[...,None])[...,0,0]
                prediction={'visible_mean':np.full_like(truth,observed.mean()),'nearest':neighbors[:,0],
                    'idw8':(weights*neighbors).sum(1),'local_affine8':local}
                distances.append(float((distance[:,0]>256.).mean()))
                assert np.allclose(weights.sum(1),1.) and not np.any(visible&hidden)
            for name in names:
                assert np.isfinite(prediction[name]).all();metrics[name].append(float(np.mean((prediction[name]-truth)**2)))
        members.append({'id':row['id'],'sensor':row['sensor'],'model_mse':records[row['id']]['loss'],
            **{name+'_mse':float(np.mean(metrics[name])) for name in names},'no_visible_channel_fallbacks':fallback,
            'nearest_visible_farther_than_256m_fraction_channel_mean':float(np.mean(distances)) if distances else None})
        if (index+1)%50==0:print('NATIVE_BASELINE',index+1,flush=True)
    means={k:float(np.mean([m[k] for m in members])) for k in ('model_mse',*(name+'_mse' for name in names))}
    assert len(members)==383 and abs(means['model_mse']-official['metric'])<1e-10
    old=json.loads((ROOT/'outputs/encoder_v3/hierarchical/a_only/diagnosis_node_525_v3/full_val_visible_mean.json').read_text())
    assert abs(means['visible_mean_mse']-old['visible_mean_mse'])<1e-7 and not torch.cuda.is_initialized()
    report={'status':'PASS','identity':binding,'observations':383,'means':means,'members':members,'elapsed_seconds':time.perf_counter()-start,
        'model_checkpoint_cursor':232,'optimizer_updates':0,'gpu_used':False,'k':8,
        'method':'Same official QA/native normalized targets/A seed1729 epoch0. Only valid visible values are used. k8 physical-distance nearest/IDW are fixed linear maps of visible data; local affine weighted least-squares uses per-query3x3 system with fixed ridge[1e-8,.01,.01]. No target fitting, clipping, resizing or learned calibration. Pixel mean, channel mean, observation mean match MAE. Models compared on identical383 observations.',
        'limits':'A hides every band in geographic patches; same-position spectral interpolation cannot be used. These are spatial baselines, not proof unique fine texture is identifiable. Single fixed research panel; full-source frozen checkpoint/inputs unchanged.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'by_source':{}}
    for sensor in ('S2','L8'):
        group=[m for m in members if m['sensor']==sensor]
        report['by_source'][sensor]={'observations':len(group),**{k:float(np.mean([m[k] for m in group])) for k in means},
            'model_worse_than_idw8_count':sum(m['model_mse']>m['idw8_mse'] for m in group)}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('NATIVE_BASELINES_PASS',json.dumps(means),flush=True)


if __name__=='__main__':main()
