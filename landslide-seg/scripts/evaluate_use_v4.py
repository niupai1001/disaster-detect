"""Public v3 denominators; four probes, only 10k/20k eligible."""
import argparse
from pathlib import Path
import numpy as np
import torch
import pretrain_use_v4 as pt
import run_use_v2 as rt
from audit_use_independent_data import sha
from evaluate_use_v3 import evaluate


def fixed_recall(probability,target):
    """Group tied scores before measuring threshold recall; no interpolation."""
    p=np.asarray(probability); y=np.asarray(target,dtype=bool)
    if p.shape!=y.shape or not np.isfinite(p).all(): raise ValueError('Invalid probability/target arrays')
    positive=int(y.sum()); negative=int((~y).sum())
    if not positive or not negative: return {'unmeasured_reason':'requires positive and negative valid pixels'}
    order=np.argsort(-p,kind='stable'); p=p[order]; y=y[order]
    ends=np.r_[np.flatnonzero(p[:-1]!=p[1:]),len(p)-1]
    recall=np.cumsum(y)[ends]/positive; fpr=np.cumsum(~y)[ends]/negative
    result={}
    for value in (.5,.8):
        available=np.flatnonzero(recall>=value)
        if not len(available): result[str(value)]={'unmeasured_reason':'target recall unreachable'}
        else:
            i=available[np.argmin(fpr[available])]
            result[str(value)]=dict(threshold=float(p[ends[i]]),actual_recall=float(recall[i]),fpr=float(fpr[i]))
    return result


@torch.no_grad()
def physical_errors(model,task,splits,stats,device,metadata_mode):
    result={}
    for sensor,rows in splits['PRETRAIN_VAL'].items():
        records=[]
        for index,row in enumerate(rows):
            original=pt.load_rows([row],stats,device); errors={}
            std=torch.as_tensor(stats[sensor]['std'],device=device)[None,:,None,None]
            for name in (('fine',) if task=='fine' else ('spectral','context')):
                values=[]
                for block in ((16,32,64) if name=='context' else (32,)):
                    view,hidden,summary=pt.task_view(original,name,800000+index*3+('spectral','fine','context').index(name),block=block,metadata_mode=metadata_mode)
                    if not hidden.any(): continue
                    pred=model(view,view['band_metadata'],view['known_mask'],summary)
                    valid=original['pixel_valid']&original['band_valid'][:,:,None,None]
                    error=pt.spatial_error(pred*std,original['x']*std,hidden,valid) if name=='context' else pt.mean_error(pred*std,original['x']*std,hidden)
                    values.append(float(error))
                if values: errors[name]=sum(values)/len(values)
            records.append(dict(file=row['file'],errors=errors))
        result[sensor]=records
    return dict(rows=result,units='original cache reflectance units before per-sensor standardisation; scale each band by its own TRAIN standard deviation')


def evaluate_job(job_id,checkpoint,device,metadata_mode='known'):
    from run_use_v4 import BASE,ROOT,matrix,load_encoder,CONFIG
    job=matrix()[job_id]; out=BASE/'runs'/job_id
    checkpoint=checkpoint or out/'last.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['step']!=job['steps'] and job['mode']=='probe': raise ValueError('Probe must finish 1000 updates')
    if job['mode']=='probe':
        source=BASE/f"runs/pretrain/u{job['point']}.pt"
        model=pt.initialize(device,job['task'],load_encoder(source))
        model.load_state_dict(saved['model'],strict=True)
        splits,stats,_,_=pt.inputs('multisensor')
        calibration_path=ROOT/'outputs/use_v3/engineering/calibration.json'
        result=evaluate(model,job['task'],splits,stats,rt.read(calibration_path),device,metadata_mode)
        result.update(public_denominator_sha256=sha(calibration_path),probe_sha256=sha(checkpoint),
            point=job['point'],raw_errors='normalized values multiplied by the attached public per-sensor denominator',
            public_coefficients=rt.read(calibration_path)['coefficients'])
        # Store raw standardised-reflectance errors, rather than hiding them behind scores.
        for sensor,values in result['sensors'].items():
            coeff=result['public_coefficients'][sensor]
            for row in values['rows']: row['raw_errors']={k:v*coeff[k] for k,v in row['errors'].items()}
        result['physical_errors']=physical_errors(model,job['task'],splits,stats,device,metadata_mode)
        model.use.verify()
    elif job['mode']=='segmentation':
        import yaml
        from landslide_seg.models.use_v4_reconstruction import Segmenter
        model=Segmenter(load_encoder(BASE/'runs/pretrain/best_unlabeled.pt')).to(device)
        model.load_state_dict(saved['model'],strict=True)
        config=yaml.safe_load(CONFIG.read_text()); _,val=rt.datasets(config)
        result=rt.downstream_evaluate(model,val,config,device,prediction_dir=out/'predictions'/checkpoint.stem)
        probabilities=[]; targets=[]
        for path in sorted((out/'predictions'/checkpoint.stem).glob('*.npz')):
            with np.load(path) as data:
                mask=data['valid'].astype(bool); probabilities.append(data['prob'][mask]); targets.append(data['mask'][mask])
        result['fixed_recall']=fixed_recall(np.concatenate(probabilities),np.concatenate(targets))
    else: raise ValueError('Use validation curves for reconstruction; selection only uses probes')
    result['checkpoint_sha256']=sha(checkpoint)
    suffix='' if metadata_mode=='known' else f'_{metadata_mode}'
    name=f'evaluation_{checkpoint.stem}{suffix}.json' if job['mode']=='segmentation' else f'evaluation{suffix}.json'
    rt.write_json(out/name,result)
    return result


def select(root,out):
    scores={}; denominator=None
    for step in (10000,20000):
        fine=rt.read(Path(root)/f'probe_{step}_fine/evaluation.json')
        context=rt.read(Path(root)/f'probe_{step}_context/evaluation.json')
        for value in (fine,context):
            if value.get('metadata_mode')!='known' or not value.get('selection_eligible'): raise ValueError('Only known metadata eligible')
            current=value['public_denominator_sha256']
            if denominator is not None and denominator!=current: raise ValueError('Public denominator mismatch')
            denominator=current
        scores[step]=sum((fine['sensors'][s]['means']['fine']+context['sensors'][s]['means']['spectral']+context['sensors'][s]['means']['context'])/3 for s in ('s2','landsat'))/2
    if not all(torch.isfinite(torch.tensor(v)) for v in scores.values()): raise ValueError('Nonfinite selection score')
    chosen=min(scores,key=lambda s:(scores[s],s)); source=Path(root)/f'pretrain/u{chosen}.pt'
    target=Path(root)/'pretrain/best_unlabeled.pt'
    if target.exists(): raise ValueError('Selected checkpoint exists; do not overwrite')
    rt.atomic_copy(source,target)
    result=dict(step=chosen,scores=scores,source_sha256=sha(source),best_unlabeled_sha256=sha(target),public_denominator_sha256=denominator,rule='sensor equal; Fine pixel/Context spectral/spatial equal; earlier tie')
    rt.write_json(out,result,True); return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--job',required=True); p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--metadata-mode',choices=['known','partial','unknown'],default='known')
    a=p.parse_args(); torch.set_num_threads(4); evaluate_job(a.job,a.checkpoint,a.device,a.metadata_mode)


if __name__=='__main__': main()
