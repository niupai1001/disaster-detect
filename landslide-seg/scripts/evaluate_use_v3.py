"""Fixed known-metadata PRETRAIN_VAL evaluation of unified frozen probes."""
import argparse
import json
from pathlib import Path
import torch
import pretrain_use_v3 as pt
from pretrain_use_v2 import task_view
from audit_use_independent_data import sha


@torch.no_grad()
def evaluate(model,task,splits,stats,calibration,device,metadata_mode='known'):
    model.eval(); sensors={}
    for sensor,rows in splits['PRETRAIN_VAL'].items():
        values=[]
        for index,row in enumerate(rows):
            original=pt.load_rows([row],stats,device)
            tasks=('fine',) if task=='fine' else ('spectral','context')
            errors={}; identifiable=None
            for name in tasks:
                blocks=(16,32,64) if name=='context' else (32,)
                result=[]
                for block in blocks:
                    view,hidden,summary=task_view(original,name,800000+index*3+('spectral','fine','context').index(name),block=block,metadata_mode=metadata_mode)
                    if name=='spectral': identifiable=float((view['known_mask'][...,0]&view['band_valid']).sum()/view['band_valid'].sum().clamp_min(1))
                    if not hidden.any(): continue
                    pred=model(view,view['band_metadata'],view['known_mask'],summary)
                    valid=original['pixel_valid']&original['band_valid'][:,:,None,None]
                    error=pt.spatial_error(pred,original['x'],hidden,valid) if name=='context' else pt.mean_error(pred,original['x'],hidden)
                    result.append(float(error)/calibration['coefficients'][sensor][name])
                if result: errors[name]=sum(result)/len(result)
            values.append(dict(file=row['file'],errors=errors,identifiable_spectral='spectral' in errors,identifiable_band_fraction=identifiable,unmeasured_reason='no identifiable hidden spectral targets' if task=='context' and 'spectral' not in errors else None))
        sensors[sensor]=dict(rows=values,means={name:sum(r['errors'][name] for r in values if name in r['errors'])/sum(name in r['errors'] for r in values) if any(name in r['errors'] for r in values) else None for name in tasks})
    model.use.verify()
    return dict(sensors=sensors,task=task,metadata_mode=metadata_mode,selection_eligible=metadata_mode=='known',
        limitation='Single-probe evaluation; combine Fine pixel and Context spectral/spatial equally for selection')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--encoder',choices=['v2','v3'],required=True)
    p.add_argument('--source',type=Path); p.add_argument('--probe',type=Path,required=True); p.add_argument('--task',choices=['fine','context'],required=True)
    p.add_argument('--device',default='cuda:1'); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--metadata-mode',choices=['known','partial','unknown'],default='known')
    args=p.parse_args()
    from run_use_v3 import load_encoder
    model=pt.initialize(args.device,args.task,load_encoder(args.encoder,args.source))
    saved=torch.load(args.probe,map_location='cpu',weights_only=False); model.load_state_dict(saved['model'],strict=True)
    splits,stats,_,_=pt.inputs('multisensor')
    calibration=json.loads((pt.ROOT/'outputs/use_v3/engineering/calibration.json').read_text())
    result=evaluate(model,args.task,splits,stats,calibration,args.device,args.metadata_mode)
    result['probe_sha256']=sha(args.probe)
    from run_use_v2 import write_json
    write_json(args.out,result,True)



def select(root,out):
    """Select v3 by fixed unified probe scores; initialization and 5k excluded."""
    from run_use_v2 import read,write_json,atomic_copy
    from pathlib import Path
    scores={}
    for step in (10000,20000,30000):
        fine=read(Path(root)/f'probe_v3_{step}_fine/evaluation.json')
        context=read(Path(root)/f'probe_v3_{step}_context/evaluation.json')
        if fine.get('metadata_mode')!='known' or context.get('metadata_mode')!='known': raise ValueError('Selection requires known metadata')
        scores[step]=sum((fine['sensors'][s]['means']['fine']+context['sensors'][s]['means']['spectral']+context['sensors'][s]['means']['context'])/3 for s in ('s2','landsat'))/2
    chosen=min(scores,key=lambda s:(scores[s],s))
    source=Path(root)/f'pretrain/u{chosen}.pt'
    atomic_copy(source,Path(root)/'pretrain/best_unlabeled.pt')
    result=dict(step=chosen,scores=scores,source_sha256=sha(source),rule='sensor equal; spectral/pixel/spatial equal; earlier tie')
    write_json(out,result,True)
    return result

if __name__=='__main__': main()
