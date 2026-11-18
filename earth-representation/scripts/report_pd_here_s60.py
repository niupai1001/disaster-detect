"""Plot the matched-budget comparison and measure final prediction sensitivity."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_pd_here_s60 import BASE, Study, TASKS, models, spatial_vit, compare_records
from train_pd_here_r3_texture_study import read, write_json
from pd_here_geotoken_dualvit_r3.data import unpatchify
from pd_here_s60.model import inputs60, retile
from diagnose_pd_here_r3_texture import hidden_prediction_effect
from pd_here_geotoken_dualvit_r3.evaluation import reduce_records


@torch.no_grad()
def sensitivity(model,values,qa,view,task):
    keep,bands=view
    x60,q60,k60=inputs60(values,qa,keep)
    generator=torch.Generator(device=keep.device).manual_seed(17)
    mean=[];shuffle=[]
    for j,(x,q) in enumerate(zip(x60,q60,strict=True)):
        active=q & k60[:,:,None,None] & bands[:,j,None,None,None]
        m=torch.where(active,x,0.).flatten(1).sum(-1)/active.flatten(1).sum(-1).clamp_min(1)
        mean.append(retile(torch.where(active,m[:,None,None,None],x),8,4))
        changed=x.clone()
        for i,p in active.flatten(2).any(-1).nonzero().tolist():
            valid=active[i,p];z=x[i,p][valid]
            changed[i,p][valid]=z[torch.randperm(len(z),device=z.device,generator=generator)]
        shuffle.append(retile(changed,8,4))
    reference=model(values,qa,*view)['prediction']
    widths=[x.shape[-1]*x.shape[-2] for x in values]
    def native(pred):
        return torch.cat([retile(t.reshape_as(x),4,8).flatten(2)
                          for t,x in zip(pred.split(widths,-1),values,strict=True)],-1)
    out={}
    for label, perturbation in [('mean',mean),('shuffle60',shuffle)]:
        pred=model(tuple(perturbation),qa,*view)['prediction']
        rows=hidden_prediction_effect(native(pred),native(reference),x60,q60,k60,bands,task)
        compact=[{k:v for k,v in r.items() if v is not None} for r in rows]
        metrics,counts=reduce_records(compact,counts=True)
        out[label]=dict(metrics=metrics,valid_parents=counts,records=rows)
    return out


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,help='Completed matched run; compare its S60 Dual-Axis arm to spatial ViT')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    run=args.run; status=read(run/'status.json')
    if status['status']!='COMPLETED':raise ValueError('Matched training is incomplete')
    history=read(run/'history.json'); last=history[-1]
    identity=read(run/'identity.json')
    if args.reference:
        reference=read(args.reference/'history.json')
        previous=read(args.reference/'identity.json')
        for key in ('data','layout_hash','exposure0_target_sha256','mask_source_sha256','seed','batch','lr','weight_decay','precision'):
            if previous[key]!=identity[key]:raise ValueError(f'Unmatched reference field: {key}')
        if read(args.reference/'status.json')['status']!='COMPLETED' or len(reference)!=len(history):
            raise ValueError('Reference budget differs')
        for old,new in zip(reference,history,strict=True):
            if old['updates']!=new['updates']:raise ValueError('Reference update count differs')
            new['arms']['s60']=old['arms']['s60']
        checkpoints={'s60':args.reference/'s60_last.pt','vit':run/'vit_last.pt'}
        labels={'s60':'S60 Dual-Axis','vit':'S60 Spatial ViT'}
    else:
        checkpoints={name:run/f'{name}_last.pt' for name in ('r3','s60')}
        labels={'r3':'Previous DualViT','s60':'S60 Dual-Axis'}
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,task in enumerate(TASKS):
        for col,key in enumerate(('mse','within_patch_centered_mse','gradient_mse')):
            ax=axes[row,col]
            for name,label in labels.items():
                ax.plot([x['epoch'] for x in history],[x['arms'][name]['val'][task]['all']['metrics'][key] for x in history],label=label)
            ax.set(xlabel='Epoch',title=f'{task} / {key}',yscale='log');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Matched native-target development VAL')
    fig.savefig(run/'curves.png',dpi=150);plt.close(fig)
    study=Study(BASE/'texture_study',args.device)
    stats,_,_=study.train_references()
    arms=models(study,stats)
    if args.reference:
        arms={'s60':arms['s60'],'vit':spatial_vit(study,stats)}
    for name,(model,_) in arms.items():
        state=torch.load(checkpoints[name],weights_only=True,map_location='cpu')
        assert state['epoch']==last['epoch'] and state['identity']['data']==study.identity
        model.load_state_dict(state['model']);model.eval()
    examples=read(BASE/'subset700_10epochs/predictions_val/examples.json')['examples']
    by_id={r['id']:r for r in study.rows['val']}
    measurements={}
    for ti,task in enumerate(TASKS):
        rows=[by_id[x['id']] for x in examples[task]]
        values,qa,views=study.inputs(rows,0); view=views[ti]
        fig,axes=plt.subplots(len(rows),3,figsize=(10,3.3*len(rows)),squeeze=False,layout='constrained')
        for ax,title in zip(axes[0],('Truth',*labels.values())):ax.set_title(title)
        predictions={name:model(values,qa,*view)['prediction'] for name,(model,_) in arms.items()}
        measurements[task]=dict(members=[r['id'] for r in rows],arms={})
        for name,(model,_) in arms.items():
            measurements[task]['arms'][name]=dict(
                sensitivity=sensitivity(model,values,qa,view,task),
                metrics=compare_records(predictions[name],values,qa,*view,task))
        sizes=[x.shape[-1]*x.shape[-2] for x in values]
        for i,row in enumerate(rows):
            j=study.layout.band_keys.index(examples[task][i]['key']) if task=='A' else int((~view[1][i]).nonzero().item())
            truth=values[j][i];valid=qa[j][i]
            hidden=valid & ((~view[0][i])[:,None,None] if task=='A' else True)
            imgs=[unpatchify(truth),*(unpatchify(predictions[n].split(sizes,-1)[j][i].reshape_as(truth)) for n in labels)]
            use=[unpatchify(valid),unpatchify(hidden),unpatchify(hidden)]
            lo=min(float(x[q].min()) for x,q in zip(imgs,use));hi=max(float(x[q].max()) for x,q in zip(imgs,use))
            for ax,x,q in zip(axes[i],imgs,use):
                ax.imshow(np.ma.array(x.cpu(),mask=~q.cpu().numpy()),cmap='gray',vmin=lo,vmax=max(hi,lo+1e-6))
                ax.set_xticks([]);ax.set_yticks([])
            axes[i,0].set_ylabel(row['id'][-8:]+' / '+study.layout.band_keys[j])
        fig.suptitle(f'{task} / epoch {last["epoch"]} last / identical native target')
        fig.savefig(run/f'{task}_predictions.png',dpi=150);plt.close(fig)
    write_json(run/'fixed_predictions_metrics.json',measurements)
    comparison = dict(status=status,last=last)
    if args.reference:
        comparison['reference']=str(args.reference)
    else:
        comparison['gate']=read(run/'epoch10_gate.json')
    write_json(run/'summary.json',comparison)
    print('Reports complete',flush=True)


if __name__=='__main__':
    main()
