"""Compare completed R3 study arms and render the original six windows."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_pd_here_r3_texture_study import BASE, Study, TASKS, VARIANTS, initial_model, read, write_json
from pd_here_geotoken_dualvit_r3.data import unpatchify
from diagnose_pd_here_r3_texture import intervention


def select(histories, reference='baseline'):
    keys = ('centered_mse','gradient_mse','mse')
    metrics = {name:{task:history[-1]['val'][task]['all']['metrics'] for task in TASKS}
               for name,history in histories.items()}
    average = {name:tuple(sum(values[t][k] for t in TASKS)/2 for k in keys)
               for name,values in metrics.items()}
    base = metrics[reference]
    qualifying = [name for name in histories if name != reference
                  if average[name][0]<=.95*average[reference][0]
                  and average[name][1]<average[reference][1]
                  and all(metrics[name][task][key]<=1.05*base[task][key]
                          for task in TASKS for key in keys)]
    return min(qualifying,key=average.get) if qualifying else reference, average


@torch.no_grad()
def prediction_plots(study, directory, variants, destination):
    models = {}
    for name in variants:
        state = torch.load(directory/name/'last.pt',map_location='cpu',weights_only=True)
        model,_ = initial_model(study.layout,name,state['identity']['statistics'],study.device)
        model.load_state_dict(state['model'])
        models[name] = model.eval()
    rows = {r['id']:r for r in study.rows['val']}
    examples = read(BASE/'subset700_10epochs/predictions_val/examples.json')['examples']
    offsets = np.cumsum([0,*[h*w for h,w in study.layout.native_shapes]])
    texture = {}
    for task in TASKS:
        texture[task] = {}
        samples = [rows[e['id']] for e in examples[task]]
        values,qa,views = study.inputs(samples,0)
        for name, model in models.items():
            texture[task][name] = intervention(model,values,qa,views[TASKS.index(task)],task)
        fig,axes = plt.subplots(3,8,figsize=(18,7.8),layout='constrained')
        for column,title in enumerate(['Visible input','Truth',*[
                item for name in models for item in (name,'Signed error')]]):
            axes[0,column].set_title(title,fontsize=10)
        for i,case in enumerate(examples[task]):
            values,qa,views = study.inputs([rows[case['id']]],0)
            keep,bands = views[TASKS.index(task)]
            j = study.layout.band_keys.index(case['key'])
            if task == 'B1':
                j = int((~bands[0]).nonzero().item())
            def native(x):
                return unpatchify(x).cpu().numpy()
            truth = native(values[j][0])
            valid = native(qa[j][0])
            mask = valid & (native((~keep[0])[:,None,None].expand_as(qa[j][0])) if task=='A' else True)
            predictions = []
            for mod in models.values():
                p = mod(values,qa,keep,bands)['prediction'][0,:,offsets[j]:offsets[j+1]]
                predictions.append(native(p.reshape_as(values[j][0])))
            lo = min(truth[valid].min(),*(p[mask].min() for p in predictions))
            hi = max(truth[valid].max(),*(p[mask].max() for p in predictions),lo+1e-4)
            cap = max(1e-4,*(np.abs(p-truth)[mask].max() for p in predictions))
            gray = dict(cmap=plt.get_cmap('gray').with_extremes(bad='#ccc'),vmin=lo,vmax=hi,interpolation='nearest')
            if task=='A':
                axes[i,0].imshow(np.ma.array(truth,mask=~(valid&~mask)),**gray)
            else:
                rgb = [study.layout.band_keys.index(b) for b in ('B04','B03','B02')]
                image = np.stack([native(values[k][0]) if bands[0,k] else np.zeros((48,48))
                                  for k in rgb],-1)
                axes[i,0].imshow(np.clip(image/.3,0,1)**(1/2.2),interpolation='nearest')
            axes[i,1].imshow(np.ma.array(truth,mask=~valid),**gray)
            for m,p in enumerate(predictions):
                axes[i,2+m*2].imshow(np.ma.array(p,mask=~mask),**gray)
                error = axes[i,3+m*2].imshow(np.ma.array(p-truth,mask=~mask),
                    cmap=plt.get_cmap('RdBu_r').with_extremes(bad='#ccc'),vmin=-cap,vmax=cap,
                    interpolation='nearest')
                fig.colorbar(error,ax=axes[i,3+m*2],fraction=.046)
            axes[i,0].set_ylabel(case['key']+' / '+case['id'][-8:])
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f'R3 texture study, epoch 30 | {task} | Fixed development VAL')
        fig.savefig(destination/f'{task}_predictions.png',dpi=150)
        plt.close(fig)
    write_json(destination/'final_texture_checks.json',texture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=BASE/'texture_study')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--variants',nargs='+',default=list(VARIANTS))
    parser.add_argument('--reference',default='baseline')
    parser.add_argument('--report-dir',type=Path)
    args = parser.parse_args()
    directory = args.directory
    report_directory = args.report_dir or directory
    report_directory.mkdir(parents=True,exist_ok=True)
    histories = {name:read(directory/name/'history.json') for name in args.variants}
    if any(h[-1]['updates']!=1320 for h in histories.values()):
        raise ValueError('All study arms must complete 1320 updates')
    selected,averages = select(histories,args.reference)
    write_json(report_directory/'comparison.json',dict(selected=selected,reference=args.reference,
        metric_order=['centered_mse','gradient_mse','mse'],averages=averages,
        final={name:h[-1] for name,h in histories.items()}))
    fig,axes = plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for i,task in enumerate(TASKS):
        for j,key in enumerate(('mse','centered_mse','gradient_mse')):
            ax = axes[i,j]
            for name,h in histories.items():
                ax.plot([e['epoch'] for e in h],[e['val'][task]['all']['metrics'][key] for e in h],label=name)
            ax.set_title(f'{task} VAL {key}')
            ax.set_xlabel('Epoch')
            ax.grid(alpha=.25)
    axes[0,0].legend()
    fig.savefig(report_directory/'curves.png',dpi=150)
    plt.close(fig)
    study = Study(directory,args.device)
    prediction_plots(study,directory,args.variants,report_directory)
    print(json.dumps(dict(selected=selected,averages=averages),indent=2))


if __name__ == '__main__':
    main()
