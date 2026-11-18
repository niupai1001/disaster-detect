"""Measure token similarity and local content response without updating weights."""
import json
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from train_pd_here_r3_texture_study import BASE, Study, initial_model, batches, TASKS
from pd_here_geotoken_dualvit_r3.data import write_json

VARIANT = 'fusion256_global2_zerohead'
LAYERS = ('spectral_reduce','spatial_reduce','fuse','global_block_input','global_block',
          'extra_global_blocks.0','encoder_norm','decoder_embed','decoder_norm','prediction',
          'global_block.attention','global_block.mlp','extra_global_blocks.0.attention','extra_global_blocks.0.mlp')


def describe(tokens):
    tokens = tokens.double()
    if len(tokens) < 2:
        return None
    mean = tokens.mean(0,keepdim=True)
    centered = tokens-mean
    rms = tokens.square().mean().sqrt()
    spread = centered.square().mean().sqrt()
    normalized = F.normalize(tokens,dim=-1)
    cosine = normalized@normalized.T
    offdiag = ~torch.eye(len(tokens),device=tokens.device,dtype=torch.bool)
    energy = torch.linalg.svdvals(centered).square()
    rank = energy.sum().square()/energy.square().sum().clamp_min(1e-30)
    distances = torch.cdist(tokens,tokens)/np.sqrt(tokens.shape[-1])
    return dict(cosine=float(cosine[offdiag].mean()),
                relative_patch_spread=float(spread/rms.clamp_min(1e-20)),
                feature_rms=float(rms),patch_spread_rms=float(spread),effective_rank=float(rank),
                min_pair_rmse=float(distances[offdiag].min()),
                exact_duplicate_pairs=int((distances[offdiag]==0).sum()//2))


def summary(records):
    keys = records[0].keys()
    return {key:dict(mean=float(np.mean([r[key] for r in records])),
                     median=float(np.median([r[key] for r in records])),
                     p05=float(np.quantile([r[key] for r in records],.05)),
                     p95=float(np.quantile([r[key] for r in records],.95))) for key in keys}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',default=VARIANT)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    variant=args.variant
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    study = Study(BASE/'texture_study','cuda:0')
    checkpoint = study.release/variant/'last.pt'
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    stats = state['identity']['statistics']
    destination = args.out or study.release/'encoder_token_audit'
    destination.mkdir(exist_ok=True)
    records = defaultdict(list)
    attention_records = defaultdict(list)
    cases = []
    captures = {}
    hooks = []
    model,_ = initial_model(study.layout,variant,stats,study.device)
    modules = dict(model.named_modules())
    def save(name):
        def hook(module,args,result):
            captures[name] = (result[0] if isinstance(result,tuple) else result).detach()
        return hook
    for name in LAYERS:
        if name != 'global_block_input':
            hooks.append(modules[name].register_forward_hook(save(name)))
    def save_input(module,args):
        captures['global_block_input'] = args[0].detach()
    hooks.append(model.global_block.register_forward_pre_hook(save_input))
    selected = json.loads((BASE/'subset700_10epochs/predictions_val/examples.json').read_text())['examples']['B1'][1]['id']
    for initialization in ('initial','trained'):
        if initialization=='trained':
            model.load_state_dict(state['model'])
        model.eval()
        for start,rows in batches(study.rows['val']):
            values,qa,views = study.inputs(rows,0)
            legal = torch.stack([q.flatten(2).any(-1) for q in qa],-1).any(-1)
            for task,view in zip(TASKS,views,strict=True):
                captures.clear()
                result = model(values,qa,*view)
                snapshot = dict(captures)
                for name,x in snapshot.items():
                    valid = legal if name in ('decoder_norm','prediction') else result['valid']
                    for i,row in enumerate(rows):
                        metric = describe(x[i,valid[i]])
                        if metric is not None:
                            records[initialization,task,name].append(metric)
                for name,x in [('global_block',snapshot['global_block_input']),
                               ('extra_global_blocks.0',snapshot['global_block'])]:
                    block = modules[name]
                    h = block.norm1(x)
                    _,weights = block.attention(h,h,h,key_padding_mask=~result['valid'],
                                                need_weights=True,average_attn_weights=False)
                    for i in range(len(rows)):
                        valid = result['valid'][i]
                        a = weights[i][:,valid][:,:,valid].double()
                        if a.shape[-1] < 2:
                            continue
                        entropy = -(a*a.clamp_min(1e-30).log()).sum(-1)/np.log(a.shape[-1])
                        attention_records[initialization,task,name].append(dict(
                            normalized_entropy=float(entropy.mean()),
                            self_attention_mass=float(a.diagonal(dim1=-2,dim2=-1).mean()),
                            query_row_difference=float((a-a.mean(-2,keepdim=True)).square().mean().sqrt())))
                if initialization=='trained' and task=='B1':
                    for i,row in enumerate(rows):
                        if row['id']==selected:
                            cases.append(dict(parent=row['id'],ids=result['ids_keep'][i].cpu().numpy(),
                                **{name:x[i].cpu().numpy() for name,x in snapshot.items()}))
    # A single real parent: modify only patch 0, same masks, then inspect every stage.
    row = next(r for r in study.rows['val'] if r['id']==selected)
    values,qa,views = study.inputs([row],0)
    keep,bands = views[1]
    captures.clear()
    model(values,qa,keep,bands)
    original = dict(captures)
    changed = [x.clone() for x in values]
    for j,x in enumerate(changed):
        if bands[0,j]:
            x[0,0] = torch.where(qa[j][0,0],x[0,0]+.1,x[0,0])
    captures.clear()
    model(tuple(changed),qa,keep,bands)
    response = {}
    for name,old in original.items():
        difference = (captures[name]-old).double()
        response[name] = dict(changed_patch_rmse=float(difference[0,0].square().mean().sqrt()),
                              other_patches_rmse=float(difference[0,1:].square().mean().sqrt()),
                              relative_total_change=float(difference.square().mean().sqrt()/
                                  old.double().square().mean().sqrt().clamp_min(1e-20)))
    # Exact mean replacement at Encoder output isolates its patch-varying component.
    def replace_encoder(module,args,result):
        return result.mean(1,keepdim=True).expand_as(result)
    captures.clear()
    baseline = model(values,qa,keep,bands)['prediction']
    replacement_hook = model.encoder_norm.register_forward_hook(replace_encoder)
    replaced = model(values,qa,keep,bands)['prediction']
    replacement_hook.remove()
    encoder_mean_effect = float((baseline-replaced).double().square().mean().sqrt())
    residuals = {}
    for layer,input_name in [('global_block','global_block_input'),('extra_global_blocks.0','global_block')]:
        residuals[layer] = dict(input_rms=float(original[input_name].double().square().mean().sqrt()),
            attention_update_rms=float(original[layer+'.attention'].double().square().mean().sqrt()),
            mlp_update_rms=float(original[layer+'.mlp'].double().square().mean().sqrt()))
    report = dict(residual_magnitudes=residuals,checkpoint=str(checkpoint),epoch=30,updates=1320,parents=300,
                  task_note='A analyzes retained encoder tokens; B1 analyzes all legal geographic patches',
                  metrics={init:{task:{layer:summary(items) for (phase,t,layer),items in records.items()
                                      if phase==init and t==task} for task in TASKS}
                           for init in ('initial','trained')},
                  attention={init:{task:{layer:summary(items) for (phase,t,layer),items in attention_records.items()
                                      if phase==init and t==task} for task in TASKS}
                           for init in ('initial','trained')},
                  perturbation=dict(parent=selected,added_reflectance=.1,patch=0,task='B1',layers=response),
                  mean_encoder_replacement_prediction_rmse=encoder_mean_effect)
    write_json(destination/'report.json',report)
    example=cases[0]
    np.savez_compressed(destination/'example_tokens.npz',**{k:v for k,v in example.items() if k!='parent'})
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,layer in zip(axes.flat,('fuse','global_block_input','global_block','extra_global_blocks.0','encoder_norm','decoder_norm'),strict=True):
        x=example[layer].astype('float64')
        cosine=(x/np.linalg.norm(x,axis=1,keepdims=True))@(x/np.linalg.norm(x,axis=1,keepdims=True)).T
        im=ax.imshow(cosine,vmin=0,vmax=1,cmap='viridis')
        ax.set_title(layer)
        ax.set_xlabel('Geographic patch')
        ax.set_ylabel('Geographic patch')
        fig.colorbar(im,ax=ax,fraction=.046)
    fig.suptitle('B1 | Pairwise token cosine | Fixed VAL window '+selected[-8:])
    fig.savefig(destination/'token_cosine.png',dpi=150)
    plt.close(fig)
    for hook in hooks:
        hook.remove()
    print(json.dumps(dict(status='PASS',variant=variant,destination=str(destination)),indent=2))


if __name__=='__main__':
    main()
