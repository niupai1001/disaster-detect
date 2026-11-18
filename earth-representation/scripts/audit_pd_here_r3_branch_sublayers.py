"""Locate patch similarity changes inside each local branch, without weight updates."""
import json
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from train_pd_here_r3_texture_study import BASE, Study, initial_model
from pd_here_geotoken_dualvit_r3.data import write_json


def describe(x):
    x=x.double().flatten(1)
    centered=x-x.mean(0,keepdim=True)
    norm=F.normalize(x,dim=-1)
    off=~torch.eye(len(x),device=x.device,dtype=torch.bool)
    gram=centered@centered.T
    e=torch.linalg.eigvalsh(gram).clamp_min(0)
    values=torch.stack(((norm@norm.T)[off].mean(), x.square().mean().sqrt(),
                        centered.square().mean().sqrt(),
                        centered.square().mean().sqrt()/x.square().mean().sqrt().clamp_min(1e-20),
                        e.sum().square()/e.square().sum().clamp_min(1e-30)))
    return dict(zip(('cosine','rms','patch_spread','relative_spread','effective_rank'),values.cpu().tolist()))


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    study=Study(BASE/'texture_study','cuda:0')
    variant='fusion256_global2_zerohead'
    state=torch.load(study.release/variant/'last.pt',map_location='cpu',weights_only=True)
    model,_=initial_model(study.layout,variant,state['identity']['statistics'],study.device)
    examples=json.loads((BASE/'subset700_10epochs/predictions_val/examples.json').read_text())['examples']
    ids={e['id'] for group in examples.values() for e in group}
    rows=study.rows['train'][:16]+[r for r in study.rows['val'] if r['id'] in ids]
    observations=[]
    # Attach hooks to actual sublayers; retain aligned positions when comparing patches.
    targets=['spatial_values','spatial_qa','spectral_reduce','spatial_reduce']
    suffixes=['','norm1','attention','norm2','mlp.0','mlp.1','mlp.2']
    for branch in ('spectral','spatial'):
        targets += [f'{branch}_block'+('.'+s if s else '') for s in suffixes]
    captures=defaultdict(list)
    def save(name):
        def hook(module,args,output):
            captures[name].append((output[0] if isinstance(output,tuple) else output).detach())
        return hook
    def before(name):
        def hook(module,args):
            captures[name].append(args[0].detach())
        return hook
    modules=dict(model.named_modules())
    hooks=[modules[name].register_forward_hook(save(name)) for name in targets]
    for branch in ('spectral','spatial'):
        hooks.append(modules[f'{branch}_block'].register_forward_pre_hook(before(f'{branch}_input')))
        hooks.append(modules[f'{branch}_block.norm2'].register_forward_pre_hook(before(f'{branch}_after_attention')))
    for phase in ('initial','trained'):
        if phase=='trained':
            model.load_state_dict(state['model'])
        model.eval()
        for row in rows:
            values,qa,views=study.inputs([row],0)
            keep,bands=views[1]
            # Compare identical available band/spatial slots across fully valid geographic patches.
            visible=[j for j in range(len(values)) if bands[0,j]]
            legal=keep[0].clone()
            for j in visible:
                legal &= qa[j][0].flatten(1).all(-1)
            indices=legal.nonzero().flatten()
            packed_indices=legal[keep[0]].nonzero().flatten()
            if len(indices)<2:
                continue
            captures.clear()
            model(values,qa,keep,bands)
            k=keep.sum().item()
            spectral={}
            raw=torch.cat([values[j][0].flatten(1) for j in visible],dim=-1)
            spectral['raw_values']=describe(raw[indices])
            spectral_stages={
                'coded_embedding':'spectral_input',
                'ln1':'spectral_block.norm1',
                'attention_update':'spectral_block.attention',
                'after_attention_residual':'spectral_after_attention',
                'ln2':'spectral_block.norm2',
                'ffn_linear1':'spectral_block.mlp.0',
                'ffn_gelu':'spectral_block.mlp.1',
                'ffn_linear2_update':'spectral_block.mlp.2',
                'after_ffn_residual_preflatten':'spectral_block'}
            for label,key in spectral_stages.items():
                spectral[label]=describe(captures[key][0].reshape(k,-1)[packed_indices])
            spectral['flatten_projection_output']=describe(captures['spectral_reduce'][0][0,packed_indices])
            # Reconstruct pre-code spectral embeddings from the actual coded input.
            band_ids=torch.tensor(visible,device=values[0].device)
            embedded=captures['spectral_input'][0]-model.band_position[band_ids][None]
            spectral['value_plus_qa_embedding']=describe(embedded[packed_indices])
            spatial={}
            spatial_stages={
                'value_linear':'spatial_values',
                'coded_embedding':'spatial_input',
                'ln1':'spatial_block.norm1',
                'attention_update':'spatial_block.attention',
                'after_attention_residual':'spatial_after_attention',
                'ln2':'spatial_block.norm2',
                'ffn_linear1':'spatial_block.mlp.0',
                'ffn_gelu':'spatial_block.mlp.1',
                'ffn_linear2_update':'spatial_block.mlp.2',
                'after_ffn_residual_preflatten':'spatial_block'}
            for label,key in spatial_stages.items():
                # Calls occur in canonical band order, skipping the one hidden band.
                packed=torch.cat(captures[key],dim=1)
                spatial[label]=describe(packed[packed_indices])
            value_qa=torch.cat([v+q for v,q in zip(captures['spatial_values'],captures['spatial_qa'],strict=True)],dim=1)
            spatial['value_plus_qa_embedding']=describe(value_qa[packed_indices])
            spatial['raw_values']=describe(raw[indices])
            spatial['flatten_projection_output']=describe(captures['spatial_reduce'][0][0,packed_indices])
            # Flatten should preserve every norm, dot product and pair distance.
            pre=captures['spatial_block']
            assert all(x.shape[0]==k for x in pre)
            observations.append(dict(phase=phase,parent=row['id'],role=study.panel[row['id']]['role'],
                                     patches=len(indices),spectral=spectral,spatial=spatial))
    for h in hooks:
        h.remove()
    summary={}
    for phase in ('initial','trained'):
        selected=[r for r in observations if r['phase']==phase]
        summary[phase]={}
        for branch in ('spectral','spatial'):
            summary[phase][branch]={stage:{metric:float(np.mean([r[branch][stage][metric] for r in selected]))
                  for metric in selected[0][branch][stage]} for stage in selected[0][branch]}
    out=study.release/'branch_sublayer_audit'
    write_json(out/'report.json',dict(checkpoint=str(study.release/variant/'last.pt'),
        task='B1 exposure0',requested_parents=len(rows),included_parents=len(observations)//2,
        qa_policy='same fully QA-valid geographic patches and visible bands at every stage',
        comparison='one concatenated vector per geographic patch with aligned band/subpatch slots; parent-equal summaries',
        note='Flatten is only a view; compare before and after its following Linear. Attention/FFN updates are not residual outputs.',
        summary=summary,observations=observations))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
