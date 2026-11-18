"""Same-band, cross-geographic-patch token similarity for the band-token model."""
from collections import defaultdict
import json
import numpy as np
import torch
from train_pd_here_r3_texture_study import BASE,Study,initial_model,batches
from audit_pd_here_r3_tokens import describe
from pd_here_geotoken_dualvit_r3.data import write_json


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    study=Study(BASE/'texture_study','cuda:0')
    state=torch.load(study.release/'band_tokens256/last.pt',map_location='cpu',weights_only=True)
    model,_=initial_model(study.layout,'band_tokens256',state['identity']['statistics'],study.device)
    model.load_state_dict(state['model']);model.eval()
    captures={}
    def save(name):
        def hook(module,args,result):captures[name]=result.detach()
        return hook
    hooks=[model.fuse.register_forward_hook(save('fusion_pre_global')),
           model.global_block.register_forward_hook(save('global1')),
           model.extra_global_blocks[0].register_forward_hook(save('global2')),
           model.encoder_norm.register_forward_hook(save('encoder_norm')),
           model.decoder_norm.register_forward_hook(save('decoder_norm'))]
    records=defaultdict(list);n=len(study.layout.band_keys);p=study.layout.patches
    for _,rows in batches(study.rows['val']):
        values,qa,views=study.inputs(rows,0)
        output=model(values,qa,*views[1])
        pair_ids=output['ids_keep'];valid=output['valid']
        for i,row in enumerate(rows):
            for j in range(n):
                selected=valid[i] & (pair_ids[i]%n==j)
                if selected.sum()<2:continue
                legal=qa[j][i].flatten(1).all(-1)
                positions=pair_ids[i,selected]//n
                fully_valid=legal[positions]
                if fully_valid.sum()<2:continue
                for name,tensor in captures.items():
                    if name=='fusion_pre_global':
                        features=tensor[i].reshape(-1,model.geographic_width)[selected][fully_valid]
                    elif name=='decoder_norm':
                        features=tensor[i].reshape(p,n,-1)[positions[fully_valid],j]
                    else:
                        features=tensor[i,selected][fully_valid]
                    r=describe(features)
                    records[name,j].append(dict(parent=row['id'],**r))
    summary={}
    for name in captures:
        bands={}
        for j,key in enumerate(study.layout.band_keys):
            items=records[name,j]
            if items:
                bands[key]=dict(parents=len(items),**{k:float(np.mean([r[k] for r in items]))
                                                      for k in items[0] if k!='parent'})
        summary[name]=dict(bands=bands,macro={k:float(np.mean([v[k] for v in bands.values()]))
                            for k in next(iter(bands.values())) if k!='parents'})
    for h in hooks:h.remove()
    write_json(study.release/'band_tokens_comparison/token_similarity.json',
               dict(scope='300 developmentVAL pool; B1; same visible band compared across full-QA geographic patches',
                    summary=summary))
    print(json.dumps({name:r['macro'] for name,r in summary.items()},indent=2))


if __name__=='__main__':main()
