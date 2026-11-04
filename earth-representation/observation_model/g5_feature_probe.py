"""Post-selection frozen Encoder probes: random/trained, H/H+native features.

Only closed-form TRAIN heads are fitted. Encoder/Decoder weights never update.
"""
from dataclasses import replace
import json
import math

import numpy as np
import torch

from .encoder_v2 import EncoderConfig,ObservationEncoderV2
from .encoder_v2_tasks import spectral_task,sample_target_indices
from .g5_protocol import G5,ObservationStream
from .audit_ridge_bands import DEST as AUDIT,family,aligned,feature_matrix,model_key,normalization,phi
from .check_encoder_v2 import state_hash
from .expansion_data import read,save,digest


def baseline_inputs(image,target,visible_keys,norm):
    bands={b.band_key:dict(shape=list(b.values.shape),transform=list(b.transform),x=b.values,q=b.keep()) for b in image.bands if b.band_key in visible_keys}
    target_dict=dict(shape=list(target.values.shape),transform=list(target.transform),x=target.values,q=target.valid)
    aligned_values=aligned(bands,target_dict,{})
    fam='S2' if target.band_key.startswith('B') else 'L8'
    x,q,labels=feature_matrix(aligned_values,sorted(visible_keys),[],fam,norm)
    return x,q,labels,fam


def baseline_values(image,target,visible_keys,fits,norm,ids):
    x,q,labels,fam=baseline_inputs(image,target,visible_keys,norm)
    fit=fits[model_key(fam,target.band_key,labels)]
    assert q.all(-1).flatten()[ids].all(), 'All comparisons require observed baseline covariates'
    x=x.reshape(-1,len(labels))[ids]
    aug=torch.cat([torch.ones(len(x),1,device=x.device),x],-1).double()
    mean,std=norm[fam+':'+target.band_key]
    ridge=(aug@fit['linear'])*std+mean
    nonlinear=ridge+(phi(x,fit).double()-fit['phi_mean'])@fit['residual']*std
    return ridge.float(),nonlinear.float(),std


def frozen_features(model,representation,target,ids):
    grid=representation.grids[0];left,bottom,right,top=grid['bounds'];ny,nx=grid['shape']
    a,_,c,_,e,f=target.grid.transform;h,w=target.grid.shape
    px=(ids%w).double()*a+(c-left)+a/2;py=(ids//w).double()*(-e)+(top-f)-e/2
    coordinates=torch.stack([px,py],-1).float()
    xx=torch.floor(px/(right-left)*nx).long().clamp(0,nx-1);yy=torch.floor(py/(top-bottom)*ny).long().clamp(0,ny-1)
    H=representation.features[0,yy*nx+xx].float()
    local=[]
    for bank in sorted(representation.optional_local_features[0],key=lambda b:b.band_key):
        indices=model.decoder.nearest_native(bank,coordinates,grid['bounds'])
        local.append(bank.features[indices].flatten(1).float())
    # No learned Decoder weights or target pixel values enter either feature set.
    # Pool the spectral set, retaining four neighbor ranks. Dimension stays192
    # when bands are missing, so a fixed TRAIN head can test unseen combinations.
    return dict(H=H,H_and_bank=torch.cat([H,torch.stack(local).mean(0)],-1))


def fit_head(x,y):
    x=x.double();y=y.double();mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-4);ym=y.mean()
    z=(x-mean)/scale
    weight=torch.linalg.solve(z.T@z/len(z)+.01*torch.eye(z.shape[1],device=z.device,dtype=z.dtype),z.T@(y-ym)/len(z))
    return dict(mean=mean,scale=scale,ym=ym,weight=weight)


def main():
    destination=G5/'frozen_probe';destination.mkdir(exist_ok=True)
    if (destination/'results.json').exists():raise FileExistsError('Preserve completed feature probe')
    config=read(G5/'config.json');device=config['device'];torch.set_num_threads(4);torch.cuda.set_device(device)
    stream=ObservationStream(device);norm=normalization();fits=torch.load(AUDIT/'TRAIN_fits.pt',map_location=device,weights_only=False)
    plan=read(G5/'feature_probe_plan.json')['plans']
    train_index={r['id']:i for i,r in enumerate(stream.train.rows)};val_index={r['id']:i for i,r in enumerate(stream.val.rows)}
    results=[];hashes={};baseline_target_hash={}
    with torch.inference_mode():
        for version,filename in [('random','initialization_untrained.pt'),('trained','best.pt')]:
            model=ObservationEncoderV2(EncoderConfig(local_width=16,query_chunk=2048)).to(device).eval()
            payload=torch.load(G5/filename,map_location='cpu',weights_only=False);model.load_state_dict(payload['model']);before=state_hash(model)
            tasks={(p['sensor'],p['target'],p['context']) for p in plan}
            for sensor,band,context in sorted(tasks):
                sets={}
                for role in ['TRAIN','VAL','VAL_unseen_combo','unseen_S2C']:
                    if role=='VAL_unseen_combo' and context=='RGB_only':continue
                    if role=='unseen_S2C':
                        if not (sensor=='S2' and band=='B08' and context=='RGB_only'):continue
                        members=[stream.val.rows[i]['id'] for i in stream.response]
                    else:members=next(p['members'] for p in plan if (p['sensor'],p['target'],p['context'],p['role'])==(sensor,band,context,'VAL' if role=='VAL_unseen_combo' else role))
                    features={'H':[],'H_and_bank':[]};truth=[];ridge_errors=[];nonlinear_errors=[];row_ids=[]
                    for member in members:
                        index=train_index[member] if role=='TRAIN' else val_index[member]
                        meta,image=stream.image(index,'TRAIN' if role=='TRAIN' else 'VAL')
                        hidden=[b.band_key for b in image.bands if b.band_key not in ['B02','B03','B04']] if context=='RGB_only' else [band]
                        if role=='VAL_unseen_combo':
                            if band=='B11':hidden=['B11','B12']
                            elif band=='B07':hidden=['B05','B06','B07']
                            else:hidden=['SR_B3','SR_B4','SR_B5']
                        task=spectral_task(image,hidden);target=next(t for t in task.targets if t.band_key==band)
                        original=next(b for b in image.bands if b.band_key==band)
                        _,support,_,_=baseline_inputs(image,original,[b.band_key for b in task.visible.bands],norm)
                        target=replace(target,valid=target.valid&support.all(-1))
                        ids=sample_target_indices(target,member,'B',128,0)
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                            representation=model.encode(task.visible)
                            feat=frozen_features(model,representation,target,ids)
                        ridge,nonlinear,std=baseline_values(image,next(b for b in image.bands if b.band_key==band),[b.band_key for b in task.visible.bands],fits,norm,ids)
                        y=target.values.flatten()[ids].float();mean=norm[sensor+':'+band][0];normalized=(y-mean)/std
                        key=(member,band,context,role)
                        if version=='random':baseline_target_hash[key]=(ids.cpu().tolist(),normalized.cpu().numpy().tobytes())
                        else:assert baseline_target_hash[key]==(ids.cpu().tolist(),normalized.cpu().numpy().tobytes())
                        for name,x in feat.items():features[name].append(x)
                        truth.append(normalized);ridge_errors.append(((ridge-y)/std).square());nonlinear_errors.append(((nonlinear-y)/std).square())
                        row_ids.extend([member]*len(y))
                    sets[role]=dict(features={k:torch.cat(v) for k,v in features.items()},truth=torch.cat(truth),
                        ridge=torch.cat(ridge_errors),nonlinear=torch.cat(nonlinear_errors),ids=row_ids)
                for feature_type in ['H','H_and_bank']:
                    head=fit_head(sets['TRAIN']['features'][feature_type],sets['TRAIN']['truth'])
                    for role,data in sets.items():
                        x=data['features'][feature_type].double();prediction=((x-head['mean'])/head['scale'])@head['weight']+head['ym']
                        errors=(prediction-data['truth']).square()
                        by_member={m:float(errors[torch.tensor([i for i,v in enumerate(data['ids']) if v==m],device=device)].mean()) for m in sorted(set(data['ids']))}
                        results.append(dict(version=version,sensor=sensor,target=band,context=context,role=role,feature_type=feature_type,
                            head_feature_dimensions=x.shape[1],members=len(by_member),pixels=len(errors),
                            ridge_nmse=float(data['ridge'].mean()),RFF_residual_control_nmse=float(data['nonlinear'].mean()),
                            frozen_encoder_probe_nmse=float(errors.mean()),Ridge_error_fraction_reduced=1-float(errors.mean()/data['ridge'].mean()),
                            per_member_nmse=by_member))
                print('FROZEN_PROBE',version,sensor,band,context,flush=True)
            assert before==state_hash(model);hashes[version]=dict(file_sha256=digest(G5/filename),model_hash_before=before,model_hash_after=state_hash(model))
            del model
    save(destination/'results.json',dict(status='COMPLETE',rows=results,model_hashes=hashes,encoder_optimizer_updates=0,
        selection='Checkpoint selected first by unlabeled primary; probes not used for checkpoint selection',
        caveat='Heads predict the full normalized target from frozen features; Ridge predictions are comparisons, not head inputs. H vs H+bank heads have different dimensionality. Trained vs random is matched within feature type. VAL_unseen_combo reuses the same TRAIN head without refitting. This is reflectance transfer, not independent semantic evidence.'))
    print('FROZEN_ENCODER_PROBE_COMPLETE')


if __name__=='__main__':main()
