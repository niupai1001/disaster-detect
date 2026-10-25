"""Masked-band reconstruction; only USE is transferred to downstream training."""
import hashlib
import argparse
import json
import random
import math
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch import nn
from landslide_seg.data.optical_observation import OpticalObservation,collate_optical
from landslide_seg.models.universal_spectral_encoder import UniversalSpectralEncoder
from landslide_seg.models.factory import build_model
from landslide_seg.config import load_segformer_config
from landslide_seg.config import set_seed


def digest(value):return hashlib.sha256(str(value).encode()).hexdigest()


class Reconstruction(nn.Module):
    def __init__(self):
        super().__init__();self.use=UniversalSpectralEncoder()
        self.head=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,33))

    def forward(self,observation,target_metadata,target_known):
        features=self.use(observation)['features']
        descriptor=torch.cat((torch.where(target_known,torch.log1p(target_metadata),0),target_known.float()),-1)
        weight=self.head(descriptor)
        return torch.einsum('bkd,bdhw->bkhw',weight[...,:32],features)+weight[...,32,None,None]


def masked_l1(prediction,target,valid):
    count=valid.sum((-1,-2));usable=count>0
    per_band=torch.where(valid,prediction-target,0).abs().sum((-1,-2))/count.clamp_min(1)
    if not usable.any(-1).all():raise ValueError('Each sample needs a valid reconstruction target')
    return ((per_band*usable).sum(-1)/usable.sum(-1)).mean()


def make_batch(selected,mask_rng,stats,root,device):
    observations=[];targets=[];valids=[];metadata=[];known=[]
    for row in selected:
        path=root/row['file']
        if digest_bytes(path)!=row['sha256']:raise ValueError('External sample changed')
        with np.load(path) as z:
            x=z['x'].copy();valid=z['pixel_valid'].copy();m=z['metadata'].copy();k=z['known'].copy()
        mu,sd=stats[row['sensor']];x=((x-mu[:,None,None])/sd[:,None,None]).astype('float32')
        order=mask_rng.permutation(len(x));target=order[:len(x)//2];visible=order[len(x)//2:]
        observations.append(OpticalObservation(torch.from_numpy(x[visible]),tuple(row['band_ids'][i] for i in visible),
            torch.from_numpy(m[visible]),torch.from_numpy(k[visible]),torch.ones(len(visible),dtype=torch.bool),
            torch.from_numpy(valid[visible]),{'crs':row['crs'],'transform':row['transform']}))
        targets.append(torch.from_numpy(x[target]));valids.append(torch.from_numpy(valid[target]))
        metadata.append(torch.from_numpy(m[target]));known.append(torch.from_numpy(k[target]))
    n=max(t.shape[0] for t in targets)
    def pad(items):
        return torch.stack([torch.cat((v,torch.zeros((n-len(v),*v.shape[1:]),dtype=v.dtype))) for v in items]).to(device)
    return collate_optical(observations).to(device),pad(targets),pad(valids),pad(metadata),pad(known)


def sample_training(train,rng):
    buckets=(('s2',2),('local_s2',2),('landsat',4)) if 'local_s2' in train else (('s2',4),('landsat',4))
    return [train[key][i] for key,n in buckets for i in rng.integers(len(train[key]),size=n)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
    parser.add_argument('--current-data',action='store_true',help='Use user-authorized frozen samples instead of waiting for expansion')
    parser.add_argument('--local-unlabeled',action='store_true',help='Include original unannotated TRAIN-region S2')
    parser.add_argument('--f-local',action='store_true',help='Include accepted F-drive S2 in the local S2 bucket; separate output version')
    parser.add_argument('--data-check-only',action='store_true',help='Validate the selected data and split without building or training a model')
    args=parser.parse_args()
    if args.f_local and not (args.current_data and args.local_unlabeled):
        parser.error('--f-local requires --current-data --local-unlabeled')
    torch.set_num_threads(4);torch.manual_seed(1234);np.random.seed(1234);random.seed(1234)
    precision_settings=set_seed(1234,True)
    root=ROOT/'outputs/use/external';manifest=root/('current_manifest.json' if args.current_data else 'manifest.json')
    for acceptance_name in ('segmentation_interface_acceptance','pretraining_engineering_acceptance'):
        if json.loads((ROOT/f'outputs/use/{acceptance_name}.json').read_text()).get('ok') is not True:
            raise ValueError('Required engineering acceptance failed: '+acceptance_name)
    rows=json.loads(manifest.read_text())
    acceptance=json.loads((root/('current_acceptance.json' if args.current_data else 'full_acceptance.json')).read_text())
    if acceptance.get('ok') is not True or acceptance.get('manifest_sha256')!=digest_bytes(manifest):
        raise ValueError('Full external data/physics/geographic audit is required')
    sensor_counts={s:sum(r['sensor']==s for r in rows) for s in ('s2','landsat')}
    if args.current_data:
        frozen=json.loads((root/'acquisition_paused.json').read_text())
        if frozen['manifest_sha256']!=digest_bytes(manifest) or frozen['sensors']!=sensor_counts:
            raise ValueError('Frozen data differs from user-authorized snapshot')
    elif any(n!=5000 for n in sensor_counts.values()):
        raise ValueError('Formal pretraining requires the accepted 10000-location expansion')
    queue=json.loads((ROOT/'outputs/use/queue_status.json').read_text())
    if queue['status']!='supervised_training_finished_requires_evaluation':
        raise ValueError('Supervised queue must finish first')
    if json.loads((ROOT/'outputs/use/supervised_acceptance.json').read_text()).get('ok') is not True:
        raise ValueError('Independent supervised evaluation/acceptance required before pretraining')
    # Freeze 5% geographic cells per sensor; no pixel from these cells fits statistics.
    train={};validation={}
    cells=sorted({tuple(r['cell']) for r in rows},key=lambda c:digest(('split1234',c)))
    held=set(cells[:max(1,round(len(cells)*.05))])
    for sensor in ('s2','landsat'):
        group=[r for r in rows if r['sensor']==sensor]
        train[sensor]=[r for r in group if tuple(r['cell']) not in held]
        validation[sensor]=[r for r in group if tuple(r['cell']) in held]
        if not train[sensor] or not validation[sensor]:raise ValueError('Empty geographic split for sensor')
    local_manifest=None
    if args.local_unlabeled:
        if not args.current_data:raise ValueError('Local union requires the user-authorized current dataset')
        local_manifest=root/'local_unlabeled_manifest.json'
        local_rows=json.loads(local_manifest.read_text())
        local_audit=json.loads((root/'local_acceptance.json').read_text())
        if local_audit.get('ok') is not True or local_audit['manifest_sha256']!=digest_bytes(local_manifest):
            raise ValueError('Local unlabeled source/quality/holdout audit required')
        if local_audit['protocol_sha256']!=digest_bytes(ROOT/'manifests/v4r1/protocol.json'):
            raise ValueError('Local TRAIN event protocol changed')
        for split,ids in ((train,local_audit['train_ids']),(validation,local_audit['validation_ids'])):
            members=set(ids);split['local_s2']=[r for r in local_rows if r['file'] in members]
            if len(split['local_s2'])!=len(members) or not members:raise ValueError('Invalid local geographic members')
        if set(local_audit['train_ids'])&set(local_audit['validation_ids']):raise ValueError('Overlapping local reconstruction split')
    f_contract=None
    if args.f_local:
        from prepare_use_f_local import load_f_members
        f_train,f_validation,f_contract=load_f_members(root)
        train['local_s2'].extend(f_train);validation['local_s2'].extend(f_validation)
    if args.data_check_only:
        print(json.dumps(dict(data_check_only=True,f_local=bool(f_contract),
            train={k:len(v) for k,v in train.items()},validation={k:len(v) for k,v in validation.items()},
            f_manifest_sha256=f_contract['manifest_sha256'] if f_contract else None,
            f_guard=len(f_contract['guard_ids']) if f_contract else 0),indent=2))
        return
    core=json.loads((ROOT/'manifests/v4r1/stats_core25.json').read_text())
    stats={'s2':(np.asarray(core['mean'])/10000,np.asarray(core['std'])/10000)}
    sums=np.zeros(6);squares=np.zeros(6);counts=np.zeros(6)
    for row in train['landsat']:
        path=root/row['file']
        if digest_bytes(path)!=row['sha256']:raise ValueError('External sample hash mismatch')
        with np.load(path) as z:
            for c in range(6):
                values=z['x'][c][z['pixel_valid'][c]].astype('float64')
                sums[c]+=values.sum();squares[c]+=(values**2).sum();counts[c]+=len(values)
    mean=sums/counts;std=np.sqrt(np.maximum(squares/counts-mean**2,1e-12));stats['landsat']=(mean,std)
    rng=np.random.default_rng(1234)
    def loss_for(selected,gen):
        inp,y,valid,m,k=make_batch(selected,gen,stats,root,'cuda:1');pred=model(inp,m,k)
        return masked_l1(pred,y,valid)
    output=ROOT/('outputs/use/pretraining_f_v1' if args.f_local else 'outputs/use/pretraining');output.mkdir(exist_ok=True)
    if any(output.iterdir()) and not args.resume:raise FileExistsError('Existing pretraining output; inspect before resume')
    fresh=build_model(load_segformer_config(ROOT/'configs/segformer/use/physical_mixed.yaml')['segformer'])
    initialization=fresh.use.state_dict();initial_hash=hashlib.sha256()
    for name,value in sorted(initialization.items()):initial_hash.update(name.encode()+value.cpu().contiguous().numpy().tobytes())
    contract=dict(manifest_sha256=digest_bytes(manifest),seed=1234,updates=10000,batch=8,lr=.0006,warmup=500,
        dataset_mode='frozen_current' if args.current_data else 'full_expansion',sensor_locations=sensor_counts,
        local_manifest_sha256=digest_bytes(local_manifest) if local_manifest else None,
        batch_sources={'external_s2':2,'local_unannotated_s2':2,'landsat':4} if local_manifest else {'external_s2':4,'landsat':4},
        precision_settings=precision_settings,
        source_sha256={str(p.relative_to(ROOT)):digest_bytes(p) for p in [Path(__file__),
            ROOT/'src/landslide_seg/models/universal_spectral_encoder.py',ROOT/'src/landslide_seg/data/optical_observation.py']},
        use_initialization='Same fresh USE parameter values as physical_mixed at seed1234',use_initialization_sha256=initial_hash.hexdigest(),
        stats={s:dict(mean=m.tolist(),std=d.tolist()) for s,(m,d) in stats.items()},
        train_ids={s:[r['file'] for r in train[s]] for s in train},
        validation_ids={s:[r['file'] for r in validation[s]] for s in validation})
    if f_contract:
        contract.update(f_local_manifest_sha256=f_contract['manifest_sha256'],
            f_local_acceptance_sha256=digest_bytes(root/'f_local_acceptance.json'),
            f_guard_ids=f_contract['guard_ids'],
            local_s2_sampling='uniform across original unannotated TRAIN S2 and accepted F TRAIN rows; two samples per batch total',
            data_version='f_local_v1')
    model=Reconstruction().to('cuda:1');opt=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=.01)
    model.use.load_state_dict(initialization,strict=True);del fresh,initialization
    best=float('inf');history=[];start_step=0;previous_seconds=0
    trace=output/'sampling.jsonl'
    if args.resume:
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        if saved['config']!=contract:raise ValueError('Pretraining configuration/data changed')
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_step=saved['step'];best=saved['best_score'];history=saved['history'];previous_seconds=saved['elapsed_seconds']
        if trace.exists():
            lines=[line for line in trace.read_text().splitlines() if json.loads(line)['step']<=start_step]
            trace.write_text('\n'.join(lines)+'\n')
    else:
        (output/'resolved_config.json').write_text(json.dumps(contract,indent=1))
    start=time.time()
    for step in range(start_step+1,10001):
        model.train();selected=sample_training(train,rng)
        sampling=dict(step=step,sample_files=[r['file'] for r in selected],mask_rng=rng.bit_generator.state)
        scale=step/500 if step<=500 else (10000-step)/9500
        for group in opt.param_groups:group['lr']=6e-4*scale
        opt.zero_grad(set_to_none=True);loss=loss_for(selected,rng)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reconstruction loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
        sampling.update(loss=float(loss.detach()),learning_rate=opt.param_groups[0]['lr'],
            seconds=previous_seconds+time.time()-start)
        with trace.open('a',encoding='utf-8') as handle:handle.write(json.dumps(sampling)+'\n')
        if step%500==0:
            model.eval();scores={}
            with torch.no_grad():
                for sensor in validation:
                    values=[];gen=np.random.default_rng(1234)
                    for i in range(0,len(validation[sensor]),8):
                        chunk=validation[sensor][i:i+8];values.append((float(loss_for(chunk,gen)),len(chunk)))
                    scores[sensor]=sum(v*n for v,n in values)/sum(n for _,n in values)
            s2_score=(scores['s2']+scores['local_s2'])/2 if 'local_s2' in scores else scores['s2']
            score=(s2_score+scores['landsat'])/2
            if not math.isfinite(score):raise FloatingPointError('Nonfinite geographic validation reconstruction loss')
            elapsed=previous_seconds+time.time()-start
            history.append(dict(step=step,loss=float(loss.detach()),validation=scores,score=score,seconds=elapsed))
            state=dict(use=model.use.state_dict(),model=model.state_dict(),optimizer=opt.state_dict(),step=step,
                       config=contract,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                       best_score=min(best,score),history=history,elapsed_seconds=elapsed)
            torch.save(state,output/'last.pt')
            if score<best:best=score;torch.save(state,output/'best.pt')
            (output/'history.json').write_text(json.dumps(history,indent=1));print(step,score,flush=True)
    (output/'summary.json').write_text(json.dumps(dict(complete=True,updates=10000,best_score=best,
        seconds=previous_seconds+time.time()-start,manifest_sha256=digest_bytes(manifest),
        best_sha256=digest_bytes(output/'best.pt'),samples_exposed=80000),indent=1))


def digest_bytes(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':main()
