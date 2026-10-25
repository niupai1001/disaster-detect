"""Independent MBM controls: no supervised model, task dataset, or task labels."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from landslide_seg.data.optical_observation import OpticalObservation, collate_optical
from landslide_seg.models.universal_spectral_encoder import UniversalSpectralEncoder
from landslide_seg.models.frozen_use import state_sha256
from audit_use_independent_data import sha, read, EXTERNAL

VARIANTS = {
    'mean': dict(attention=False, queries=0, metadata=False),
    'attention': dict(metadata=False),
    'physical': {},
    'random_identity': dict(identity='random'),
    'multisensor': {},
    'spatial': {},
}
OUT = ROOT/'outputs/use_independent_frozen'


class Reconstruction(nn.Module):
    def __init__(self, encoder_config, spatial=False):
        super().__init__()
        self.use = UniversalSpectralEncoder(**encoder_config)
        self.head = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 33))
        self.spatial_decoder = nn.Sequential(nn.Conv2d(32,32,3,padding=2,dilation=2),nn.GELU(),
                                            nn.Conv2d(32,32,3,padding=2,dilation=2)) if spatial else None

    def forward(self, observation, target_metadata, target_known, spatial=False):
        feature = self.use(observation)['features']
        if spatial:
            if self.spatial_decoder is None:
                raise ValueError('Spatial objective not configured')
            feature = self.spatial_decoder(feature)
        descriptor = torch.cat((torch.where(target_known, torch.log1p(target_metadata), 0), target_known.float()), -1)
        weight = self.head(descriptor)
        return torch.einsum('bkd,bdhw->bkhw', weight[..., :32], feature) + weight[..., 32, None, None]


def initialize(variant, device):
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = UniversalSpectralEncoder()
    model = Reconstruction(VARIANTS[variant], spatial=variant=='spatial')
    state = model.use.state_dict()
    reference_state = reference.state_dict()
    model.use.load_state_dict({k: reference_state[k] if k in reference_state and reference_state[k].shape == v.shape else v
                               for k, v in state.items()}, strict=True)
    # The temporary readout must also start identically across all controls.
    torch.manual_seed(1235)
    for module in model.head.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()
    return model.to(device)


def masked_mae(prediction, target, valid):
    if valid.dtype != torch.bool or target.shape != prediction.shape or valid.shape != target.shape:
        raise ValueError('Invalid MBM target shapes/types')
    if not torch.isfinite(target[valid]).all() or not torch.isfinite(prediction[valid]).all():
        raise ValueError('Nonfinite MBM target or prediction')
    count = valid.sum((-1, -2))
    usable = count > 0
    if not usable.any(-1).all():
        raise ValueError('Every sample needs a valid target')
    band = torch.where(valid, prediction-target, 0).abs().sum((-1, -2)) / count.clamp_min(1)
    return ((band*usable).sum(-1)/usable.sum(-1)).mean()


def batch(rows, rng, stats, device):
    observations, targets, valids, metadata, known, scales = [], [], [], [], [], []
    trace = []
    for row in rows:
        path = EXTERNAL/row['file']
        if sha(path) != row['sha256']:
            raise ValueError('Cache changed after audit')
        with np.load(path, allow_pickle=False) as arrays:
            x, valid, m, k = (arrays[key].copy() for key in ('x', 'pixel_valid', 'metadata', 'known'))
        mu, sd = (np.asarray(stats[row['sensor']][key], dtype='float32') for key in ('mean', 'std'))
        x = ((x-mu[:, None, None])/sd[:, None, None]).astype('float32')
        order = rng.permutation(len(x))
        target, visible = order[:len(x)//2], order[len(x)//2:]
        observations.append(OpticalObservation(torch.from_numpy(x[visible]), tuple(row['band_ids'][i] for i in visible),
            torch.from_numpy(m[visible]), torch.from_numpy(k[visible]), torch.ones(len(visible), dtype=torch.bool),
            torch.from_numpy(valid[visible]), {'crs': row['crs'], 'transform': row['transform']}))
        targets.append(torch.from_numpy(x[target])); valids.append(torch.from_numpy(valid[target]))
        metadata.append(torch.from_numpy(m[target])); known.append(torch.from_numpy(k[target])); scales.append(torch.from_numpy(sd[target]))
        trace.append(dict(file=row['file'], sensor=row['sensor'], target_bands=[row['band_ids'][i] for i in target],
                          visible_bands=[row['band_ids'][i] for i in visible]))
    width = max(len(t) for t in targets)
    def pad(items):
        return torch.stack([torch.cat((v, torch.zeros((width-len(v), *v.shape[1:]), dtype=v.dtype))) for v in items]).to(device)
    return collate_optical(observations).to(device), pad(targets), pad(valids), pad(metadata), pad(known), pad(scales), trace


def inputs(variant):
    directory = OUT/'cache_audit_v1'
    acceptance = read(directory/'acceptance.json')
    if acceptance['ok'] is not True or sha(directory/'members.jsonl') != acceptance['members_sha256']:
        raise ValueError('Full independent cache audit required')
    for name, expected in acceptance['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Accepted source changed: ' + name)
    rows = [json.loads(line) for line in (directory/'members.jsonl').read_text(encoding='utf8').splitlines()]
    sensors = ('s2', 'landsat') if variant in ('multisensor','spatial') else ('s2',)
    splits = {role: {sensor: [r for r in rows if r['role'] == role and r['sensor'] == sensor]
                     for sensor in sensors} for role in ('TRAIN', 'PRETRAIN_VAL')}
    if any(not members for split in splits.values() for members in split.values()):
        raise ValueError('Empty source split')
    stats = read(ROOT/'outputs/use/pretraining/resolved_config.json')['stats']
    return splits, stats, sha(directory/'acceptance.json'), acceptance['members_sha256']


def sample(split, rng):
    count = 8//len(split)
    return [rows[i] for rows in split.values() for i in rng.integers(len(rows), size=count)]


def learning_rate(step):
    return .0006 * (step/500 if step <= 500 else (10000-step)/9500)


def spatial_batch(rows, rng, stats, device):
    observations, targets, valids, metadata, known = [], [], [], [], []
    for row in rows:
        path=EXTERNAL/row['file']
        if sha(path)!=row['sha256']: raise ValueError('Cache changed after audit')
        with np.load(path,allow_pickle=False) as arrays:
            x, valid, m, k=(arrays[key].copy() for key in ('x','pixel_valid','metadata','known'))
        mu,sd=(np.asarray(stats[row['sensor']][key],dtype='float32') for key in ('mean','std'))
        x=((x-mu[:,None,None])/sd[:,None,None]).astype('float32')
        h,w=x.shape[-2:]
        coarse=rng.random(((h+3)//4,(w+3)//4)) < .25
        if coarse.size<2: raise ValueError('Spatial masking requires at least two blocks')
        if not coarse.any(): coarse.flat[0]=True
        if coarse.all(): coarse.flat[-1]=False
        hidden=np.repeat(np.repeat(coarse,4,axis=0),4,axis=1)[:h,:w]
        visible=valid & ~hidden[None]
        target_valid=valid & hidden[None]
        if not target_valid.any() or not visible.any(): raise ValueError('Empty spatial target or input')
        observations.append(OpticalObservation(torch.from_numpy(x),tuple(row['band_ids']),torch.from_numpy(m),torch.from_numpy(k),
            torch.ones(len(x),dtype=torch.bool),torch.from_numpy(visible),{'crs':row['crs'],'transform':row['transform']}))
        targets.append(torch.from_numpy(x));valids.append(torch.from_numpy(target_valid));metadata.append(torch.from_numpy(m));known.append(torch.from_numpy(k))
    width=max(len(t) for t in targets)
    def pad(items):
        return torch.stack([torch.cat((v,torch.zeros((width-len(v),*v.shape[1:]),dtype=v.dtype))) for v in items]).to(device)
    return collate_optical(observations).to(device),pad(targets),pad(valids),pad(metadata),pad(known)


def spatial_loss(model, rows, rng, stats, device):
    obs,target,valid,m,k=spatial_batch(rows,rng,stats,device)
    return masked_mae(model(obs,m,k,spatial=True),target,valid)


def calibrate_spatial(model, split, stats, device):
    # Initial TRAIN batches, no updates; fix relative loss scales before the smoke or formal run.
    band_rng=np.random.default_rng(1234);space_rng=np.random.default_rng(1236)
    band_losses=[];space_losses=[];files=[]
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            rows=sample(split,band_rng)
            obs,target,valid,m,k,_,_=batch(rows,band_rng,stats,device)
            band_losses.append(float(masked_mae(model(obs,m,k),target,valid)))
            space_losses.append(float(spatial_loss(model,rows,space_rng,stats,device)))
            files.append([row['file'] for row in rows])
    weight=float(np.median(band_losses)/max(np.median(space_losses),1e-8))
    if not math.isfinite(weight) or weight<=0: raise ValueError('Invalid spatial loss calibration')
    return dict(weight=weight,mbm_losses=band_losses,spatial_losses=space_losses,files=files,updates=0)


def validation(model, split, stats, device):
    model.eval()
    result = {}
    with torch.no_grad():
        for sensor, rows in split.items():
            rng = np.random.default_rng(1234)
            space_rng = np.random.default_rng(1236)
            loss_sum = 0.
            spatial_sum = 0.
            metrics = {}
            for start in range(0, len(rows), 8):
                selected = rows[start:start+8]
                obs, target, valid, m, k, sd, trace = batch(selected, rng, stats, device)
                prediction = model(obs, m, k)
                loss_sum += float(masked_mae(prediction, target, valid))*len(selected)
                if model.spatial_decoder is not None:
                    spatial_sum += float(spatial_loss(model,selected,space_rng,stats,device))*len(selected)
                for i, item in enumerate(trace):
                    for j, name in enumerate(item['target_bands']):
                        delta = (prediction[i,j]-target[i,j])[valid[i,j]].double()*sd[i,j].double()
                        totals = metrics.setdefault(name, dict(absolute=0., squared=0., pixels=0))
                        totals['absolute'] += float(delta.abs().sum()); totals['squared'] += float(delta.square().sum()); totals['pixels'] += delta.numel()
            result[sensor] = dict(standardized_mae=loss_sum/len(rows), reflectance={name: dict(mae=t['absolute']/t['pixels'], rmse=math.sqrt(t['squared']/t['pixels']), pixels=t['pixels']) for name,t in metrics.items()})
            if model.spatial_decoder is not None:
                result[sensor]['standardized_spatial_mae']=spatial_sum/len(rows)
    return result, sum(v['standardized_mae'] for v in result.values())/len(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--action', required=True, choices=['smoke', 'freeze', 'train'])
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    torch.set_num_threads(4)
    splits, stats, audit_hash, members_hash = inputs(args.variant)
    model = initialize(args.variant, args.device)
    rng = np.random.default_rng(1234)
    spatial_rng = np.random.default_rng(1236)
    calibration = calibrate_spatial(model,splits['TRAIN'],stats,args.device) if args.variant=='spatial' else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0006, weight_decay=.01)
    source_paths = [Path(__file__), Path(__file__).with_name('audit_use_independent_data.py'),
                    ROOT/'src/landslide_seg/models/universal_spectral_encoder.py',
                    ROOT/'src/landslide_seg/models/frozen_use.py', ROOT/'src/landslide_seg/data/optical_observation.py']
    contract = dict(variant=args.variant, use_config=VARIANTS[args.variant], seed=1234, device=args.device,
                    torch_version=torch.__version__, python_version=sys.version, precision='FP32', deterministic=True, tf32=False,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    command=[sys.executable,str(Path(__file__).resolve()),'--variant',args.variant,'--action','train','--device',args.device],
                    updates=10000, batch=8, optimizer='AdamW', lr=.0006, weight_decay=.01,
                    cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                    loss='valid-pixel MAE per hidden band, mean over usable bands per sample, then mean over samples',
                    task_labels=False, supervision='masked_band_and_spatial_reconstruction' if calibration else 'masked_band_reconstruction_only',
                    spatial_objective=dict(block_pixels=4,mask_probability=.25,calibration=calibration,extra_views_per_sample=1) if calibration else None,
                    batch_sources={sensor:8//len(splits['TRAIN']) for sensor in splits['TRAIN']},
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    selection_role='PRETRAIN_VAL only; downstream DEV not used',
                    warmup_updates=500, schedule='linear warmup, linear decay to zero at update 10000', grad_clip=1.,
                    mask='random floor(C/2) hidden bands per observation', validation_every=500,
                    selection='mean of sensor-wise sample/band-balanced standardized validation MAE; no downstream labels',
                    statistics=stats, cache_acceptance_sha256=audit_hash, members_sha256=members_hash,
                    train_ids={s:[r['file'] for r in rows] for s,rows in splits['TRAIN'].items()},
                    validation_ids={s:[r['file'] for r in rows] for s,rows in splits['PRETRAIN_VAL'].items()},
                    initialization_sha256=state_sha256(model), use_initialization_sha256=state_sha256(model.use),
                    source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths})
    if args.action == 'smoke':
        model.train()
        losses = []
        for _ in range(3):
            selected=sample(splits['TRAIN'],rng)
            obs, target, valid, m, k, _, _ = batch(selected, rng, stats, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mae(model(obs,m,k), target,valid)
            if calibration:
                loss=loss+calibration['weight']*spatial_loss(model,selected,spatial_rng,stats,args.device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        path = OUT/f'smoke_{args.variant}.json'
        path.write_text(json.dumps(dict(passed=True, losses=losses, contract=contract),indent=2)+'\n')
        return
    smoke = read(OUT/f'smoke_{args.variant}.json')
    if smoke.get('passed') is not True or smoke['contract'] != contract:
        raise ValueError('Matching engineering smoke required; initialize formal training again')
    directory = OUT/'pretraining'/args.variant
    if args.action == 'freeze':
        directory.mkdir(parents=True, exist_ok=False)
        (directory/'resolved_config.json').write_text(json.dumps(contract, indent=2)+'\n')
        for source in source_paths:
            target=directory/'source_snapshot'/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
            if sha(target)!=contract['source_sha256'][str(source.relative_to(ROOT))]:
                raise ValueError('Source snapshot changed during freeze')
        torch.save(dict(use=model.use.state_dict(), model=model.state_dict(), config=contract), directory/'initialization.pt')
        return
    engineering = read(OUT/'engineering_acceptance.json')
    if engineering.get('passed') is not True:
        raise ValueError('Full frozen-interface engineering gate has not passed')
    for name, expected in engineering.get('source_sha256', {}).items():
        if sha(ROOT/name) != expected:
            raise ValueError('Engineering gate source changed; rerun acceptance')
    if engineering.get('cache_acceptance_sha256') != audit_hash:
        raise ValueError('Engineering gate data changed')
    if read(directory/'resolved_config.json') != contract or (directory/'last.pt').exists():
        raise ValueError('Configuration changed or existing run; no implicit resume/overwrite')
    initial = torch.load(directory/'initialization.pt', map_location='cpu', weights_only=False)
    if state_sha256(model) != initial['config']['initialization_sha256']:
        raise ValueError('Formal reinitialization differs')
    history = []; best = float('inf'); started = time.time()
    with (directory/'sampling.jsonl').open('x', encoding='utf8') as trace:
        for step in range(1, 10001):
            model.train()
            selected=sample(splits['TRAIN'],rng)
            obs, target, valid, m, k, _, exposure = batch(selected, rng, stats, args.device)
            for group in optimizer.param_groups: group['lr'] = learning_rate(step)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mae(model(obs,m,k),target,valid)
            band_value=float(loss.detach())
            space_value=None
            space_state=spatial_rng.bit_generator.state if calibration else None
            if calibration:
                extra=spatial_loss(model,selected,spatial_rng,stats,args.device)
                space_value=float(extra.detach())
                loss=loss+calibration['weight']*extra
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            trace.write(json.dumps(dict(step=step, loss=float(loss.detach()),mbm_loss=band_value,spatial_loss=space_value,
                                       spatial_mask_rng=space_state,lr=learning_rate(step), exposure=exposure, seconds=time.time()-started))+'\n')
            if step % 500 == 0:
                trace.flush()
                metrics, score = validation(model,splits['PRETRAIN_VAL'],stats,args.device)
                if not math.isfinite(score): raise FloatingPointError('Nonfinite validation score')
                history.append(dict(step=step,score=score,metrics=metrics,seconds=time.time()-started))
                saved = dict(use=model.use.state_dict(),model=model.state_dict(),optimizer=optimizer.state_dict(),
                             config=contract,step=step,history=history,best_score=min(best,score),
                             rng=rng.bit_generator.state,spatial_rng=spatial_rng.bit_generator.state,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
                torch.save(saved,directory/'last.pt')
                if score < best: best=score; torch.save(saved,directory/'best.pt')
                (directory/'history.json').write_text(json.dumps(history,indent=2)+'\n')
                print(step,score,flush=True)
    (directory/'summary.json').write_text(json.dumps(dict(complete=True,updates=10000,best_score=best,
          config_sha256=sha(directory/'resolved_config.json'),members_sha256=members_hash,
          best_sha256=sha(directory/'best.pt'),last_sha256=sha(directory/'last.pt')),indent=2)+'\n')


if __name__ == '__main__':
    main()
