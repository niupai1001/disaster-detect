"""Release-bound dual-route execution; engineering never starts formal jobs."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml
import pretrain_use_v4 as pt
import run_use_v2 as rt
import run_use_v4 as v4
from audit_use_independent_data import EXTERNAL, sha, audit_cache
from landslide_seg.models.use_v4_mit import V3, MIT, FRONTEND
from landslide_seg.models.use_mit_dual_routes import (DualRouteEncoder, PatchTeacher,
    rgb_indices, visible_rgb, support_fraction, alignment_loss)
from landslide_seg.models.use_v4_reconstruction import Reconstruction, Segmenter
from landslide_seg.models.use_v3_reconstruction import packed_state_sha256

ROOT = pt.ROOT
BASE = ROOT / 'outputs/use_mit_dual_routes'
CONFIG = ROOT / 'configs/use_mit_dual_routes/execution.yaml'
MEMBERS = ROOT / 'outputs/use_independent_frozen/cache_audit_v1/members.jsonl'
CALIBRATION = ROOT / 'outputs/use_v4/engineering/calibration.json'
PUBLIC = ROOT / 'outputs/use_v3/engineering/calibration.json'


def matrix():
    jobs = {}
    for route in ('A', 'B'):
        jobs[f'pretrain_{route}'] = dict(route=route, mode='pretrain', steps=20000, warmup=1000)
        for task in ('fine', 'context'):
            jobs[f'probe_{route}_{task}'] = dict(route=route, mode='probe', task=task, steps=1000, warmup=50)
    for route in ('C0', 'A', 'B'):
        jobs[f'segment_{route}'] = dict(route=route, mode='segmentation', steps=5000, warmup=250)
    assert sum(j['steps'] for j in jobs.values()) == 59000
    return jobs


def prepare():
    config = yaml.safe_load((ROOT / 'configs/use_v4/execution.yaml').read_text())
    contract = config.pop('use_v4')
    contract.update(jobs=matrix(), formal_budget=59000, alignment_weight=.20,
                    selection='fixed u20000; no encoder checkpoint selection',
                    calibration_path=CALIBRATION.relative_to(ROOT).as_posix(),
                    calibration_sha256=sha(CALIBRATION), public_calibration_sha256=sha(PUBLIC),
                    encoder_evaluation_checkpoint=20000)
    config['use_mit_dual_routes'] = contract
    config['segformer']['model']['type'] = 'use_mit_dual_routes'
    config['segformer']['output_dir'] = 'outputs/use_mit_dual_routes/runs'
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    for name in ('execution.yaml', 'engineering.yaml'):
        with (CONFIG.parent / name).open('x', encoding='utf8') as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    preflight()


def preflight():
    from check_research_protocol import audit_config
    for path, digest in v4.EXPECTED.items():
        if sha(path) != digest:
            raise ValueError('Source identity changed: ' + str(path))
    config = yaml.safe_load(CONFIG.read_text())['use_mit_dual_routes']
    if config['jobs'] != matrix() or config['formal_budget'] != 59000 or config['alignment_weight'] != .2:
        raise ValueError('Budget/objective contract changed')
    if config['calibration_sha256'] != sha(CALIBRATION) or config['public_calibration_sha256'] != sha(PUBLIC):
        raise ValueError('Calibration changed')
    for name in ('execution.yaml', 'engineering.yaml'):
        rt.write_json(BASE / 'engineering' / ('protocol_' + name + '.json'), audit_config(CONFIG.parent / name))


def audit_rgb():
    """Re-read every member's actual arrays; keep the original common membership."""
    preflight()
    _, stats, audit, members = pt.inputs('multisensor')
    rows = [json.loads(line) for line in MEMBERS.read_text().splitlines()]
    counts, failures = {}, []
    path = BASE / 'engineering/rgb_members.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as output:
        for index, row in enumerate(rows):
            record = dict(file=row['file'], role=row['role'], sensor=row['sensor'], sha256=row['sha256'])
            try:
                actual = audit_cache(EXTERNAL, row)
                indices = rgb_indices(row['band_ids'])
                with np.load(EXTERNAL / row['file'], allow_pickle=False) as arrays:
                    x, valid, metadata, known = [arrays[k] for k in ('x', 'pixel_valid', 'metadata', 'known')]
                    waves = metadata[indices, 0]
                    expected = [.665, .560, .490] if row['sensor'] == 's2' else [.65, .56, .48]
                    if not known[indices, 0].all() or np.max(np.abs(waves - expected)) > .035:
                        raise ValueError('RGB wavelength/source mismatch')
                    common = valid[indices].all(0)
                    support = support_fraction(torch.from_numpy(common[None]), 7, 4, 3)
                    if not (support >= .8).any():
                        raise ValueError('No teacher RGB support')
                    transforms = row.get('physical_transforms')
                    record.update(rgb_indices=indices, wavelength_um=waves.tolist(),
                        rgb_valid_ratio=float(common.mean()), teacher_valid_ratio=float((support >= .8).float().mean()),
                        physical_transforms=transforms, cache_units='physical reflectance; cached transform already applied',
                        rgb_min=float(x[indices][valid[indices]].min()), rgb_max=float(x[indices][valid[indices]].max()),
                        actual=actual, passed=True)
                key = row['role'] + ':' + row['sensor']
                counts[key] = counts.get(key, 0) + 1
            except Exception as error:
                record.update(passed=False, error=str(error)); failures.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            if (index + 1) % 500 == 0:
                output.flush(); print(json.dumps(dict(audited=index + 1)), flush=True)
    expected = {'TRAIN:s2':9510, 'TRAIN:landsat':1622, 'PRETRAIN_VAL:s2':521, 'PRETRAIN_VAL:landsat':78,
                'GUARD:s2':239}
    # GUARD is recorded for provenance only and never passed to the training loader.
    passed = not failures and all(counts.get(k) == n for k, n in expected.items() if not k.startswith('GUARD'))
    result = dict(status='PASS' if passed else 'BLOCKED', counts=counts, failures=failures,
        members_sha256=members, previous_cache_audit_sha256=audit, records_sha256=sha(path),
        statistics=stats, statistics_sha256=hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest(),
        limitation='Source product provenance/scale audit reused; actual cached RGB arrays, quality, metadata and bytes re-read. Negative reflectance preserved.')
    rt.write_json(BASE / 'engineering/rgb_acceptance.json', result, True)
    if not passed:
        raise ValueError('RGB member audit failed; membership not changed')
    return result


def load_rows(rows, stats, device, augmentation_seed=None):
    observation = pt.load_rows(rows, stats, device, augmentation_seed)
    for provenance, row in zip(observation['spatial_metadata'], rows):
        provenance.update(normalization_mean=stats[row['sensor']]['mean'],
                          normalization_std=stats[row['sensor']]['std'],
                          units='sensor_train_zscore', sensor=row['sensor'])
    return observation


def load_encoder(route, source=None):
    pt.deterministic(); random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
    encoder = DualRouteEncoder(route)
    shared_path = BASE/'engineering/common_initial.pt'
    if shared_path.exists():
        shared = torch.load(shared_path,map_location='cpu',weights_only=False)
        current = encoder.state_dict()
        for name,value in shared['frontend'].items():
            if not torch.equal(current[name],value): raise ValueError('Common spectral source differs: '+name)
            current[name] = value.clone()
        encoder.load_state_dict(current,strict=True)
        if route != 'C0': encoder.adapter.load_state_dict(shared['adapter'],strict=True)
    if source:
        saved = torch.load(source, map_location='cpu', weights_only=False)
        encoder.load_state_dict(saved['use'], strict=True)
        if packed_state_sha256(encoder) != saved['use_sha256']:
            raise ValueError('Encoder bytes mismatch')
    return encoder


def segmenter(route, device='cpu', source=None):
    encoder=load_encoder(route,source);torch.manual_seed(1235)
    model=Segmenter(encoder)
    path=BASE/'engineering/common_initial.pt'
    if path.exists():
        shared=torch.load(path,map_location='cpu',weights_only=False)['task_head']
        current=model.state_dict()
        for name,value in shared.items():
            if not torch.equal(current[name],value):raise ValueError('Shared task head initialization differs: '+name)
            current[name]=value.clone()
        model.load_state_dict(current,strict=True)
    return model.to(device)


def initialize(route, device='cpu', task=None, source=None):
    encoder = load_encoder(route, source)
    torch.manual_seed(1235)
    model = Reconstruction(encoder, task).to(device)
    if task is None and route == 'B':
        model.teacher = PatchTeacher(encoder).to(device)
    return model


def optimizer(model):
    if model.task is not None:
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=6e-4, weight_decay=.01)
    front, new = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (front if name.split('.')[0] == 'use' and name.split('.')[1] in FRONTEND else new).append(p)
    return torch.optim.AdamW([dict(params=front, lr=1e-4, peak_lr=1e-4),
                             dict(params=new, lr=6e-4, peak_lr=6e-4)], weight_decay=.01)


def views(model, original, seed, block, include_alignment=True):
    features, raw, counts, gradients, coverage, aligns = [], {}, {}, [], [], []
    valid = original['pixel_valid'] & original['band_valid'][:, :, None, None]
    teacher = model.teacher(original) if include_alignment and hasattr(model, 'teacher') else None
    for i, task in enumerate(('spectral', 'fine', 'context')):
        view, hidden, summary = pt.task_view(original, task, seed+i, block=block, metadata_mode='known')
        feature = model.features(view); features.append(feature)
        pred = model.predict(feature, view['band_metadata'], view['known_mask'], summary)
        raw[task] = pt.spatial_error(pred, original['x'], hidden, valid) if task == 'context' else pt.mean_error(pred, original['x'], hidden)
        counts[task] = int(hidden.sum())
        grad, count = pt.gradient_error(pred, original['x'], hidden, valid)
        gradients.append(grad); counts['gradient_'+task] = count
        if teacher:
            loss, report = alignment_loss(feature['patch_tokens'], *teacher, view)
            aligns.append(loss); coverage.append(dict(view=task, **report))
    raw['gradient'] = torch.stack(gradients).mean()
    raw['consistency'], counts['consistency'] = pt.consistency(features[0], features[1])
    return raw, counts, torch.stack(aligns).mean() if aligns else original['x'].new_zeros(()), coverage


def update(model, opt, split, stats, step, calibration, device):
    rows = pt.sample_rows(split, step); opt.zero_grad(set_to_none=True)
    records = []
    for micro in range(4):
        current = rows[micro*2:micro*2+2]
        original = load_rows(current, stats, device, 1234+step*8+micro*2)
        raw, counts, align, coverage = views(model, original, 1234+step*100+micro*3, (16,32,64)[(step-1)%3])
        denom = calibration['coefficients'][current[0]['sensor']]
        weighted = {k: v/denom[k]*(.1 if k in ('gradient','consistency') else 1)/3.2 for k,v in raw.items()}
        mae = sum(weighted.values())
        ((mae+.20*align)/4).backward()
        records.append(dict(sensor=current[0]['sensor'], raw={k:float(v.detach()) for k,v in raw.items()},
            weighted={k:float(v.detach()) for k,v in weighted.items()}, mae=float(mae.detach()),
            align=float(align.detach()), align_weighted=float(.2*align.detach()), cosine=1-float(align.detach()),
            counts=counts, coverage=coverage, transforms=original['spatial_metadata']))
    gradients = {name: float(p.grad.norm()) for name,p in model.use.named_parameters()
                 if p.grad is not None and (name.startswith(('adapter.', 'residual_zero.', 'readout.')))}
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    if not torch.isfinite(norm):
        raise ValueError('Nonfinite gradients')
    v4.check_frozen_boundary(model, opt, 'pretrain')
    if hasattr(model, 'teacher') and any(p.requires_grad or p.grad is not None for p in model.teacher.parameters()):
        raise ValueError('Teacher gradients')
    opt.step()
    return dict(step=step, gradient_norm=float(norm), path_gradients=gradients, microbatches=records,
                files=[r['file'] for r in rows], sensors=[r['sensor'] for r in rows])


@torch.no_grad()
def validate(model, splits, stats, device, step):
    model.eval(); result = {}
    for sensor, rows in splits['PRETRAIN_VAL'].items():
        records = []
        for i,row in enumerate(rows):
            raw, counts, align, coverage = views(model, load_rows([row], stats, device), 800000+i*3, (16,32,64)[i%3], include_alignment=False)
            records.append(dict(file=row['file'], raw={k:float(v) for k,v in raw.items()}, counts=counts))
        result[sensor] = dict(rows=records, means={k:float(np.mean([r['raw'][k] for r in records])) for k in records[0]['raw']})
    model.train()
    return dict(step=step, sensors=result, role='PRETRAIN_VAL', selection_eligible=False)


def probe_update(model, opt, split, stats, step, task, device):
    rows = pt.sample_rows(split, step); opt.zero_grad(set_to_none=True); losses = []
    for micro in range(4):
        original = load_rows(rows[micro*2:micro*2+2], stats, device, 1234+step*8+micro*2)
        tasks = ('fine',) if task == 'fine' else ('spectral','context')
        for name in tasks:
            view, hidden, summary = pt.task_view(original, name, 1234+step*100+micro*3,
                block=(16,32,64)[(step-1)%3], metadata_mode='known')
            pred = model(view, view['band_metadata'], view['known_mask'], summary)
            valid = original['pixel_valid'] & original['band_valid'][:,:,None,None]
            loss = pt.spatial_error(pred, original['x'], hidden, valid) if name == 'context' else pt.mean_error(pred, original['x'], hidden)
            (loss/(4*len(tasks))).backward(); losses.append(float(loss.detach()))
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    if not torch.isfinite(norm): raise ValueError('Nonfinite probe gradients')
    opt.step()
    return dict(step=step, loss=float(np.mean(losses)), gradient_norm=float(norm), files=[r['file'] for r in rows])


class PhysicalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    @property
    def records(self): return self.dataset.records
    def set_epoch(self, step): self.dataset.set_epoch(step)
    def __getitem__(self, index):
        sample = self.dataset[index]
        obs = sample['image']
        from landslide_seg.data.segformer_dataset import S2_10_BANDS
        indices = [S2_10_BANDS.index(b) for b in obs.band_ids]
        obs.spatial_metadata.update(normalization_mean=[self.dataset.stats['mean'][i]*.0001 for i in indices],
                                    normalization_std=[self.dataset.stats['std'][i]*.0001 for i in indices])
        return sample


def datasets(config):
    return tuple(PhysicalDataset(d) for d in rt.datasets(config))


def source_files():
    # Pin actual dependencies, without binding unrelated mutable research documents.
    return sorted(list((ROOT/'src/landslide_seg').rglob('*.py')) +
        [p for p in (ROOT/'scripts').glob('*.py') if p.name not in ('diagnose_use_v4_bn_residual.py','diagnose_use_v4_interface.py')] +
        [ROOT/'tests/test_use_mit_dual_routes.py'])


def verify_release(release):
    release = Path(release); manifest = rt.read(release/'manifest.json')
    for name,digest in manifest['files'].items():
        if sha(release/'snapshot'/name) != digest or sha(ROOT/name) != digest:
            raise ValueError('Release/source changed: '+name)
    return manifest


def identity(job, stats, release_sha=None):
    return dict(job=job, release_sha256=release_sha, config_sha256=sha(CONFIG), overrides=[],
        members_sha256=sha(MEMBERS), rgb_audit_sha256=sha(BASE/'engineering/rgb_acceptance.json'),
        calibration_sha256=sha(CALIBRATION), public_calibration_sha256=sha(PUBLIC),
        statistics=v4.statistics_identity(yaml.safe_load(CONFIG.read_text()), stats),
        common_initial_sha256=sha(BASE/'engineering/common_initial.pt'),
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in source_files()})


def residual_diagnostic(model, splits, stats, device):
    if model.use.route != 'A': return None
    row = splits['PRETRAIN_VAL']['s2'][0]
    original = load_rows([row], stats, device)
    with torch.no_grad():
        feature = model.use(original, diagnostics=True)
        rgb = feature['mit_input']-feature['residual']
        ratio = float(feature['residual'].square().mean().sqrt()/rgb.square().mean().sqrt().clamp_min(1e-12))
        full = views(model, original, 800000, 32)[0]
        state = copy.deepcopy(model.use.residual_zero.state_dict())
        try:
            for p in model.use.residual_zero.parameters(): p.zero_()
            zero = views(model, original, 800000, 32)[0]
        finally: model.use.residual_zero.load_state_dict(state)
    return dict(file=row['file'], residual_rgb_rms_ratio=ratio,
                full={k:float(v) for k,v in full.items()}, residual_zero={k:float(v) for k,v in zero.items()})


def execute(job_id, device, release=None, marker=None, resume=False, engineering=False, route=None):
    preflight(); pt.deterministic()
    if engineering:
        if route not in ('A','B'): raise ValueError('Engineering route A/B required')
        job_id = 'pretrain_'+route
        out = BASE/'engineering'/('trial_'+route)
        release_sha = None
    else:
        verify_release(release); release_sha = sha(Path(release)/'manifest.json'); proof = rt.read(marker)
        if proof.get('status') != 'READY_FOR_TRAINING' or proof.get('release_sha256') != release_sha:
            raise ValueError('Independent release-bound READY_FOR_TRAINING required')
        out = BASE/'runs'/job_id
    job = matrix()[job_id]; route = job['route']; mode = job['mode']
    if rt.read(BASE/'engineering/rgb_acceptance.json')['status'] != 'PASS': raise ValueError('RGB audit required')
    splits, stats, _, _ = pt.inputs('multisensor'); config = yaml.safe_load(CONFIG.read_text())
    out.mkdir(parents=True, exist_ok=True)
    with v4.guarded_device(device) as gate:
        rt.write_json(out/'device_gate.json', gate)
        source = None if mode == 'pretrain' or route == 'C0' else BASE/f'runs/pretrain_{route}/u20000.pt'
        if mode == 'segmentation':
            model = segmenter(route,device,source)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=6e-4,weight_decay=.01)
            train, val = datasets(config)
        else:
            model = initialize(route, device, job.get('task'), source); opt = optimizer(model)
        actual_identity = identity(job, stats, release_sha)
        actual_identity.update(source_checkpoint_sha256=sha(source) if source else None,
                               engineering=engineering, stop_updates=200 if engineering else job['steps'])
        reference = packed_state_sha256(v4.check_frozen_boundary(model,opt,mode))
        history = []; start = 0; best = None
        if resume:
            saved = pt.restore(model,opt,out/'last.pt',actual_identity)
            history, start, best = saved['history'],saved['step'],saved['best']
            v4.verify_frozen(model,opt,mode,reference)
        else:
            if (out/'initial.pt').exists(): raise ValueError('Existing run; explicit --resume required')
            v4.save_checked(model,opt,0,None,actual_identity,[],out/'initial.pt',reference,mode)
            rt.write_json(out/'parameter_groups.json', dict(groups=[dict(lr=g['lr'], names=[n for n,p in model.named_parameters() if any(p is q for q in g['params'])], parameters=sum(p.numel() for p in g['params'])) for g in opt.param_groups],
                encoder_sha256=packed_state_sha256(model.use.encoder if hasattr(model.use,'encoder') else model.use)))
            if mode == 'pretrain': rt.write_json(out/'residual_initial.json',residual_diagnostic(model,splits,stats,device))
        stop = 200 if engineering else job['steps']
        calibration = rt.read(CALIBRATION)
        for step in range(start+1,stop+1):
            model.train(); v4.check_frozen_boundary(model,opt,mode)
            rates = v4.schedule(opt,step,job); began = time.perf_counter()
            cuda = torch.device(device).type == 'cuda'
            if cuda: torch.cuda.reset_peak_memory_stats(device)
            if mode == 'pretrain': record = update(model,opt,splits['TRAIN'],stats,step,calibration,device)
            elif mode == 'probe': record = probe_update(model,opt,splits['TRAIN'],stats,step,job['task'],device)
            else: record = v4.segment_update(model,opt,train,step,config,device)
            if cuda: torch.cuda.synchronize(device)
            record.update(lr=rates,seconds=time.perf_counter()-began,
                peak_allocated=torch.cuda.max_memory_allocated(device) if cuda else None,
                peak_reserved=torch.cuda.max_memory_reserved(device) if cuda else None)
            history.append(record); v4.check_frozen_boundary(model,opt,mode)
            if engineering and route == 'B':
                empty = [not any(any(n > 0 for n in c['tokens']) for m in h['microbatches'] for c in m['coverage']) for h in history]
                if sum(empty) > .1*len(empty): raise ValueError('More than 10% updates have no alignment')
            interval = 50 if engineering else 1000
            if mode == 'pretrain' and step%interval == 0:
                rt.write_json(out/f'validation_{step}.json',validate(model,splits,stats,device,step))
            if mode == 'segmentation' and step%500 == 0:
                evaluation = rt.downstream_evaluate(model,val,config,device)
                rt.write_json(out/f'eval_{step}.json',evaluation)
                if best is None or evaluation['event_mean_iou'] > best:
                    best = evaluation['event_mean_iou']
                    v4.save_checked(model,opt,step,best,actual_identity,history,out/'best.pt',reference,mode)
            if step%10 == 0 or step == stop:
                rt.write_json(out/'curve.json',history)
                v4.save_checked(model,opt,step,best,actual_identity,history,out/'last.pt',reference,mode)
                print(json.dumps(dict(job=job_id,engineering=engineering,step=step,seconds=record['seconds'])),flush=True)
            if engineering and step == 199:
                v4.save_checked(model,opt,step,best,actual_identity,history,out/'recovery_pre.pt',reference,mode)
            if mode == 'pretrain' and (step in (5000,10000,20000) or step == stop):
                v4.save_checked(model,opt,step,best,actual_identity,history,out/f'u{step}.pt',reference,mode)
                rt.write_json(out/f'residual_{step}.json',residual_diagnostic(model,splits,stats,device))
        if engineering and route == 'B':
            for sensor in ('s2','landsat'):
                if not any(any(n>0 for n in c['tokens']) for h in history for m in h['microbatches'] if m['sensor']==sensor for c in m['coverage']):
                    raise ValueError('Sensor has no alignment')
        rt.write_json(out/'summary.json',dict(complete=True,updates=stop,engineering=engineering,
            formal_updates=0 if engineering else stop,identity=actual_identity,
            frozen_sha256=reference,frozen_final_sha256=v4.verify_frozen(model,opt,mode,reference),
            initial_sha256=sha(out/'initial.pt'),last_sha256=sha(out/'last.pt')))


def freeze(name):
    verification = rt.read(BASE/'engineering/verification.json')
    if verification['status'] != 'PASS' or set(verification.get('recovery',{})) != {'A','B'}:
        raise ValueError('Verification and both complete optimizer/RNG recovery replays required')
    for route in ('A','B'):
        trial = rt.read(BASE/f'engineering/trial_{route}/summary.json')
        if not trial['complete'] or not 1 <= trial['updates'] <= 200: raise ValueError('Engineering trial required')
    release = BASE/'releases'/name; release.mkdir(parents=True,exist_ok=False)
    files = source_files()+list(CONFIG.parent.glob('*'))+list((ROOT/'manifests/v4r1').glob('*'))
    files += [V3,CALIBRATION,PUBLIC,MEMBERS,ROOT/'outputs/use_independent_frozen/cache_audit_v1/acceptance.json',
              ROOT/'outputs/use/pretraining/resolved_config.json',ROOT/'docs/assets/research/research_policy.json']+list(MIT.glob('*'))
    files += list((BASE/'engineering').rglob('*'))
    hashes = {}
    for path in sorted(set(p for p in files if p.is_file())):
        name_ = path.relative_to(ROOT).as_posix(); hashes[name_] = sha(path)
        target = release/'snapshot'/name_; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    manifest = dict(files=hashes,jobs=matrix(),status='FROZEN_FOR_REVIEW',engineering_updates=400,formal_updates=0,
        environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                         packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    rt.write_json(release/'manifest.json',manifest,True); verify_release(release)
    rt.write_json(release/'READY_FOR_AUDIT.json',dict(status='READY_FOR_AUDIT',release_sha256=sha(release/'manifest.json'),formal_updates=0),True)
    print(str(release),sha(release/'manifest.json'),flush=True)


def evaluate(job_id, checkpoint, device, metadata_mode='known'):
    job = matrix()[job_id]; out = BASE/'runs'/job_id; checkpoint = checkpoint or out/'last.pt'
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    with v4.guarded_device(device):
        route = job['route']; source = None if route == 'C0' else BASE/f'runs/pretrain_{route}/u20000.pt'
        if job['mode'] == 'probe':
            if saved['step'] != 1000: raise ValueError('Only final probe')
            model = initialize(route,device,job['task'],source); model.load_state_dict(saved['model'],strict=True)
            splits, stats, _, _ = pt.inputs('multisensor')
            # Public evaluator accepts the same observation schema; attach physical stats in its loader.
            result = evaluate_probe(model,job['task'],splits,stats,device,metadata_mode)
            target = out/('evaluation.json' if metadata_mode=='known' else f'evaluation_{metadata_mode}.json')
        elif job['mode'] == 'segmentation':
            model = segmenter(route,device,source)
            model.load_state_dict(saved['model'],strict=True)
            config = yaml.safe_load(CONFIG.read_text()); _,val = datasets(config)
            result = rt.downstream_evaluate(model,val,config,device,prediction_dir=out/'predictions'/checkpoint.stem)
            from evaluate_use_v4 import fixed_recall
            probabilities,targets = [],[]
            for path in sorted((out/'predictions'/checkpoint.stem).glob('*.npz')):
                with np.load(path) as data:
                    mask=data['valid'].astype(bool); probabilities.append(data['prob'][mask]); targets.append(data['mask'][mask])
            result['fixed_recall']=fixed_recall(np.concatenate(probabilities),np.concatenate(targets))
            target=out/f'evaluation_{checkpoint.stem}.json'
        else: raise ValueError('Pretraining has validation curves, no selection')
        result.update(checkpoint_sha256=sha(checkpoint),encoder_checkpoint='fixed u20000',metadata_mode=metadata_mode)
        rt.write_json(target,result); return result


@torch.no_grad()
def evaluate_probe(model, task, splits, stats, device, metadata_mode):
    model.eval(); result = {}; coefficients = rt.read(PUBLIC)['coefficients']
    for sensor,rows in splits['PRETRAIN_VAL'].items():
        records=[]
        for index,row in enumerate(rows):
            original=load_rows([row],stats,device); errors={}; physical={}
            std=torch.as_tensor(stats[sensor]['std'],device=device)[None,:,None,None]
            for name in (('fine',) if task=='fine' else ('spectral','context')):
                values=[]; physical_values=[]
                for block in ((16,32,64) if name=='context' else (32,)):
                    view,hidden,summary=pt.task_view(original,name,800000+index*3+('spectral','fine','context').index(name),block=block,metadata_mode=metadata_mode)
                    if not hidden.any(): continue
                    pred=model(view,view['band_metadata'],view['known_mask'],summary)
                    valid=original['pixel_valid']&original['band_valid'][:,:,None,None]
                    function=pt.spatial_error if name=='context' else lambda p,t,h,v:pt.mean_error(p,t,h)
                    values.append(float(function(pred,original['x'],hidden,valid)))
                    physical_values.append(float(function(pred*std,original['x']*std,hidden,valid)))
                if values:
                    errors[name]=float(np.mean(values)); physical[name]=float(np.mean(physical_values))
            records.append(dict(file=row['file'],raw_errors=errors,errors={k:v/coefficients[sensor][k] for k,v in errors.items()},physical_errors=physical))
        means = {k:float(np.mean([r['errors'][k] for r in records if k in r['errors']]))
                 if any(k in r['errors'] for r in records) else None
                 for k in (('fine',) if task=='fine' else ('spectral','context'))}
        result[sensor]=dict(rows=records,means=means)
    return dict(sensors=result,public_denominator_sha256=sha(PUBLIC),public_coefficients=coefficients,
                selection_eligible=False,role='PRETRAIN_VAL',probe_training_role='TRAIN')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','audit-rgb','verify','engineering','freeze','run','probe','segment','evaluate','benchmark'])
    parser.add_argument('--device',default='cuda:1'); parser.add_argument('--route',choices=['A','B','C0'])
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--job',choices=list(matrix()))
    parser.add_argument('--release',type=Path); parser.add_argument('--marker',type=Path)
    parser.add_argument('--name',default='engineering_v1'); parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--metadata-mode',choices=['known','partial','unknown'],default='known')
    args=parser.parse_args(); torch.set_num_threads(4)
    if args.action=='prepare': prepare()
    elif args.action=='audit-rgb': audit_rgb()
    elif args.action=='verify':
        if args.release: verify_release(args.release)
        else:
            from verify_use_mit_dual_routes import verify
            verify(args.device)
    elif args.action=='engineering': execute(None,args.device,resume=args.resume,engineering=True,route=args.route)
    elif args.action=='freeze': freeze(args.name)
    elif args.action=='evaluate': evaluate(args.job,args.checkpoint,args.device,args.metadata_mode)
    elif args.action=='benchmark':
        from verify_use_mit_dual_routes import benchmark
        benchmark(args.device,args.release)
    else:
        job=args.job or ('segment_'+args.route if args.action=='segment' else None)
        if job is None: parser.error('--job required')
        execute(job,args.device,args.release,args.marker,args.resume)


if __name__=='__main__': main()
