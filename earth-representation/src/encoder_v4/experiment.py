"""Reproducible manifests, short fits, checkpoint recovery and native evaluation."""
from collections import defaultdict
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch

from encoder_v3.data import sha256
from encoder_v3.latent_response.checkpoint import atomic_replace, digest, restore, snapshot
from encoder_v3.shared_mae.train import write

from .data import Band, Observation, OpticalDataset, Physics, derive_members, read_rows, task_view
from .geometry import Grid
from .model import Config, Pretraining, objective

TASKS = ('A', 'B1')
SENSORS = ('S2', 'L8', 'L9')
SCHEMA = 'pd_here_v1'


def setup(device, seed):
    torch.set_num_threads(2)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def coordinates(pack, observation, kind):
    return {b.key: np.flatnonzero(np.unpackbits(pack[f'{observation.id}/{kind}/{b.key}'],
                                               count=b.values.numel()))
            for b in observation.bands if f'{observation.id}/{kind}/{b.key}' in pack}


def prepare(args, dataset, config, settings):
    output = args.out / 'members'
    if (output / 'identity.json').exists():
        raise ValueError(f'Members already frozen: {output}')
    release = args.root / 'outputs/encoder_v3/data_download_plan/research_dataset_v2'
    roles, identity = derive_members(release)
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    for sensor in SENSORS:
        row = next(row for row in roles['TRAIN'] if row['sensor'] == sensor)
        obs = dataset.load(row, torch.device('cpu'))
        checks.append(dict(sensor=sensor, member=obs.id, bands=[
            dict(key=b.key, shape=b.grid.shape, affine=b.grid.affine, crs=b.grid.crs,
                 valid=int(b.valid.sum()), reflectance_range=[float(b.values[b.valid].min()),
                                                              float(b.values[b.valid].max())],
                 physics={key: value.status for key, value in vars(b.physics).items()})
            for b in obs.bands if b.valid.any()]))
    identity['native_input_checks'] = checks
    identity['roles'] = {}
    identity['panels'] = {}
    identity['seed'] = settings['seed']
    identity['panel_stride'] = config.stride
    for role, rows in roles.items():
        path = output / f'{role.lower()}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        identity['roles'][role] = sha256(path)
        if role == 'TRAIN':
            continue
        targets = {}
        skipped = []
        for row in rows:
            obs = dataset.load(row, torch.device('cpu'), qa_only=True)
            shapes = {b.key: b.grid.shape for b in obs.bands}
            for kind in TASKS:
                view = task_view(obs, kind, config.stride, seed=settings['seed'])
                if view is None:
                    skipped.append(dict(id=obs.id, task=kind, reason='no legal target'))
                    continue
                for key, request in zip(view.keys, view.requests, strict=True):
                    mask = np.zeros(math.prod(shapes[key]), np.uint8)
                    mask[request.pixels.numpy()] = 1
                    targets[f'{obs.id}/{kind}/{key}'] = np.packbits(mask)
        path = output / f'{role.lower()}_panel.npz'
        np.savez_compressed(path, **targets)
        identity['panels'][role] = dict(sha256=sha256(path), band_masks=len(targets), skipped=skipped)
        print(f'{role}: {len(rows)} members, {len(targets)} native masks', flush=True)
    write(output / 'identity.json', identity)
    print(json.dumps(identity['counts']), flush=True)


def read_members(out, role):
    folder = out / 'members'
    identity = json.loads((folder / 'identity.json').read_text(encoding='utf-8'))
    manifest = folder / f'{role.lower()}.jsonl'
    if identity['roles'][role] != sha256(manifest):
        raise ValueError(f'Derived {role} members changed')
    return read_rows(manifest), identity


def bind(args, config, settings, members):
    source = Path(__file__).parent
    v3 = args.root / 'src/encoder_v3'
    files = {f'encoder_v4/{p.name}': sha256(p) for p in source.glob('*.py')}
    for name in ('hierarchical/geometry.py', 'tasks.py', 'latent_response/checkpoint.py',
                 'shared_mae/metadata.py', 'data.py'):
        files[f'encoder_v3/{name}'] = sha256(v3 / name)
    catalog = args.root / 'data/physics'
    usable = json.loads((catalog / 'sensor_physics.json').read_text(encoding='utf-8'))['sensors']
    physical = {'sensor_physics.json': sha256(catalog / 'sensor_physics.json')}
    for key, item in usable.items():
        if key.startswith('S2') or key.startswith('LANDSAT'):
            file = item['srf']['csv']
            physical[file] = sha256(catalog / file)
    return dict(schema=SCHEMA, model=asdict(config), settings=settings, source=files, physics=physical,
                members=members['roles'], panels={k: v['sha256'] for k, v in members['panels'].items()})


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def capacity(args, dataset, config, settings, device):
    rows, _ = read_members(args.out, 'TRAIN')
    setup(device, settings['seed'])
    model = Pretraining(config).to(device)
    samples = []
    for sensor in SENSORS:
        row = max((r for r in rows if r['sensor'] == sensor),
                  key=lambda r: sum(math.prod(b['shape']) for b in r['bands']))
        started = time.perf_counter()
        obs = dataset.load(row, device)
        sync(device)
        samples.append((obs, time.perf_counter() - started))
    grid = Grid((128, 128), (30., 0., 500000., 0., -30., 4200000.), 'EPSG:32633')
    values = torch.rand((224, *grid.shape), device=device)
    valid = torch.ones(grid.shape, device=device, dtype=torch.bool)
    samples.append((Observation('synthetic-224', 'synthetic', tuple(
        Band(str(i), image, valid, grid, Physics()) for i, image in enumerate(values))), 0.))
    report = []
    for obs, load_time in samples:
        for task in TASKS:
            model.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            view = task_view(obs, task, config.stride)
            sync(device)
            geometry_time = time.perf_counter() - started
            if view is None:
                raise ValueError(f'Capacity case has no legal target: {obs.id}/{task}')
            record = dict(id=obs.id, sensor=obs.sensor, task=task, bands=len(obs.bands),
                          visible_bands=len(view.visible.bands), native_pixels=sum(b.values.numel() for b in obs.bands),
                          targets=sum(t.numel() for t in view.truth), load_seconds=load_time,
                          geometry_seconds=geometry_time, work_positions=math.prod(view.visible.work.shape),
                          slots=config.slots + 1, position_chunk=config.position_chunk, target_chunk=config.target_chunk)
            started = time.perf_counter()
            try:
                pyramid = model.encoder(view.visible)
                prediction = model.decoder(pyramid, view.requests)
                loss = objective(prediction, view.truth)
                sync(device)
                forward = time.perf_counter() - started
                started = time.perf_counter()
                loss.backward()
                sync(device)
                norms = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
                shapes = [tuple(stage.values.shape) for stage in pyramid.stages]
                record.update(status='PASS', stage_shapes=shapes, loss=float(loss.detach()),
                              grad_norm=float(norms), forward_seconds=forward,
                              backward_seconds=time.perf_counter() - started,
                              position_scores=[(h*w, heads, t, t) for (h,w,t,_),heads in zip(shapes,(4,4,8))],
                              window_scores=[(heads, min(h,8)*min(w,8), min(h,8)*min(w,8))
                                             for (h,w,_,_),heads in zip(shapes,(4,4,8))])
            except torch.OutOfMemoryError as error:
                record.update(status='OOM', error=str(error))
            record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
            report.append(record)
            write(args.out / 'capacity.json', report)
            print(json.dumps(record), flush=True)
            if record['status'] != 'PASS':
                raise SystemExit(1)
            del pyramid, prediction, loss, view


def select_fit(args, dataset, config, settings, device):
    rows, members = read_members(args.out, 'TRAIN')
    rng = random.Random(settings['seed'])
    selection = []
    for sensor, total in zip(SENSORS, (3, 3, 2)):
        choices = [r for r in rows if r['sensor'] == sensor]
        rng.shuffle(choices)
        accepted = 0
        for row in choices:
            obs = dataset.load(row, device)
            views = {task: task_view(obs, task, config.stride, seed=settings['seed']) for task in TASKS}
            if all(views.values()):
                selection.append((row['id'], views))
                accepted += 1
                if accepted == total:
                    break
        if accepted != total:
            raise ValueError(f'Unable to select legal A and B1 fit examples for {sensor}')
    return selection, bind(args, config, settings, members)


def save_fit(path, model, optimizer, scheduler, state, identity):
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    os.close(fd)
    try:
        torch.save(dict(schema=SCHEMA, identity=identity, **snapshot(model, optimizer, state, scheduler)), temp)
        atomic_replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


@torch.no_grad()
def fit_scores(model, selection, kinds):
    model.eval()
    values = {kind: sum(float(objective(model(views[kind]), views[kind].truth))
                        for _, views in selection) / len(selection) for kind in kinds}
    model.train()
    return values


def train_fit(path, selection, binding, config, settings, device, task, steps, stop, resume=False):
    if not 1 <= stop <= steps <= 200:
        raise ValueError('Authorized short-fit limit is 200 updates per task')
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = path / 'last.pt'
    if checkpoint.exists() != resume:
        raise ValueError(f'{path} requires a new run or an explicit resume of an existing checkpoint')
    setup(device, settings['seed'])
    model = Pretraining(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=settings['lr'] / 10)
    kinds = TASKS if task == 'joint' else (task,)
    identity = dict(**binding, task=task, updates=steps, selection=[name for name, _ in selection], device=str(device))
    if resume:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if payload['schema'] != SCHEMA or payload['identity'] != identity:
            raise ValueError('PD-HERE checkpoint configuration, code, or members changed')
        state = {}
        restore(payload, model, optimizer, state, scheduler)
        if not state['cursor'] < stop:
            raise ValueError('Resume must advance the checkpoint cursor')
    else:
        state = dict(cursor=0, history=[], initial=fit_scores(model, selection, kinds))
        write(path / 'identity.json', dict(**identity, command=sys.argv))
        save_fit(checkpoint, model, optimizer, scheduler, state, identity)
    started = time.perf_counter()
    for update in range(state['cursor'], stop):
        before = time.perf_counter()
        kind = kinds[update % len(kinds)]
        member, views = selection[(update // len(kinds)) % len(selection)]
        view = views[kind]
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(view), view.truth)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite {kind} training loss for {member}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        lr = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        state['cursor'] = update + 1
        state['history'].append(dict(update=update + 1, task=kind, member=member, loss=float(loss.detach()),
                                     lr=lr, gradient_norm=float(norm), seconds=time.perf_counter() - before))
        if (update + 1) % settings['checkpoint_every'] == 0 or update + 1 == stop:
            save_fit(checkpoint, model, optimizer, scheduler, state, identity)
            print(f'{task}: {update + 1}/{stop} {kind} {float(loss.detach()):.7f}', flush=True)
    final = fit_scores(model, selection, kinds)
    summary = dict(task=task, updates=stop, target_updates=steps, initial=state['initial'], final=final,
                   improved={k: final[k] < state['initial'][k] for k in kinds},
                   seconds=time.perf_counter() - started, next_task=kinds[stop % len(kinds)],
                   member_cursor=stop // len(kinds), history=state['history'])
    write(path / 'fit.json', summary)
    return torch.load(checkpoint, map_location='cpu', weights_only=False), summary


def recovery(args, selection, binding, config, settings, device):
    location = args.out / 'recovery'
    flags = (selection, binding, config, settings, device, 'joint', 4)
    expected, _ = train_fit(location / 'continuous', *flags, stop=4)
    train_fit(location / 'interrupted', *flags, stop=1)
    actual, resumed = train_fit(location / 'interrupted', *flags, stop=4, resume=True)
    if (expected['identity'] != actual['identity'] or expected['scheduler'] != actual['scheduler']
            or expected['runner']['cursor'] != actual['runner']['cursor']):
        raise ValueError('Recovery changed release identity, schedule or cursor')
    torch.testing.assert_close(expected['model'], actual['model'])
    torch.testing.assert_close(expected['optimizer'], actual['optimizer'])
    torch.testing.assert_close(expected['rng']['cpu'], actual['rng']['cpu'], rtol=0, atol=0)
    torch.testing.assert_close(expected['rng']['cuda'], actual['rng']['cuda'], rtol=0, atol=0)
    torch.testing.assert_close(expected['rng']['numpy'][1], actual['rng']['numpy'][1], rtol=0, atol=0)
    if (expected['rng']['python'] != actual['rng']['python']
            or expected['rng']['numpy'][0] != actual['rng']['numpy'][0]
            or expected['rng']['numpy'][2:] != actual['rng']['numpy'][2:]):
        raise ValueError('Recovery changed host RNG state')
    compared = ('update', 'task', 'member', 'lr')
    task_schedule = lambda p: [tuple(row[k] for k in compared) for row in p['runner']['history']]
    if task_schedule(expected) != task_schedule(actual):
        raise ValueError('Recovery changed the member or task sequence')
    write(location / 'result.json', dict(status='PASS', interrupted_after=1, resumed_at=2, updates=4,
                                         next_task=resumed['next_task'], member_cursor=resumed['member_cursor'],
                                         checked=['model', 'optimizer', 'scheduler', 'RNG', 'task/member cursor', 'release']))


def native_metrics(pred, truth):
    error = pred - truth
    finite = np.isfinite(error)
    values = error[finite]
    basic = dict(mse=float(np.mean(values**2)), mae=float(np.mean(abs(values))), bias=float(np.mean(values)))
    d_pred = np.concatenate([np.diff(pred, axis=i).ravel() for i in (0, 1)])
    d_truth = np.concatenate([np.diff(truth, axis=i).ravel() for i in (0, 1)])
    keep = np.isfinite(d_pred) & np.isfinite(d_truth)
    gradients = (d_pred[keep] - d_truth[keep])**2
    texture = abs(d_truth[keep])
    basic['gradient_mse'] = float(gradients.mean()) if len(gradients) else None
    basic['texture_mse'] = float(gradients[texture >= np.quantile(texture, .9)].mean()) if len(gradients) else None
    for scale in (2, 4):
        h, w = (math.ceil(n / scale) * scale for n in error.shape)
        pad = ((0, h - error.shape[0]), (0, w - error.shape[1]))
        padded = np.pad(np.where(finite, error, 0.), pad)
        mask = np.pad(finite, pad)
        mass = mask.reshape(h // scale, scale, w // scale, scale).sum((1, 3))
        sums = padded.reshape(h // scale, scale, w // scale, scale).sum((1, 3))
        basic[f'mse_{scale}x{scale}'] = float(np.mean((sums[mass > 0] / mass[mass > 0])**2))
    return basic


def metric_means(metrics):
    keys = tuple(metrics[0])
    return {key: float(np.mean([r[key] for r in metrics if r[key] is not None]))
            if any(r[key] is not None for r in metrics) else None for key in keys}


@torch.no_grad()
def evaluate(args, dataset, config, settings, device):
    role = args.split
    if role == 'TEST' and not args.test_selected_on_val:
        raise ValueError('Lock the VAL-selected checkpoint and authorize the held-out evaluation first')
    rows, members = read_members(args.out, role)
    binding = bind(args, config, settings, members)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['schema'] != SCHEMA or any(checkpoint['identity'].get(k) != v for k, v in binding.items()):
        raise ValueError('PD-HERE evaluation checkpoint and dataset are incompatible')
    model = Pretraining(config).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    panel = args.out / 'members' / f'{role.lower()}_panel.npz'
    if sha256(panel) != members['panels'][role]['sha256']:
        raise ValueError(f'{role} evaluation panel changed')
    path = args.checkpoint.parent / role.lower()
    path.mkdir(exist_ok=False)
    groups, groups_band, examples, skipped = defaultdict(list), defaultdict(list), set(), []
    with np.load(panel, allow_pickle=False) as fixed:
        for row in rows:
            obs = dataset.load(row, device)
            for kind in TASKS:
                view = task_view(obs, kind, config.stride, panel=coordinates(fixed, obs, kind))
                if view is None:
                    skipped.append(dict(id=obs.id, task=kind, reason='no legal target'))
                    continue
                pyramid = model.encoder(view.visible)
                predictions = model.decoder(pyramid, view.requests)
                by_band = []
                for key, request, pred, labels in zip(view.keys, view.requests, predictions, view.truth, strict=True):
                    pixels = request.pixels.cpu().numpy()
                    arrays = [np.full(request.grid.shape, np.nan, np.float32) for _ in range(2)]
                    arrays[0].flat[pixels] = pred.cpu().numpy()
                    arrays[1].flat[pixels] = labels.cpu().numpy()
                    if not np.isfinite(arrays[0].flat[pixels]).all():
                        raise ValueError(f'Nonfinite prediction: {obs.id}/{kind}/{key}')
                    metrics = native_metrics(*arrays)
                    groups_band[obs.sensor, kind, key].append(metrics)
                    by_band.append(metrics)
                    if (obs.sensor, kind) not in examples:
                        observed = pyramid.stages[0].support.flatten()[request.cells]
                        support = observed.new_zeros(len(pixels)).index_add_(0, request.output_index, observed * request.area)
                        support_map = np.full_like(arrays[0], np.nan)
                        support_map.flat[pixels] = support.cpu().numpy()
                        np.savez_compressed(path / f'{obs.sensor}_{kind}.npz', member=obs.id, band=key,
                                            affine=request.grid.affine, crs=request.grid.crs,
                                            prediction=arrays[0], truth=arrays[1], error=arrays[0]-arrays[1],
                                            support=support_map)
                        examples.add((obs.sensor, kind))
                observation = metric_means(by_band)
                groups[obs.sensor, kind].append(observation)
                groups['all', kind].append(observation)
    report = dict(split=role, observations=len(rows), skipped=skipped, members=members['roles'][role],
                  panel=members['panels'][role], checkpoint=sha256(args.checkpoint),
                  metrics=[dict(sensor=s, task=t, observations=len(rows), **metric_means(rows))
                           for (s,t), rows in sorted(groups.items())],
                  bands=[dict(sensor=s, task=t, band=b, observations=len(rows), **metric_means(rows))
                         for (s,t,b), rows in sorted(groups_band.items())])
    write(path / 'metrics.json', report)
    print(json.dumps(report['metrics']), flush=True)


def run(args):
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    config = Config(**settings.pop('model'))
    device = torch.device(args.device)
    torch.set_num_threads(2)
    dataset = OpticalDataset(args.root)
    if args.command == 'prepare':
        prepare(args, dataset, config, settings)
    elif args.command == 'capacity':
        capacity(args, dataset, config, settings, device)
    elif args.command == 'evaluate':
        setup(device, settings['seed'])
        evaluate(args, dataset, config, settings, device)
    elif args.command == 'reference':
        from .reference import evaluate_references
        evaluate_references(args.out, dataset, config.stride)
    else:
        setup(device, settings['seed'])
        selection, binding = select_fit(args, dataset, config, settings, device)
        if args.command == 'recovery':
            recovery(args, selection, binding, config, settings, device)
        else:
            folder = args.out / f'fit_{args.task}'
            _, summary = train_fit(folder, selection, binding, config, settings, device, args.task,
                                   args.steps, args.stop_after or args.steps, args.resume)
            print(json.dumps({k: v for k, v in summary.items() if k != 'history'}), flush=True)
