"""Train the adaptive two-branch model with the existing native A/B1 objective."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import re
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))

from encoder_v3.latent_response.tvlp_writer import RunWriter
from index_pd_here_s60_versions import CATALOG, index_versions, sync_viewer_labels, viewer_labels
from pd_here_geotoken_dualvit_r3.data import write_json
from pd_here_s60.mae_schedule import BASE_LR, BETAS, WEIGHT_DECAY, after_warmup, mae_lr
from pd_here_s60.whole_scene_adaptive_batch import adaptive_batch_totals, adaptive_signature
from pd_here_s60.whole_scene_adaptive_query import (
    AdaptiveQueryDualViT, MODEL_IDS, native_task_mse, pack_adaptive_scene,
)
from train_pd_here_s60_whole_scene import (
    CORE_SHA, EXTENSION_SHA, FROZEN, TASKS, ExpandedTrainingData, IneligiblePairedScene,
    digest, early_stop_state, masks as historical_masks, read, save, save_best,
)


def adaptive_masks(scene, parent, exposure):
    (selected, bands), b1 = historical_masks(scene, parent, exposure)
    # The historical scheduler samples a quarter of the legal groups; these are now the targets.
    return (~selected, bands), b1


def load_data(full):
    data = ExpandedTrainingData(ROOT)
    subset_path = FROZEN / 'subset_expanded_2000_seed17.json'
    if digest(subset_path) != read(FROZEN/'expanded2000_plan.json')['subset_sha256']:
        raise ValueError('Frozen expanded subset identity differs')
    subset = read(subset_path)
    rows = data.members() if full else subset['parents']
    train, cache, excluded = [], {}, []
    for row in rows:
        scale = subset['dn_scales'][row['domain']] if row['encoding'] == 'stored_dn' else 1.
        scene = pack_adaptive_scene(data.load(row, 'cpu'), scale, row['source'])
        try:
            adaptive_masks(scene, row['id'], 0)
        except IneligiblePairedScene:
            if not full:
                raise
            excluded.append(row['id'])
            continue
        train.append(row)
        cache[row['id']] = scene
    if full and (len(cache) != 13020 or excluded != ['FLAIR-HUB S2:4-14/22']):
        raise ValueError('Full A/B1 membership differs from the audited release')
    if not full and len(cache) != 2000:
        raise ValueError('Expected 2,000 unique TRAIN observations')
    validation, val_cache = [], {}
    for row in data.dataset.core_roles['VAL']:
        key, source = row['id'], f'Core {row["sensor"]}'
        if key in cache or key in val_cache:
            raise ValueError(f'Duplicate validation member: {key}')
        scene = pack_adaptive_scene(data.dataset.core.load(row, 'cpu'), source=source)
        adaptive_masks(scene, key, 0)
        validation.append(dict(id=key, source=source, domain='surface_reflectance'))
        val_cache[key] = scene
    if len(validation) != 275:
        raise ValueError('Expected the frozen 275 Core VAL observations')
    return train, cache, validation, val_cache


def prepare_execution(cache, device, execution):
    resident = {key: scene.to(device) for key, scene in cache.items()} if execution != 'serial' else None
    signatures = {key: adaptive_signature(scene) for key, scene in cache.items()} if execution == 'packed' else None
    return resident, signatures


def train_step(model, optimizer, rows, cache, resident, signatures, device, exposure):
    optimizer.zero_grad(set_to_none=True)
    if resident is None:
        totals = []
        for row in rows:
            scene = cache[row['id']].to(device)
            totals.append(torch.stack([
                native_task_mse(model, scene, view, task, backward_scale=.5/len(rows))
                for task, view in zip(TASKS, adaptive_masks(scene, row['id'], exposure), strict=True)
            ]))
        total = torch.stack(totals).sum(0)
    else:
        total = adaptive_batch_totals(model, rows, cache, resident, signatures, adaptive_masks, exposure, backward=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return total.tolist()


@torch.no_grad()
def validation_mse(model, rows, cache, resident, signatures, batch):
    model.eval()
    sources = defaultdict(list)
    for row in rows:
        sources[row['source']].append(row)
    scores = {task: {} for task in TASKS}
    for source, group in sources.items():
        totals = [adaptive_batch_totals(model, group[at:at+batch], cache, resident, signatures,
                                       adaptive_masks, 0, backward=False)
                  for at in range(0, len(group), batch)]
        for task, value in zip(TASKS, (torch.stack(totals).sum(0) / len(group)).tolist(), strict=True):
            scores[task][source] = dict(mse=value)
    model.train()
    return {task: dict(source_macro=dict(mse=sum(v['mse'] for v in score.values())/len(score)), by_source=score)
            for task, score in scores.items()}


def run(args, logs, weights, viewer, version):
    run_start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    train, cache, validation, val_cache = load_data(args.full)
    resident, signatures = prepare_execution(cache, args.device, args.execution)
    val_resident, val_signatures = prepare_execution(val_cache, args.device, args.execution)
    model = AdaptiveQueryDualViT(args.width).to(args.device)
    params = sum(parameter.numel() for parameter in model.parameters())
    updates_per_epoch = math.ceil(len(train) / args.batch)
    warmup = min(8, args.epochs - 1) if args.schedule == 'mae' else 0
    peak_lr = BASE_LR * args.batch / 256 if args.schedule == 'mae' else 1e-4
    policy = (dict(patience=args.patience, min_epochs=warmup + args.patience, relative_min_delta=.001)
              if args.patience else None)
    identity = dict(model_id=MODEL_IDS[args.width], model_name=version['name'], fusion_width=args.width,
                    parents=len(train), validation_parents=len(validation), core_sha256=CORE_SHA,
                    extension_sha256=EXTENSION_SHA, training_pool='full' if args.full else 'expanded2000',
                    subset_sha256=digest(FROZEN/'subset_expanded_2000_seed17.json'),
                    epochs=args.epochs, batch=args.batch, seed=17,
                    mask='A: hide a quarter of legal geographic groups; B1: hide one cyclic legal band',
                    warmup_epochs=warmup, early_stopping=policy, parameters=params,
                    execution=args.execution, schedule=args.schedule, peak_lr=peak_lr,
                    precision='BF16 autocast, FP32 weights and loss',
                    device=args.device,
                    weights=weights.relative_to(ROOT).as_posix())
    if args.schedule == 'mae':
        decay = [{'params': [p for p in model.parameters() if p.ndim > 1], 'weight_decay': WEIGHT_DECAY},
                 {'params': [p for p in model.parameters() if p.ndim == 1], 'weight_decay': 0.}]
        optimizer = torch.optim.AdamW(decay, lr=peak_lr, betas=BETAS)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=.04)
    history = []
    if args.resume:
        state = torch.load(weights/'last.pt', weights_only=True, map_location='cpu')
        if state['identity'] != identity:
            raise ValueError('Checkpoint belongs to a different run configuration')
        history = state['history']
        if state['epoch'] != len(history) or history[-1]['updates'] != updates_per_epoch * len(history):
            raise ValueError('Checkpoint progress differs from the epoch history')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], args.device)
    else:
        write_json(logs/'identity.json', identity)
        write_json(logs/'members.json', dict(train=train, validation=validation))
        write_json(logs/'preflight.json', dict(parameters=params, device=args.device))
    print(f'ready {len(train)} TRAIN / {len(validation)} VAL | {params:,} parameters | {args.execution}', flush=True)
    stop = history[-1]['early_stop'] if history else {}
    metric_names = [f'{task}_{kind}' for task in TASKS for kind in ('train_loss', 'val_mse')]
    metrics = {name: dict(label=name, direction='min', role='loss', unit='') for name in metric_names}
    experiment, variant = viewer_labels(version, len(train))
    if args.resume:
        sync_viewer_labels(viewer, version, len(train))
    variant.update(kind='modification',
                   description='Native-grid Query pooling, two-token fusion and shared native head')
    with RunWriter(viewer, experiment=experiment, variant=variant,
                   config=identity, metrics=metrics, charts=metric_names, resume=args.resume) as log:
        if history and stop['best_epoch'] == len(history):
            save_best(weights/'best.pt', model, identity, history[-1])
        final_epoch = len(history) if stop.get('triggered') else args.epochs
        for epoch in range(len(history) + 1, final_epoch + 1):
            started = time.perf_counter()
            order = train.copy()
            random.Random(17 + epoch).shuffle(order)
            totals = Counter()
            progress = dict(status='RUNNING', epochs=epoch-1, current_epoch=epoch,
                            max_epochs=args.epochs, viewer=str(viewer))
            write_json(logs/'status.json', progress | dict(updates=updates_per_epoch*(epoch-1)))
            for step, start in enumerate(range(0, len(order), args.batch)):
                group = order[start:start + args.batch]
                lr = (mae_lr(peak_lr, epoch-1 + step/updates_per_epoch, args.epochs, warmup)
                      if args.schedule == 'mae' else peak_lr)
                for parameters in optimizer.param_groups:
                    parameters['lr'] = lr
                values = train_step(model, optimizer, group, cache, resident, signatures, args.device, epoch-1)
                for task, value in zip(TASKS, values, strict=True):
                    totals[task] += value
                if (step+1) % 25 == 0:
                    updates = updates_per_epoch*(epoch-1)+step+1
                    write_json(logs/'status.json', progress | dict(updates=updates))
                    print(f'epoch {epoch} updates {updates} elapsed {time.perf_counter()-started:.1f}s', flush=True)
            train_seconds = time.perf_counter()-started
            val = validation_mse(model, validation, val_cache, val_resident, val_signatures, args.batch)
            scores = {task: val[task]['source_macro']['mse'] for task in TASKS}
            score = sum(scores.values()) / len(TASKS)
            stop, is_best = early_stop_state(stop, score, epoch, policy)
            if policy:
                after_warmup(stop, epoch, warmup)
            train_loss = {task: totals[task]/len(train) for task in TASKS}
            item = dict(epoch=epoch, updates=updates_per_epoch*epoch, train=train_loss, validation=val,
                        validation_paired_mse=score, early_stop=stop, lr=lr,
                        train_seconds=train_seconds, seconds=time.perf_counter()-started)
            history.append(item)
            save(weights/'last.pt', model, optimizer, epoch, identity, history, args.device)
            if is_best:
                save_best(weights/'best.pt', model, identity, item)
            write_json(logs/'history.json', history)
            log.log(item['updates'], {f'{t}_train_loss': train_loss[t] for t in TASKS} |
                    {f'{t}_val_mse': scores[t] for t in TASKS}, epoch=epoch)
            print(f'{epoch} TRAIN {train_loss} VAL {scores} seconds={item["seconds"]:.2f}', flush=True)
            if stop.get('triggered'):
                break
    write_json(logs/'timing.json', dict(train_and_epoch_eval_seconds=sum(x['seconds'] for x in history),
                                      total_wall_seconds_this_session=time.perf_counter()-run_start))
    write_json(logs/'status.json', dict(status='COMPLETED', epochs=len(history), updates=history[-1]['updates'],
                                      early_stop=stop, viewer=str(viewer),
                                      stop_reason='early_stopping' if stop.get('triggered') else 'max_epochs'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Unique lowercase Run name')
    parser.add_argument('--width', type=int, choices=MODEL_IDS, default=256)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execution', choices=('serial','resident','packed'), default='serial')
    parser.add_argument('--schedule', choices=('constant','mae'), default='mae')
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--full', action='store_true', help='Use the audited 13,020 paired TRAIN observations')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9][a-z0-9_-]*', args.run):
        parser.error('Run name must use lowercase letters, digits, underscores and hyphens')
    if not (1 <= args.batch <= 32 and args.epochs > 0 and args.patience >= 0):
        parser.error('Expected batch 1–32, positive epochs and nonnegative patience')
    catalog = read(CATALOG)
    version = next(row for row in catalog['versions'] if MODEL_IDS[args.width] in row['model_ids'])
    logs = ROOT / catalog['logs_root'] / args.run
    weights = ROOT / catalog['weights_root'] / version['key'] / args.run
    viewer = ROOT / catalog['viewer_root'] / args.run
    if args.resume:
        if not (weights/'last.pt').is_file() or read(logs/'status.json')['status'] == 'COMPLETED':
            parser.error('Resume requires an unfinished run and its matching last.pt')
    else:
        if any(path.exists() for path in (logs, weights, viewer)):
            parser.error('Run name already exists')
        logs.mkdir(parents=True)
        weights.mkdir(parents=True)
        write_json(logs/'status.json', dict(status='PREFLIGHT'))
    try:
        run(args, logs, weights, viewer, version)
    except Exception as error:
        if not args.resume:
            write_json(logs/'status.json', dict(status='FAILED', error=str(error)))
        raise
    index_versions()


if __name__ == '__main__':
    main()
