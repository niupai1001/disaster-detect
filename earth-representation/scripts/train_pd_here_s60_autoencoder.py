"""Train whole-scene models to reconstruct all QA-valid native pixels."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
from statistics import mean, median
import time

from pd_here_native_expanded_subset import CORE_SHA, EXTENSION_SHA, ExpandedTrainingData, digest, read
import torch

from index_pd_here_s60_versions import confirmed_version, viewer_labels
from render_pd_here_s60_sensor_validation import load_view
from train_pd_here_s60_whole_scene import MODELS, ROOT, SPATIAL_ID, VIEWER, RunWriter, save, write_json
from pd_here_s60.whole_scene import pack_scene, tile_bounds
from pd_here_s60.whole_scene_batch import packed_forward, scene_signature
from pd_here_s60.whole_scene_runtime import band_slots, prepared_backward
from pd_here_s60.whole_scene_spectral_conv import MODEL_ID, pack_spectral_conv_scene


OUTPUTS = ROOT / 'outputs/pd_here_s60'
SENSOR_AUDIT = OUTPUTS / 'sensor_gsd_validation_seed17'
SPLIT = SENSOR_AUDIT / 'coverage_qc_revision'
TRAIN = SPLIT / 'screened_train2000_candidate.json'
VAL = SPLIT / 'screened_val275_candidate.json'
SEED = 17


def make_scene(dataset, member, row, scales, patch_m=120):
    scale = scales[row['domain']] if member.get('value_encoding') == 'stored_dn' else 1.
    observation = (dataset.core.load(member, 'cpu') if row['source'].startswith('Core ')
                   else load_view(dataset, member, row))
    scene = (pack_scene(observation, scale, patch_m) if patch_m == 60
             else pack_spectral_conv_scene(observation, scale, row['source']))
    if not scene.valid_pixels:
        raise ValueError(f"No valid input or target pixels in {row['id']}")
    return scene, observation, scale


def load_data(out, patch_m):
    data = ExpandedTrainingData(ROOT)
    frozen = read(TRAIN)
    train = [dict(row, role='TRAIN') for row in frozen['parents']]
    validation = read(VAL)['members']
    train_ids = {r['id'] for r in train}
    val_ids = {r['id'] for r in validation}
    assert len(train_ids) == len(train) == 2000 and len(val_ids) == len(validation) == 275
    assert not train_ids & val_ids
    sources = dict(data.core, **data.extension)
    heldout = {r['id']: r for rows in (data.dataset.core_roles['VAL'], data.dataset.roles['VAL_EXT']) for r in rows}
    cache, manifest, output_means = {}, {}, defaultdict(list)
    for i, row in enumerate(train + validation, 1):
        key = row['id']
        member = (sources if row['role'] == 'TRAIN' else heldout)[key]
        scene, obs, _ = make_scene(data.dataset, member, row, frozen['dn_scales'], patch_m)
        cache[key] = scene
        bounds = tile_bounds(scene)
        assert all((r1-r0)*h <= 512 and (c1-c0)*w <= 512
                   for r0, r1, c0, c1 in bounds for _, h, w in scene.layout)
        manifest[key] = dict(**row, layout=scene.layout, patch_grid=scene.grid, tiles=bounds,
                             valid_pixels=scene.valid_pixels,
                             bands=[dict(key=b.key, gsd_m=b.grid.spacing, shape=b.grid.shape,
                                         valid_pixels=int(b.valid.sum())) for b in obs.bands])
        if row['role'] == 'TRAIN':
            widths, _ = band_slots(scene.layout)
            output_means[scene.layout].append([
                float(x.sum()/q.sum()) if q.any() else None
                for x, q in zip(scene.values.split(widths, -1), scene.qa.split(widths, -1), strict=True)])
        if i % 250 == 0:
            print(f'packed {i}/{len(train)+len(validation)}', flush=True)
    layouts = {key: [mean(ok) if (ok := [x for x in band if x is not None]) else 0.
                     for band in zip(*means, strict=True)] for key, means in output_means.items()}
    for key in val_ids:
        manifest[key]['scored'] = cache[key].layout in layouts
        if not manifest[key]['scored']:
            manifest[key]['reason'] = 'No trained output head for native band layout'
            del cache[key]
    write_json(out/'members.json', list(manifest.values()))
    scored = [row for row in validation if manifest[row['id']]['scored']]
    print(f'{len(train)} TRAIN | {len(scored)}/{len(validation)} scoreable VAL | {len(layouts)} heads', flush=True)
    return train, scored, cache, layouts


def full_view(scene):
    return torch.ones(len(scene.qa), dtype=torch.bool), torch.ones(len(scene.layout), dtype=torch.bool)


def initialize_model(layouts, decoder_width, model_id):
    torch.manual_seed(SEED)
    baseline = MODELS[model_id](layouts)
    if decoder_width == baseline.decoder_width:
        return baseline
    torch.manual_seed(SEED)
    wider = MODELS[model_id](layouts, decoder_width=decoder_width)
    initial = wider.state_dict()
    shared = {name: value for name, value in baseline.state_dict().items()
              if value.shape == initial[name].shape}
    wider.load_state_dict(initial | shared)
    return wider


def batch_losses(model, rows, cpu, gpu, views, signatures, *, backward=False):
    outputs, buckets = {}, defaultdict(list)
    scale = 1 / len(rows) if backward else None
    for row in rows:
        key = row['id']
        if signatures[key] is None:
            outputs[key], = prepared_backward(model, cpu[key], gpu[key], (views[key],), scale,
                                               reconstruct_all=True)
        else:
            buckets[signatures[key]].append(key)
    for keys in buckets.values():
        jobs = [(None, cpu[key], gpu[key], *views[key]) for key in keys]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses, _, _ = packed_forward(model, jobs)
        if backward:
            (losses.sum() * scale).backward()
        outputs.update(zip(keys, losses.detach().unbind(), strict=True))
    return torch.stack([outputs[row['id']] for row in rows])


def benchmark(model, train, cpu, gpu, views, signatures, device):
    grouped = defaultdict(list)
    for row in train:
        signature = signatures[row['id']]
        if signature is not None:
            grouped[signature].append(row)
    heavy = max(train, key=lambda row: len(tile_bounds(cpu[row['id']])))
    candidates = sorted(grouped.values(), key=len, reverse=True)
    sample = [heavy] + [row for group in candidates[:4] for row in group[:8] if row != heavy]
    # Use real tail-batch sizes and include the largest tile path.
    sample = sample[:16]
    unbatched = dict.fromkeys(signatures)
    # Compare meaningful encoder gradients as the production output weights start at zero.
    with torch.no_grad():
        for head in model.projections.values():
            head.output.weight.normal_(std=.01)
    encoder_weight = (model.fusion.weight if hasattr(model, 'fusion')
                      else model.spatial_block.attention.in_proj_weight)
    times, result, gradients, peaks = {}, {}, {}, {}
    for name, mode in (('resident', unbatched), ('packed', signatures)):
        times[name] = []
        torch.cuda.reset_peak_memory_stats(device)
        for i in range(4):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            values = batch_losses(model, sample, cpu, gpu, views, mode, backward=True)
            torch.cuda.synchronize(device)
            if i:
                times[name].append(time.perf_counter()-start)
        peaks[name] = torch.cuda.max_memory_allocated(device)
        result[name] = values
        gradients[name] = encoder_weight.grad.detach().float().clone()
    torch.testing.assert_close(result['packed'], result['resident'], atol=2e-5, rtol=.01)
    g1, g2 = gradients.values()
    cosine = float(torch.nn.functional.cosine_similarity(g1.flatten(), g2.flatten(), dim=0))
    relative_gradient_diff = float((g1-g2).norm()/g1.norm())
    assert cosine > .995 and relative_gradient_diff < .08, (cosine, relative_gradient_diff)
    # Confirm the support and normalized loss directly against the native packed values.
    key = next(row['id'] for row in sample if signatures[row['id']] is not None)
    scene = gpu[key]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        losses, output, lengths = packed_forward(model, [(None, cpu[key], scene, *views[key])])
        structural = scene.qa.any(-1)
        qa = scene.qa[structural]
        residual = output[0, :lengths[0]].float() - scene.values[structural]
        widths, _ = band_slots(scene.layout)
        native_loss = torch.stack([e[q].square().mean()
                                  for e, q in zip(residual.split(widths,-1), qa.split(widths,-1), strict=True)
                                  if q.any()]).mean()
    assert int(qa.sum()) == scene.valid_pixels
    torch.testing.assert_close(losses[0], native_loss)
    medians = {k: median(v) for k, v in times.items()}
    return dict(seconds_per_step=medians, peak_allocated_bytes=peaks,
                benchmark_parents=[r['id'] for r in sample], checked_native_pixels=scene.valid_pixels,
                gradient_cosine=cosine, relative_gradient_difference=relative_gradient_diff,
                largest_tile_count=len(tile_bounds(cpu[heavy['id']])), selected=min(medians, key=medians.get))


@torch.no_grad()
def evaluate(model, rows, cpu, gpu, views, signatures, batch):
    model.eval()
    results = defaultdict(list)
    for start in range(0, len(rows), batch):
        group = rows[start:start+batch]
        scores = batch_losses(model, group, cpu, gpu, views, signatures).tolist()
        for row, mse in zip(group, scores, strict=True):
            results[row['sensor']].append(mse)
    model.train()
    sensors = {name: dict(mse=mean(values), parents=len(values)) for name, values in results.items()}
    return dict(parent_mse=mean(mse for values in results.values() for mse in values),
                sensor_macro_mse=mean(v['mse'] for v in sensors.values()), sensors=sensors)


def run(args, out, weights, viewer, model_id, version):
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    patch_m = 60 if model_id == SPATIAL_ID else 120
    train, validation, cpu, layouts = load_data(out, patch_m)
    gpu = {key: scene.to(args.device) for key, scene in cpu.items()}
    views = {key: full_view(scene) for key, scene in cpu.items()}
    signatures = {key: scene_signature(scene) for key, scene in cpu.items()}
    if not args.resume:
        probe = initialize_model(layouts, args.decoder_width, model_id).to(args.device)
        preflight = benchmark(probe, train, cpu, gpu, views, signatures, args.device)
        del probe
        write_json(out/'preflight.json', preflight)
    else:
        preflight = read(out/'preflight.json')
    print(f'preflight {preflight["seconds_per_step"]} selected {preflight["selected"]}', flush=True)
    if preflight['selected'] == 'resident':
        signatures = dict.fromkeys(signatures)
    model = initialize_model(layouts, args.decoder_width, model_id).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.04, fused=True)
    updates_per_epoch = math.ceil(len(train)/args.batch)
    identity = dict(task='AE', mask_fraction=0., model_id=model_id, model_name=version['name'],
                    encoder_width=128 if model_id == SPATIAL_ID else 256,
                    decoder_width=args.decoder_width, patch_m=patch_m,
                    native_output_max_hw=[512, 512], parameters=sum(p.numel() for p in model.parameters()),
                    train_parents=len(train), val_scored_parents=len(validation), val_manifest_parents=275,
                    train_sensor_counts=Counter(r['sensor'] for r in train),
                    val_scored_sensor_counts=Counter(r['sensor'] for r in validation),
                    train_manifest_sha256=digest(TRAIN), val_manifest_sha256=digest(VAL),
                    members_sha256=digest(out/'members.json'), core_sha256=CORE_SHA, extension_sha256=EXTENSION_SHA,
                    layouts=[dict(bands=key, train_means=value) for key, value in layouts.items()],
                    epochs=args.epochs, batch=args.batch, updates_per_epoch=updates_per_epoch,
                    lr=1e-4, weight_decay=.04, optimizer='AdamW fused', seed=SEED,
                    initialization='seed17 scratch; TRAIN band-mean output bias' if args.decoder_width == 128 else
                    'seed17 scratch; 128-wide run initialization for same-shape tensors; TRAIN band-mean bias',
                    precision='BF16 autocast, FP32 loss and weights', device=args.device,
                    objective='all QA-valid input pixels; native pixels then active bands then parents',
                    weights=weights.relative_to(ROOT).as_posix())
    if model_id != SPATIAL_ID:
        identity['fusion_width'] = 256
    history = []
    if args.resume:
        checkpoint = torch.load(weights/'last.pt', map_location='cpu', weights_only=True)
        if read(out/'identity.json')['members_sha256'] != identity['members_sha256'] or any(
                checkpoint['identity'][k] != identity[k] for k in ('model_id', 'epochs', 'batch', 'decoder_width', 'train_manifest_sha256',
                                                                  'val_manifest_sha256', 'layouts')):
            raise ValueError('Resume configuration differs from checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['cpu_rng'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng'], args.device)
        history = checkpoint['history']
    else:
        write_json(out/'identity.json', identity)
        initial = evaluate(model, validation, cpu, gpu, views, signatures, args.batch)
        write_json(out/'initial_validation.json', initial)
        print(f'initial VAL: {initial["parent_mse"]:.6g}', flush=True)
    experiment, variant = viewer_labels(version, len(train))
    variant['id'] += '-autoencoder'
    if model_id == SPATIAL_ID:
        variant['name'] += '（无遮盖 Autoencoder）'
    elif args.decoder_width != 128:
        variant['id'] += f'-decoder-{args.decoder_width}'
        variant['name'] += f'（无遮盖 Autoencoder，Decoder {args.decoder_width} 维）'
    variant['kind'] = 'modification'
    variant['description'] = 'Autoencoder, 0% mask, all valid native pixels'
    metrics = {k: dict(label=label, direction='min', role='loss', unit='')
               for k, label in (('train_mse', 'AE TRAIN MSE'), ('val_mse', 'AE VAL MSE'),
                                ('val_sensor_macro_mse', 'AE VAL sensor-macro MSE'))}
    best = min((item['val']['parent_mse'] for item in history), default=math.inf)
    with RunWriter(viewer, experiment=experiment, variant=variant, config=identity,
                   metrics=metrics, charts=list(metrics), resume=args.resume) as log:
        for epoch in range(len(history)+1, args.epochs+1):
            began = time.perf_counter()
            order = train.copy()
            random.Random(SEED+epoch).shuffle(order)
            total = 0.
            for step, offset in enumerate(range(0, len(order), args.batch), 1):
                group = order[offset:offset+args.batch]
                optimizer.zero_grad(set_to_none=True)
                losses = batch_losses(model, group, cpu, gpu, views, signatures, backward=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                total += float(losses.sum())
                if step % 20 == 0:
                    print(f'epoch {epoch}/{args.epochs} step {step}/{updates_per_epoch} '
                          f'seconds {time.perf_counter()-began:.1f}', flush=True)
            train_seconds = time.perf_counter()-began
            val = evaluate(model, validation, cpu, gpu, views, signatures, args.batch)
            update = epoch*updates_per_epoch
            item = dict(epoch=epoch, updates=update, train_mse=total/len(order), val=val,
                        train_seconds=train_seconds, seconds=time.perf_counter()-began)
            history.append(item)
            if val['parent_mse'] < best:
                best = val['parent_mse']
                path = weights/'best.tmp'
                torch.save(dict(model=model.state_dict(), epoch=epoch, identity=identity), path)
                path.replace(weights/'best.pt')
            save(weights/'last.pt', model, optimizer, epoch, identity, history, args.device)
            write_json(out/'history.json', history)
            log.log(update, dict(train_mse=item['train_mse'], val_mse=val['parent_mse'],
                                 val_sensor_macro_mse=val['sensor_macro_mse']), epoch=epoch)
            print(f'epoch {epoch}: TRAIN {item["train_mse"]:.6g} VAL {val["parent_mse"]:.6g} '
                  f'time {item["seconds"]:.1f}s', flush=True)
    best_epoch = min(history, key=lambda item: item['val']['parent_mse'])['epoch']
    write_json(out/'status.json', dict(status='COMPLETED', epochs=len(history), best_epoch=best_epoch,
                                      updates=history[-1]['updates'], stop_reason='epoch_budget',
                                      wall_seconds=time.perf_counter()-start,
                                      peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--spatial60', action='store_true', help='Use the 60 m single spatial ViT')
    parser.add_argument('--decoder-width', type=int, choices=(128, 512), default=128)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.spatial60 and args.decoder_width != 128:
        parser.error('The 60 m experiment uses the original 128-wide decoder')
    model_id = SPATIAL_ID if args.spatial60 else MODEL_ID
    version = confirmed_version(model_id)
    output = OUTPUTS/args.run
    weights = ROOT/'weights/pd_here_s60'/version['key']/args.run
    viewer = VIEWER/args.run
    if not args.resume and any(p.exists() and any(p.iterdir()) for p in (output, weights, viewer)):
        parser.error('Select an unused run name')
    output.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    run(args, output, weights, viewer, model_id, version)
