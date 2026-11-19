"""Audit sensor proportions and evaluate native and matched-footprint GSD."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
from pathlib import Path
from statistics import mean
import time

from pd_here_native_expanded_subset import allocate
from train_pd_here_s60_adaptive_query import (
    AdaptiveQueryDualViT, ExpandedTrainingData, FROZEN, IneligiblePairedScene, ROOT,
    TASKS, adaptive_masks, digest, native_task_mse, pack_adaptive_scene, read, torch, write_json,
)
from pd_here_s60.whole_scene_adaptive_query import restore_native_bands


def sensor(row):
    if row.get('sensor') in ('S2', 'L8', 'L9'):
        return {'S2': 'Sentinel-2 MSI', 'L8': 'Landsat 8 OLI', 'L9': 'Landsat 9 OLI-2'}[row['sensor']]
    instrument = row['instrument']
    if instrument.startswith(('S2', 'Sentinel-2')):
        return 'Sentinel-2 MSI'
    return {
        'LC09': 'Landsat 9 OLI-2', 'Landsat OLI platform unbound': 'Landsat OLI (unresolved)',
        'VENUS': 'VENuS', 'SPOT6': 'SPOT 6', 'SPOT7': 'SPOT 7',
        'SPOT6/7 unresolved': 'SPOT 6/7 (unresolved)', 'WorldView3': 'WorldView-3',
        'WORLDVIEW02': 'WorldView-2', 'QUICKBIRD02': 'QuickBird-2', 'GEOEYE01': 'GeoEye-1',
    }[instrument]


def stratify(rows, quotas):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['sensor']].append(row)
    return [row for key, size in sorted(quotas.items()) for row in
            sorted(buckets[key], key=lambda x: hashlib.sha256(f'17:{x["id"]}'.encode()).digest())[:size]]


def validation_panel(candidates, population, size):
    """Allocation based on actual TRAIN shares, bounded by eligible validation support."""
    available = Counter(r['sensor'] for r in candidates)
    desired = allocate(population, size)
    actual = {key: min(available[key], value) for key, value in desired.items()}
    while sum(actual.values()) < size:
        key = max((k for k in actual if actual[k] < available[k]),
                  key=lambda k: (size*population[k]/sum(population.values())-actual[k], k))
        actual[key] += 1
    return dict(desired=desired, actual=actual, candidates=available, members=stratify(candidates, actual))


def gsd_bounds(obs):
    return [round(v, 2) for v in (min(s for b in obs.bands for s in b.grid.spacing),
                                 max(s for b in obs.bands for s in b.grid.spacing))]


def measure_native(model, dataset, device, scales):
    results, rejected = [], []
    for role, rows in (('Core VAL', dataset.core_roles['VAL']), ('VAL_EXT', dataset.roles['VAL_EXT'])):
        for row in rows:
            key = row['id']
            if role == 'VAL_EXT' and set(row['eligible_tasks']) != {'A','B1'}:
                rejected.append(dict(id=key, sensor=sensor(row), reason='no paired data-side eligibility'))
                continue
            source = f'Core {row["sensor"]}' if role == 'Core VAL' else row['dataset']
            domain = 'surface_reflectance' if role == 'Core VAL' else row['radiometric_domain']
            scale = scales[domain] if row.get('value_encoding') == 'stored_dn' else 1.
            obs = dataset.core.load(row, 'cpu') if role == 'Core VAL' else dataset.load(row, 'cpu')
            try:
                scene = pack_adaptive_scene(obs, scale, source)
                views = adaptive_masks(scene, key, 0)
            except (IneligiblePairedScene, ValueError) as error:
                rejected.append(dict(id=key, sensor=sensor(row), reason=str(error)))
                continue
            scene = scene.to(device)
            losses = {task: float(native_task_mse(model, scene, tuple(x.to(device) for x in view), task))
                      for task, view in zip(TASKS, views, strict=True)}
            band = obs.bands[int((~views[1][1]).nonzero().item())]
            results.append(dict(id=key, sensor=sensor(row), instrument=row.get('instrument', row.get('sensor')),
                                source=source, domain=domain, role=role, native_gsd_m=gsd_bounds(obs),
                                b1_target_gsd_m=list(band.grid.spacing), patch_grid=scene.grid, mse=losses))
        print(f'{role}: {sum(x["role"]==role for x in results)} scored', flush=True)
    buckets = defaultdict(list)
    for row in results:
        buckets[(row['sensor'], row['source'], row['domain'])].append(row)
    summary = [dict(sensor=k[0], source=k[1], domain=k[2], parents=len(v),
                    native_gsd_m=[min(x['native_gsd_m'][0] for x in v), max(x['native_gsd_m'][1] for x in v)],
                    mse={t:mean(x['mse'][t] for x in v) for t in TASKS})
               for k,v in sorted(buckets.items())]
    return results, summary, rejected


def average(values, factor):
    return torch.nn.functional.avg_pool2d(values[None,None], factor)[0,0]


def native_view(obs, factor):
    bands = []
    for band in obs.bands:
        dx, _, x, _, dy, y = band.grid.affine
        values = average(band.values[:120,:120], factor)
        grid = replace(band.grid, shape=tuple(values.shape), affine=(dx*factor, 0., x, 0., dy*factor, y))
        bands.append(replace(band, values=values, valid=torch.ones_like(values, dtype=torch.bool), grid=grid))
    return replace(obs, bands=tuple(bands))


def matched_footprints(model, dataset, device):
    output = []
    for row in sorted(dataset.core_roles['VAL'], key=lambda r: r['id']):
        if row['sensor'] != 'S2':
            continue
        obs = dataset.core.load(row, 'cpu')
        obs = replace(obs, bands=tuple(b for b in obs.bands if b.key in ('B02','B03','B04','B08')))
        if len(obs.bands) != 4 or any(min(b.values.shape) < 120 or b.grid.spacing != (10.,10.) or
                                      not b.valid[:120,:120].all() for b in obs.bands):
            continue
        assert len({b.grid for b in obs.bands}) == 1
        scores = {}
        baseline = None
        for factor in (1,2,3):
            coarse = native_view(obs, factor)
            scene = pack_adaptive_scene(coarse, source='Core S2').to(device)
            assert scene.grid == (10,10)
            views = adaptive_masks(scene, row['id'], 0)
            if baseline is None:
                baseline = views
            for (keep,bands),(ref_keep,ref_bands) in zip(views,baseline,strict=True):
                assert torch.equal(keep,ref_keep) and torch.equal(bands,ref_bands)
            scores[10*factor] = {}
            for task,(keep,bands) in zip(TASKS, views, strict=True):
                predictions = restore_native_bands(coarse, scene, model(scene,keep,bands).float())
                selected = range(4) if task == 'A' else [int((~bands).nonzero().item())]
                region = (~keep.reshape(10,10)).repeat_interleave(2,0).repeat_interleave(2,1).cpu() if task=='A' else slice(None)
                scores[10*factor][task] = mean(float((average(predictions[i].cpu(),6//factor) -
                    average(obs.bands[i].values[:120,:120],6)).square()[region].mean()) for i in selected)
        output.append(dict(id=row['id'], common_60m_target_mse=scores))
        if len(output) == 12:
            break
    assert len(output) == 12
    return dict(parent_count=len(output), footprint_m=[1200,1200], patch_grid=[10,10],
                common_target_gsd_m=60, aggregate={v:{t:mean(x['common_60m_target_mse'][v][t]
                            for x in output) for t in TASKS} for v in (10,20,30)}, rows=output)


@torch.inference_mode()
def evaluate(args):
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    subset = read(FROZEN/'subset_expanded_2000_seed17.json')
    dataset = ExpandedTrainingData(ROOT).dataset
    identities = {r['id']:r for r in (*dataset.core_roles['TRAIN'],*dataset.roles['TRAIN'])}
    full = read(ROOT/'outputs/pd_here_s60/native512_spectral_conv256_full13020_mae_batch32_ep100_seed17/members.json')
    full = [dict(r, sensor=sensor(identities[r['id']])) for r in full]
    assert len(full) == 13020
    full_count = Counter(r['sensor'] for r in full)
    old_count = Counter(sensor(identities[r['id']]) for r in subset['parents'])
    candidates = stratify(full, allocate(full_count, 2000))
    assert len(candidates) == 2000 and len({r['id'] for r in candidates}) == 2000
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = AdaptiveQueryDualViT(checkpoint['identity']['fusion_width'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(args.device).eval()
    native, summary, rejected = measure_native(model, dataset, args.device, subset['dn_scales'])
    comparison = matched_footprints(model, dataset, args.device)
    train_ids = {r['id'] for r in full}
    assert not train_ids & {r['id'] for r in native}
    coverage = dict(full_train=full_count, historical_2000=old_count,
                    proposed_2000=Counter(r['sensor'] for r in candidates),
                    core_val=Counter(r['sensor'] for r in native if r['role']=='Core VAL'),
                    val_ext=Counter(r['sensor'] for r in native if r['role']=='VAL_EXT'))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'sensor_proportional_train2000.json', dict(counts=coverage['proposed_2000'],
               parents=[{k:r[k] for k in ('id','sensor','source','domain','encoding')} for r in candidates],
               dn_scales=subset['dn_scales']))
    write_json(args.out/'sensor_proportional_val275.json', validation_panel(native,full_count,275))
    write_json(args.out/'evaluation.json', dict(checkpoint=args.checkpoint.as_posix(),
               checkpoint_sha256=digest(args.checkpoint), checkpoint_epoch=checkpoint['epoch'],
               parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-start,
               optimizer_updates=0, coverage=coverage, report_by_sensor_and_product=summary,
               real_native_parents=native, exclusions=rejected, matched_footprints=comparison))
    print('matched same ground window:', comparison['aggregate'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    evaluate(parser.parse_args())
