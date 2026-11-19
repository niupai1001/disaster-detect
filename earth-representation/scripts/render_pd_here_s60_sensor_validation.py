"""Render native A/B1 reconstructions from the sensor/GSD validation report."""
import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from evaluate_pd_here_s60_sensor_gsd import FROZEN, ROOT, digest
from render_pd_here_s60_checkpoint import (
    MODELS, AdaptiveQueryDualViT, ExpandedTrainingData, adaptive_masks, masks, np, pack_adaptive_scene,
    pack_spectral_conv_scene, plt, predict, read, texture_diagnostics, torch, write_json,
)
from pd_here_s60.whole_scene import native_geometry
from pd_here_s60.whole_scene_spectral_conv import band_center
from pd_here_spacenet_qc import MIN_SUPPORT, edge_zero_padding


SENSORS = {
    'Sentinel-2 MSI': 'sentinel2',
    'Landsat 8 OLI': 'landsat8',
    'Landsat 9 OLI-2': 'landsat9',
    'SPOT 6/7 (unresolved)': 'spot6_or_7',
    'VENuS': 'venus',
    'SPOT 6': 'spot6',
    'SPOT 7': 'spot7',
    'WorldView-2': 'worldview2',
    'WorldView-3': 'worldview3',
}


CORE_SENSORS = tuple(SENSORS)[:3]


def load_view(dataset, row, result):
    if result['role'] == 'Core VAL':
        return dataset.core.load(row, 'cpu')
    obs = dataset.load(row, 'cpu')
    if result['source'] == 'SpaceNet 3' and len(obs.bands) == 8:
        fill = torch.from_numpy(edge_zero_padding([band.values.numpy() for band in obs.bands]))
        bands = tuple(replace(band, valid=band.valid & ~fill) for band in obs.bands)
        assert float(torch.stack([band.valid for band in bands]).all(0).float().mean()) >= MIN_SUPPORT
        obs = replace(obs, bands=bands)
    return obs


def observations(model, dataset, row, result, scales, device, *,
                 pack=pack_adaptive_scene, views_for=adaptive_masks, tasks=('A','B1')):
    source = result['source']
    scale = scales[result['domain']] if row.get('value_encoding') == 'stored_dn' else 1.
    obs = load_view(dataset, row, result)
    scene = pack(obs, scale, source).to(device)
    wavelengths = [band_center(b, source)[0] for b in obs.bands]
    red = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i]-.655))
    output = {}
    for task, (keep, bands) in zip(('A', 'B1'), views_for(scene, row['id'], 0), strict=True):
        if task not in tasks:
            continue
        predictions = predict(model, scene, keep, bands, obs.bands)
        target = red if task == 'A' else int((~bands).nonzero().item())
        band = obs.bands[target]
        pred, patch, slot, _, _ = predictions[target]
        keep = keep.cpu()
        visible = keep[patch] if keep.ndim == 1 else keep[patch, slot]
        hidden = band.valid & (~visible if task == 'A' else True)
        assert hidden.any() and pred[hidden].isfinite().all()
        truth = band.values/scale
        error = pred-truth
        band_losses = []
        for j, (predicted, ids, slots, _, _) in enumerate(predictions):
            other = obs.bands[j]
            if task == 'A':
                support = (other.valid & ~(keep[ids] if keep.ndim == 1 else keep[ids,slots]))
            elif bands[j]:
                continue
            else:
                support = other.valid
            if support.any():
                band_losses.append(float((predicted[support]-other.values[support]/scale).square().mean()))
        if task == 'A':
            context, observed, context_name = truth, band.valid & ~hidden, band.key
        else:
            visible_bands = [i for i in bands.nonzero().flatten().tolist()
                             if obs.bands[i].grid == band.grid and obs.bands[i].valid.any()]
            index = min(visible_bands, key=lambda i: abs(wavelengths[target]-wavelengths[i]))
            other = obs.bands[index]
            assert other.grid == band.grid
            context, observed, context_name = other.values/scale, other.valid, other.key
        dx, dy = band.grid.spacing
        height, width = truth.shape
        widths = [h*w for _,h,w in scene.layout]
        native = texture_diagnostics(pred, truth, hidden, patch, slot-sum(widths[:target]), widths[target])
        metrics = dict(id=row['id'], sensor=result['sensor'], source=source, domain=result['domain'], role=result['role'],
                       band=band.key, input_band=context_name, gsd_m=[dx,dy], ground_extent_m=[width*dx,height*dy],
                       band_count=len(obs.bands), all_bands=[b.key for b in obs.bands],
                       patch_grid=scene.grid, valid_pixels=int(band.valid.sum()), hidden_pixels=int(hidden.sum()),
                       target_band_mse=float(error[hidden].square().mean()),
                       parent_mse=sum(band_losses)/len(band_losses),
                       diagnostics=native)
        output[task] = (metrics, [(context, observed), (truth, band.valid), (pred, hidden), (error, hidden)])
    return output


def limits(data, qa):
    lo, hi = map(float, data[qa].quantile(torch.tensor([.01,.99])))
    return lo, max(lo+1e-6, hi)


def draw(rows, title, path, show_task=False):
    gray = plt.get_cmap('gray').with_extremes(bad='#eff1f4')
    residual = plt.get_cmap('RdBu_r').with_extremes(bad='#eff1f4')
    fig, axes = plt.subplots(len(rows), 4, squeeze=False, figsize=(12, 2.55*len(rows)), layout='constrained')
    fig.suptitle(title, fontsize=15)
    labels = ('Visible input', 'Native truth', 'Prediction on hidden pixels', 'Signed error')
    for i, (task, (m, fields)) in enumerate(rows):
        dx, dy = m['gsd_m']
        width, height = m['ground_extent_m']
        truth_scale = limits(*fields[1])
        err_scale = max(1e-6, float(fields[3][0][fields[3][1]].abs().quantile(.98)))
        for j, (ax, (pixels, qa)) in enumerate(zip(axes[i], fields, strict=True)):
            lo, hi = ((-err_scale, err_scale) if j==3 else
                      (limits(pixels,qa) if task=='B1' and j==0 else truth_scale))
            ax.imshow(np.ma.array(pixels.numpy(), mask=~qa.numpy()), extent=(0,width,height,0),
                      interpolation='nearest', cmap=residual if j==3 else gray, vmin=lo, vmax=hi)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(labels[j])
        name = m['sensor'].replace(' (unresolved)', '\n(platform unresolved)')
        note = f'{name}\n{dx:.2f} × {dy:.2f} m pixels\n{width:.0f} × {height:.0f} m view'
        axes[i,0].set_ylabel((f'{task}\n' if show_task else '')+note, fontsize=8.5)
        axes[i,0].set_xlabel(m['input_band'], fontsize=8)
        axes[i,1].set_xlabel(m['band'], fontsize=8)
        axes[i,2].set_xlabel(f'MSE {m["target_band_mse"]:.4g}', fontsize=8)
        axes[i,3].set_xlabel(f'±{err_scale:.3g}', fontsize=8)
        length = max(v for v in (20,50,100,200,400) if v <= min(width,height)*.45)
        bar = AnchoredSizeBar(axes[i,1].transData, length, f'{length} m', 'lower right',
                              pad=.3, color='white', frameon=True, size_vertical=0)
        bar.patch.set_facecolor('#20262e')
        bar.patch.set_edgecolor('#20262e')
        axes[i,1].add_artist(bar)
    fig.savefig(path, dpi=155, facecolor='white')
    plt.close(fig)


def panel_candidates(dataset, report, evaluation):
    rows = {r['id']:r for r in (*dataset.core_roles['VAL'],*dataset.roles['VAL_EXT'])}
    audit = {r['id']:r['effective_support'] for r in read(evaluation.parent/'spacenet_padding_audit.json')}
    by_sensor = defaultdict(list)
    for result in report['real_native_parents']:
        name = result['sensor']
        if name in SENSORS and (name not in CORE_SENSORS or result['role'] == 'Core VAL'):
            by_sensor[name].append(result)
    return rows, by_sensor, audit


@torch.inference_mode()
def render(evaluation, device):
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = read(evaluation)
    checkpoint = ROOT/report['checkpoint']
    assert digest(checkpoint) == report['checkpoint_sha256']
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['epoch'] == report['checkpoint_epoch']
    model = AdaptiveQueryDualViT(state['identity']['fusion_width'])
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    dataset = ExpandedTrainingData(ROOT).dataset
    heldout, available, audit = panel_candidates(dataset, report, evaluation)
    scales = read(FROZEN/'subset_expanded_2000_seed17.json')['dn_scales']
    out = evaluation.parent/'predictions_with_core'
    out.mkdir(exist_ok=True)
    combined = {task:[] for task in ('A','B1')}
    index = {}
    for sensor, filename in SENSORS.items():
        selected = sorted((x for x in available[sensor] if audit.get(x['id'],1.) >= MIN_SUPPORT),
                          key=lambda row:row['id'])[:2]
        assert len(selected) == 2
        panels = [observations(model, dataset, heldout[item['id']], item, scales, device) for item in selected]
        for task, rows in combined.items():
            rows.append((task, panels[0][task]))
        rows = [(task, panel[task]) for panel in panels for task in combined]
        draw(rows, f'{sensor} | {selected[0]["role"]} | best epoch {state["epoch"]}',
             out/f'{filename}.png', show_task=True)
        index[sensor] = {task:[panel[task][0] for panel in panels] for task in combined}
        print(sensor, [item['id'] for item in selected], flush=True)
    for task, rows in combined.items():
        draw(rows, f'{task} | Core VAL + VAL_EXT | best epoch {state["epoch"]}', out/f'{task}_overview.png')
    write_json(out/'prediction_examples.json', dict(checkpoint=report['checkpoint'], epoch=state['epoch'],
               checkpoint_sha256=report['checkpoint_sha256'], mask_exposure=0,
               pixels_resampled_for_display=False, selection='first two eligible sorted IDs per sensor',
               parameters=report['parameters'], optimizer_updates=0, sensors=index))


def layout_for(row, patch_m):
    return native_geometry([(b['key'], tuple(b['shape']), tuple(b['affine']), b['crs']) for b in row['bands']],
                           patch_m)[2]


@torch.inference_mode()
def render_full(evaluation, run, device):
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = read(evaluation)
    path = run/'best.pt'
    status = read(run/'status.json')
    state = torch.load(path, map_location='cpu', weights_only=True)
    identity = state['identity']
    assert status['status'] == 'COMPLETED' and state['epoch'] == status['early_stop']['best_epoch']
    assert identity['parents'] == 13020 and identity['model_id'] == 'S120-SpectralTransformer-SpatialConv'
    layouts = {tuple(map(tuple, item['bands'])): item['train_means'] for item in identity['layouts']}
    model = MODELS[identity['model_id']](layouts)
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    data = ExpandedTrainingData(ROOT)
    dataset = data.dataset
    heldout, available, audit = panel_candidates(dataset, report, evaluation)
    scales = read(FROZEN/'subset_expanded_2000_seed17.json')['dn_scales']
    out = run/'best_predictions'/f"epoch_{state['epoch']:03d}"/'sensor_predictions_with_core'
    out.mkdir(parents=True, exist_ok=True)
    combined = {task:[] for task in ('A','B1')}
    comparison = []
    index, coverage = {}, {}
    core_manifest = run/'val_members.json'
    assert digest(core_manifest) == identity['validation_members_sha256']
    core_layouts = {row['id']: tuple(map(tuple,row['layout'])) for row in read(core_manifest)}
    for sensor, filename in SENSORS.items():
        ordered = sorted(available[sensor], key=lambda item:item['id'])
        eligible = [x for x in ordered if audit.get(x['id'],1.) >= MIN_SUPPORT and
                    (core_layouts[x['id']] if x['role'] == 'Core VAL' else
                     layout_for(heldout[x['id']], identity['patch_m'])) in layouts]
        selected = eligible[:2]
        coverage[sensor] = dict(role=ordered[0]['role'], paired_val=len(ordered), compatible=len(eligible),
                                query_examples=[x['id'] for x in ordered[:2]],
                                selected=[x['id'] for x in selected])
        if not selected:
            coverage[sensor]['first_missing_layout'] = layout_for(heldout[ordered[0]['id']], identity['patch_m'])
            continue
        panels = [observations(model, dataset, heldout[item['id']], item, scales, device,
                               pack=pack_spectral_conv_scene, views_for=masks) for item in selected]
        for task, rows in combined.items():
            rows.append((task, panels[0][task]))
        rows = [(task,panel[task]) for panel in panels for task in combined]
        draw(rows, f'{sensor} | Full-data model | {selected[0]["role"]} | best epoch {state["epoch"]}',
             out/f'{filename}.png', show_task=True)
        index[sensor] = {task:[panel[task][0] for panel in panels] for task in combined}
        item = selected[0]
        matched = observations(model, dataset, heldout[item['id']], item, scales, device,
                               pack=pack_spectral_conv_scene, views_for=adaptive_masks, tasks=('A',))
        comparison.append(('A', matched['A']))

    train_preview = []
    if not coverage['WorldView-3']['compatible']:
        candidates = sorted((x for x in data.extension.values()
                             if 'WorldView3' == x['instrument'] and audit.get(x['id'],0.) >= MIN_SUPPORT and
                             layout_for(x,identity['patch_m']) in layouts), key=lambda x:x['id'])[:2]
        for row in candidates:
            item = dict(sensor='WorldView-3', source=row['dataset'], domain=row['radiometric_domain'], role='TRAIN')
            panel = observations(model, dataset, row, item, scales, device,
                                 pack=pack_spectral_conv_scene, views_for=masks)
            train_preview.extend((task,panel[task]) for task in combined)
        if train_preview:
            draw(train_preview, f'WorldView-3 | TRAIN examples | best epoch {state["epoch"]}',
                 out/'worldview3_train.png', show_task=True)
    for task, rows in combined.items():
        draw(rows, f'{task} | Full-data model | Core VAL + VAL_EXT | best epoch {state["epoch"]}', out/f'{task}_overview.png')
    draw(comparison, f'A | Full-data weights with 25% masking | Core VAL + VAL_EXT | epoch {state["epoch"]}',
         out/'A_matched25_overview.png')
    write_json(out/'prediction_examples.json', dict(checkpoint=path.as_posix(), epoch=state['epoch'],
               checkpoint_sha256=digest(path), parameters=sum(p.numel() for p in model.parameters()),
               mask_exposure=0, standard_a_mask='historical approximately 75% hidden spatial groups',
               matched_a_mask='approximately 25% hidden spatial groups',
               val_ext_coverage=coverage, sensors=index, matched25_A=[panel[0] for _,panel in comparison],
               train_preview=[dict(task=task, **panel[0]) for task,panel in train_preview]))
    print(coverage, flush=True)
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--full-run', type=Path)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.full_run is None:
        render(args.evaluation, args.device)
    else:
        render_full(args.evaluation, args.full_run, args.device)
