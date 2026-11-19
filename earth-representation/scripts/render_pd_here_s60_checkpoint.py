"""Render fixed native validation predictions from a PD-HERE best checkpoint."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from index_pd_here_s60_versions import confirmed_version
from report_pd_here_s60_whole_scene import MODELS, ROOT, ExpandedTrainingData, np, plt, predict, read, torch, write_json
from pd_here_s60.whole_scene import pack_scene
from pd_here_s60.whole_scene_adaptive_query import AdaptiveQueryDualViT, MODEL_IDS, pack_adaptive_scene
from pd_here_s60.whole_scene_spectral_conv import pack_spectral_conv_scene
from train_pd_here_s60_adaptive_query import adaptive_masks
from train_pd_here_s60_whole_scene import masks


def texture_diagnostics(prediction, truth, hidden, patch, slot, native_slots):
    boundaries, interiors = [], []
    for dim in (0, 1):
        size = patch.shape[dim]-1
        if not size:
            continue
        support = hidden.narrow(dim, 0, size) & hidden.narrow(dim, 1, size)
        boundary = patch.diff(dim=dim) != 0
        error = (prediction.diff(dim=dim)-truth.diff(dim=dim)).square()
        boundaries.append(error[support & boundary])
        interiors.append(error[support & ~boundary])
    result = {}
    for name, parts in (('patch_boundary', boundaries), ('patch_interior', interiors)):
        if parts and sum(len(part) for part in parts):
            pairs = torch.cat(parts)
            result[name] = dict(pairs=len(pairs), gradient_rmse=float(pairs.mean().sqrt()))
    _, groups, counts = torch.unique(patch[hidden], return_inverse=True, return_counts=True)
    complete = counts == native_slots
    if int(complete.sum()) >= 2:
        selection = complete[groups]
        flattened = complete.long().cumsum(0)[groups[selection]]-1
        slots = slot[hidden][selection]
        template = {}
        for name, field in (('prediction',prediction), ('truth',truth)):
            values = field.new_empty(int(complete.sum()), native_slots)
            values[flattened, slots] = field[hidden][selection]
            texture = values-values.mean(1, keepdim=True)
            energy = texture.square().mean()
            template[name] = float(texture.mean(0).square().mean()/energy) if energy > 1e-20 else None
        result['complete_hidden_patches'] = int(complete.sum())
        result['common_template_fraction'] = template
    return result


@torch.inference_mode()
def render(run, device):
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    identity = read(run/'identity.json')
    adaptive = identity['model_id'] in MODEL_IDS.values()
    source = ROOT/identity['weights']/'best.pt' if adaptive else run/'best.pt'
    with TemporaryDirectory(dir=run, prefix='best-render-') as scratch:
        snapshot = Path(scratch)/'best.pt'
        os.link(source, snapshot)
        state = torch.load(snapshot, map_location='cpu', weights_only=True)
        with snapshot.open('rb') as handle:
            sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert state['identity']['model_id'] == identity['model_id']
    out = run/'best_predictions'/f"epoch_{state['epoch']:03d}"
    out.mkdir(parents=True, exist_ok=True)
    version = confirmed_version(identity['model_id'])
    if adaptive:
        model = AdaptiveQueryDualViT(identity['fusion_width'])
        views_for = adaptive_masks
    else:
        layouts = {tuple(map(tuple, entry['bands'])): entry['train_means'] for entry in identity['layouts']}
        model = MODELS[identity['model_id']](layouts)
        views_for = masks
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    dataset = ExpandedTrainingData(ROOT).dataset
    validation = dataset.core_roles['VAL']
    assert len(validation) == 275
    if not adaptive:
        file = run/'val_members.json'
        assert hashlib.sha256(file.read_bytes()).hexdigest() == identity['validation_members_sha256']
        manifest = {row['id']: row['cache_sha256'] for row in read(file)}
        assert all(manifest[row['id']] == row['cache_sha256'] for row in validation)
    rows = [row for sensor in ('S2','L8','L9')
            for row in sorted((v for v in validation if v['sensor'] == sensor), key=lambda v: v['id'])[:2]]
    stats = {task: [] for task in ('A','B1')}
    figures = {task: plt.subplots(len(rows), 4, figsize=(12, 2.25 * len(rows)), layout='constrained')
               for task in stats}
    gray = plt.get_cmap('gray').with_extremes(bad='#e9edf1')
    signed = plt.get_cmap('RdBu_r').with_extremes(bad='#e9edf1')
    for i, row in enumerate(rows):
        src = f'Core {row["sensor"]}'
        obs = dataset.core.load(row, 'cpu')
        if adaptive:
            scene = pack_adaptive_scene(obs, 1., src)
        elif 'wavelengths_sha256' in identity:
            scene = pack_spectral_conv_scene(obs, 1., src)
        else:
            scene = pack_scene(obs, 1., identity['patch_m'])
        scene = scene.to(device)
        for task, (keep, bands) in zip(stats, views_for(scene, row['id'], 0), strict=True):
            red = 'B04' if row['sensor'] == 'S2' else 'SR_B4'
            band_index = (next(j for j,b in enumerate(obs.bands) if b.key == red)
                          if task == 'A' else int((~bands).nonzero().item()))
            band = obs.bands[band_index]
            prediction, patch, slot, _, _ = predict(model, scene, keep, bands, obs.bands)[band_index]
            keep = keep.cpu()
            visible = (keep[patch] if keep.ndim == 1 else keep[patch, slot]) if task == 'A' else False
            truth, valid = band.values, band.valid
            hidden = valid & (~visible if task == 'A' else True)
            assert hidden.any() and prediction[hidden].isfinite().all()
            error = prediction - truth
            mse = float(error[hidden].square().mean())
            widths = [h*w for _,h,w in scene.layout]
            texture = texture_diagnostics(prediction, truth, hidden, patch, slot-sum(widths[:band_index]),
                                          widths[band_index])
            lo, hi = map(float, truth[valid].quantile(torch.tensor([.01, .99])))
            cap = max(1e-6, float(error[hidden].abs().quantile(.98)))
            fig, axes = figures[task]
            display = [(truth, valid & ~hidden, 'Visible target band'), (truth, valid, 'Native truth'),
                       (prediction, hidden, 'Prediction on hidden pixels'), (error, hidden, 'Signed error')]
            for j, (ax, (values, qa, title)) in enumerate(zip(axes[i], display, strict=True)):
                limits = (-cap, cap) if j == 3 else (lo, max(lo+1e-6, hi))
                ax.imshow(np.ma.array(values.numpy(), mask=~qa.numpy()), interpolation='nearest',
                          cmap=signed if j == 3 else gray, vmin=limits[0], vmax=limits[1])
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(title)
            axes[i,0].set_ylabel(f'{src}\n{band.key} · {row["id"][-8:]}')
            axes[i,2].set_xlabel(f'MSE {mse:.6g}')
            stats[task].append(dict(id=row['id'], source=src, band=band.key, hidden_pixels=int(hidden.sum()),
                                    valid_pixels=int(valid.sum()), mse=mse, signed_error_limits=[-cap,cap],
                                    texture=texture))
    for task, (fig, _) in figures.items():
        fig.suptitle(f'{task} | Core VAL, best epoch {state["epoch"]} | {identity["parents"]:,} TRAIN')
        fig.savefig(out/f'{task}_predictions.png', dpi=170)
        plt.close(fig)
    write_json(out/'prediction_examples.json', dict(
        model_name=version['name'], checkpoint=source.as_posix(), checkpoint_sha256=sha256, epoch=state['epoch'],
        updates=state['updates'], validation=state['validation'], validation_paired_mse=state['validation_paired_mse'],
        mask_exposure=0, displayed=stats))
    if adaptive:
        history = read(run/'history.json')
        epochs = [item['epoch'] for item in history]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
        for ax, task in zip(axes, stats, strict=True):
            ax.plot(epochs, [item['train'][task] for item in history], marker='.', label='Train loss')
            ax.plot(epochs, [item['validation'][task]['source_macro']['mse'] for item in history],
                    marker='.', label='Core VAL MSE')
            ax.axvline(state['epoch'], color='#667085', linestyle='--', linewidth=1, label='Best checkpoint')
            ax.set(xlabel='Epoch', ylabel='MSE', title=f'{task} reconstruction', xticks=epochs)
            ax.grid(alpha=.2)
            ax.legend()
        fig.savefig(run/'loss_curves.png', dpi=160)
        plt.close(fig)
    print(f'{out} | epoch {state["epoch"]} | VAL {state["validation_paired_mse"]:.8f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    render(args.run, args.device)
