"""Measure native pixel and gradient recovery of the unmasked Autoencoder."""
import argparse
from collections import defaultdict
import re
from statistics import mean

from train_pd_here_s60_autoencoder import (
    MODELS, OUTPUTS, ROOT, SEED, TRAIN, ExpandedTrainingData,
    full_view, make_scene, pack_scene, read, torch, write_json,
)
from render_pd_here_s60_checkpoint import texture_diagnostics
from report_pd_here_s60_whole_scene import gradient_metrics, native_indices, np, plt, predict
from pd_here_s60.whole_scene_spectral_conv import band_center


def metrics(prediction, truth, qa, patch, slot, width):
    truth_valid = truth[qa].double()
    prediction_valid = prediction[qa].double()
    error = prediction_valid - truth_valid
    patch_ids = patch[qa]
    unique, inverse, counts = torch.unique(patch_ids, return_inverse=True, return_counts=True)
    patch_mean = truth_valid.new_zeros(len(unique)).scatter_add_(0, inverse, truth_valid)/counts
    patch_error = error.new_zeros(len(unique)).scatter_add_(0, inverse, error)/counts
    patch_constant = torch.zeros_like(truth)
    patch_constant[qa] = patch_mean[inverse].to(truth.dtype)
    result = dict(mse=float(error.square().mean()),
                  bias_sq=float(error.mean().square()),
                  centered_mse=float((error-error.mean()).square().mean()),
                  within_patch_centered_mse=float((error-patch_error[inverse]).square().mean()),
                  oracle_patch_mean_mse=float((truth_valid-patch_mean[inverse]).square().mean()))
    gradients = gradient_metrics(prediction, truth, qa)
    result.update(gradients)
    oracle = gradient_metrics(patch_constant, truth, qa)
    if oracle:
        result['oracle_patch_mean_gradient_mse'] = oracle['gradient_mse']
        grad = lambda u: torch.cat((u.diff(dim=0)[qa[:-1] & qa[1:]], u.diff(dim=1)[qa[:, :-1] & qa[:, 1:]]))
        target_rms = float(grad(truth).square().mean().sqrt())
        if target_rms >= 1e-6:
            result['gradient_rms_ratio'] = float(grad(prediction).square().mean().sqrt())/target_rms
    texture = texture_diagnostics(prediction, truth, qa, patch, slot, width)
    for key in ('patch_boundary', 'patch_interior'):
        if key in texture:
            result[key+'_gradient_mse'] = texture[key]['gradient_rmse']**2
    for key, val in texture.get('common_template_fraction', {}).items():
        if val is not None:
            result[key+'_template_fraction'] = val
    return result


def aggregate(rows):
    keys = set().union(*(r.keys() for r in rows))
    return {key: dict(mean=mean(vals), count=len(vals)) for key in sorted(keys)
            if (vals := [row[key] for row in rows if key in row])}


def plot(examples, epoch, path, architecture):
    gray = plt.get_cmap('gray').with_extremes(bad='#e9edf2')
    diverging = plt.get_cmap('RdBu_r').with_extremes(bad='#e9edf2')
    fig, ax = plt.subplots(len(examples), 3, figsize=(9.4, 2.38*len(examples)),
                           squeeze=False, layout='constrained')
    fig.suptitle(f'{architecture} | Unmasked Autoencoder | epoch {epoch}', fontsize=15)
    for i, (row, item, truth, pred, qa) in enumerate(examples):
        low, high = (float(x) for x in torch.quantile(truth[qa], torch.tensor([.01, .99])))
        high = max(high, low+1e-6)
        error = pred-truth
        lim = max(float(error[qa].abs().quantile(.98)), 1e-6)
        for j, (v,title) in enumerate(zip((truth, pred, error),
                                         ('Native input / target', 'Reconstruction', 'Prediction − target'),
                                         strict=True)):
            ax[i,j].imshow(np.ma.array(v.numpy(), mask=~qa.numpy()),
                           cmap=diverging if j==2 else gray, interpolation='nearest',
                           vmin=-lim if j==2 else low, vmax=lim if j==2 else high)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            if i == 0:
                ax[i,j].set_title(title)
        dx, dy = item['gsd_m']
        ax[i,0].set_ylabel(f'{row["sensor"]}\n{row["role"]} | {dx:.2f} × {dy:.2f} m\n'
                           f'{len(row["bands"])} bands | {item["key"]}', fontsize=8)
        row_metrics = item['metrics']
        ax[i,1].set_xlabel(f'MSE {row_metrics["mse"]:.3g} | '
                           f'grad corr {row_metrics.get("gradient_correlation", float("nan")):.3f}', fontsize=8)
        ax[i,2].set_xlabel(f'±{lim:.3g}   |   QA valid {float(qa.float().mean()):.1%}', fontsize=8)
    fig.savefig(path, dpi=145)
    plt.close(fig)


@torch.inference_mode()
def evaluate(out, weights, device, checkpoint):
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    identity = read(out/'identity.json')
    members = read(out/'members.json')
    targets = [row for row in members if row.get('scored')]
    missed = [row for row in members if row.get('scored') is False]
    assert len(targets) == identity['val_scored_parents']
    layouts = {tuple(map(tuple, row['bands'])): row['train_means'] for row in identity['layouts']}
    model = MODELS[identity['model_id']](layouts, decoder_width=identity['decoder_width']).to(device).eval()
    patch_m = identity['patch_m']
    architecture = (f'{patch_m} m Spatial ViT' if patch_m == 60 else
                    f'{patch_m} m Spectral / Conv, Decoder {identity["decoder_width"]}')
    state = torch.load(weights/f'{checkpoint}.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    assert state['identity']['members_sha256'] == identity['members_sha256']
    epoch = state['epoch']
    del state
    data = ExpandedTrainingData(ROOT)
    dataset = data.dataset
    index = {r['id']:r for rs in (dataset.core_roles['VAL'], dataset.roles['VAL_EXT']) for r in rs}
    scales = read(TRAIN)['dn_scales']

    def representative(rows):
        return max(rows, key=lambda row: (row['role'] == 'Core VAL',
                   min(b['valid_pixels']/np.prod(b['shape']) for b in row['bands']), row['id']))

    sensors = sorted({row['sensor'] for row in targets+missed})
    core = ['Sentinel-2 MSI', 'Landsat 8 OLI', 'Landsat 9 OLI-2']
    sensors = core + [name for name in sensors if name not in core]
    previews = {}
    for sensor in sensors:
        options = [r for r in targets if r['sensor'] == sensor]
        if not options:
            options = [r for r in members if r['role'] == 'TRAIN' and r['sensor'] == sensor]
            index.update({r['id']:data.extension[r['id']] for r in options})
        row = representative(options)
        previews[row['id']] = sensor
        if row['role'] == 'TRAIN':
            targets.append(row)
    output = out/'predictions'/f'{checkpoint}_epoch_{epoch:03d}'
    output.mkdir(parents=True, exist_ok=True)
    pictures, parent_metrics = {}, []
    for i, row in enumerate(targets, 1):
        key = row['id']
        scene, obs, scale = make_scene(dataset, index[key], row, scales, patch_m)
        keep, available = full_view(scene)
        preds = predict(model, scene.to(device), keep.to(device), available.to(device), obs.bands)
        # Compare within-block texture on the same physical partition as the 120 m runs.
        reference = scene if patch_m == 120 else pack_scene(obs, scale, 120)
        bands, offset, valid = [], 0, 0
        for i_band, ((pred, patch, slot, _, _), band, (_, h, w), meta) in enumerate(zip(
                preds, obs.bands, reference.layout, row['bands'], strict=True)):
            truth, qa = band.values/scale, band.valid
            if qa.any():
                if not pred[qa].isfinite().all():
                    raise FloatingPointError(f'Nonfinite native prediction: {key}/{band.key}')
                valid += int(qa.sum())
                if reference is not scene:
                    patch, slot, _, _ = native_indices(band, reference, offset)
                m = metrics(pred, truth, qa, patch, slot-offset, h*w)
                bands.append(dict(**meta, metrics=m))
                if key in previews:
                    center = (float(scene.metadata.wavelengths[i_band]) if scene.metadata is not None
                              else band_center(band, row['source'])[0])
                    rank = (abs(center-.655), band.key)
                    if key not in pictures or rank < pictures[key][0]:
                        pictures[key] = rank, (row, bands[-1], truth, pred, qa)
            offset += h*w
        assert valid == row['valid_pixels']
        normal = {k: v['mean'] for k, v in aggregate([b['metrics'] for b in bands]).items()}
        parent_metrics.append(dict(id=key, role=row['role'], sensor=row['sensor'], source=row['source'],
                                   domain=row['domain'], patch_grid=row['patch_grid'], valid_pixels=valid,
                                   bands=bands, metrics=normal))
        if i % 50 == 0:
            print(f'native evaluation {i}/{len(targets)}', flush=True)
    rows = [r for r in parent_metrics if r['role'] != 'TRAIN']
    summary = aggregate([r['metrics'] for r in rows])
    by_sensor = {name: dict(parents=len(group), metrics=aggregate([r['metrics'] for r in group]))
                 for name in sensors if (group := [r for r in rows if r['sensor'] == name])}
    selected = [pictures[key][1] for key in previews]
    plot(selected, epoch, output/'overview.png', architecture)
    plot([picture for picture in selected if picture[0]['sensor'] in core],
         epoch, output/'core_overview.png', architecture)
    for picture in selected:
        filename = re.sub(r'[^a-z0-9]+', '_', picture[0]['sensor'].lower()).strip('_')
        plot([picture], epoch, output/f'{filename}.png', architecture)
    write_json(output/'native_metrics.json', dict(epoch=epoch, checkpoint=checkpoint, scored_val=len(rows),
               unsupported_val=missed, reduction='valid native pixels then active bands then parents',
               diagnostic_patch_m=120,
               patch_mean='oracle: uses a separate true input mean in every native band and 120m patch',
               summary=summary, by_sensor=by_sensor, parents=parent_metrics, fixed_examples=list(previews)))
    history = read(out/'history.json')
    initial = read(out/'initial_validation.json')['parent_mse']
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot([r['epoch'] for r in history], [r['train_mse'] for r in history], 'o-', label='TRAIN')
    ax.plot([0]+[r['epoch'] for r in history], [initial]+[r['val']['parent_mse'] for r in history],
            'o-', label='VAL, parent mean')
    ax.set(xlabel='Epoch', ylabel='Native MSE', yscale='log', title='Unmasked Autoencoder')
    ax.legend()
    fig.savefig(out/'loss_curves.png', dpi=145)
    plt.close(fig)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--checkpoint', choices=('best','last'), default='last')
    args = parser.parse_args()
    out = OUTPUTS/args.run
    evaluate(out, ROOT/read(out/'identity.json')['weights'], args.device, args.checkpoint)
