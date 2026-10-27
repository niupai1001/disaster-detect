"""Six fixed validation cases with the frozen USE encoding: fused input, Fine head, Context, predictions."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

import run_use_v2 as rt
import run_use_v2_full_decoders as fd
from landslide_seg.models.use_v2 import canonical_context

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']


def pca_rgb(values, masks):
    """One basis for all six cases per panel; 2-98 percentile stretch so rows stay comparable."""
    flat = [value.reshape(value.shape[0], -1).T for value in values]
    stacked = torch.cat([item[mask] for item, mask in zip(flat, masks)])
    stacked = stacked - stacked.mean(0, keepdim=True)
    basis = torch.linalg.svd(stacked, full_matrices=False)[2][:3]
    projected = [(item @ basis.T).reshape(value.shape[1], value.shape[2], 3).numpy()
                 for item, value in zip(flat, values)]
    pool = np.concatenate([value.reshape(-1, 3)[mask.numpy()] for value, mask in zip(projected, masks)])
    low, high = np.percentile(pool, 2, axis=0), np.percentile(pool, 98, axis=0)
    return [np.clip((value - low) / np.maximum(high - low, 1e-6), 0, 1) for value in projected]


def collect(kind, run, device):
    """One read-only pass: model, fixed cases, RGB, Fine, Context and the fused input."""
    summary = rt.read(run / 'summary.json')
    checkpoint = run / 'best.pt'
    if rt.sha(checkpoint) != summary['checkpoints']['best.pt']:
        raise ValueError('Best checkpoint changed')
    config = rt.load_configuration()[0]
    model = fd.build(config['use_v2']['jobs'][kind], 'cpu')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    model.eval()
    boundary = model.use.verify()
    model.to(device)
    _, dataset = rt.datasets(config)
    stats = rt.load_stats(rt.ROOT / config['segformer']['data']['stats'])
    fixed = rt.read(rt.ROOT / 'docs/assets/research/use/case_provenance.json')['cases']
    lookup = {record['sample_id']: index for index, record in enumerate(dataset.records)}
    data = dict(kind=kind, run=run, summary=summary, checkpoint=checkpoint, model=model,
                boundary=boundary, stats=stats, fixed=fixed, rgb=[], fine=[], context=[],
                fused=[], truth=[], probability=[], valid=[])
    for case in fixed:
        sample = dataset[lookup[case['sample_id']]]
        with torch.no_grad():
            output = model.use(rt.collate([sample])['image'].to(device))
            probability = model(rt.collate([sample])['image'].to(device))[0, 0].sigmoid().cpu().numpy()
        fine = output['fine'][0].float().cpu()
        context = sum(F.interpolate(canonical_context(layer), size=fine.shape[-2:], mode='bilinear', align_corners=False)
                      for layer in output['context']) / len(output['context'])
        visible = sample['image'].x.numpy() * np.array(stats['std'])[:, None, None] + np.array(stats['mean'])[:, None, None]
        data['rgb'].append(np.clip(visible[[2, 1, 0]].transpose(1, 2, 0) * .0001 / .3, 0, 1))
        data['fine'].append(fine)
        data['context'].append(context[0].float().cpu())
        data['fused'].append((fine + context[0].float().cpu()))
        data['truth'].append(sample['mask'].numpy().astype(bool))
        data['probability'].append(probability)
        data['valid'].append(torch.from_numpy(sample['valid'].numpy().astype(bool).reshape(-1)))
    data['mask2d'] = [mask.numpy().reshape(data['truth'][0].shape) for mask in data['valid']]
    return data


def draw(data, columns, filename, title, dest):
    figure, axes = plt.subplots(len(data['fixed']), len(columns), figsize=(3 * len(columns), 3 * len(data['fixed'])),
                                layout='constrained')
    for index, case in enumerate(data['fixed']):
        for ax, (panel, label) in zip(axes[index], columns):
            data_array = panel[index]
            if data_array.ndim == 3:
                ax.imshow(data_array, interpolation='nearest')
            else:
                cmap = plt.get_cmap('magma' if data_array.dtype != bool else 'gray').copy()
                cmap.set_bad('#607d8b')
                ax.imshow(np.ma.masked_where(~data['mask2d'][index], data_array), cmap=cmap, vmin=0, vmax=1,
                          interpolation='nearest')
            ax.set_title(label if index == 0 else '', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        axes[index, 0].set_ylabel(case['sample_id'], fontsize=9)
        for column, (panel, _) in enumerate(columns):
            if panel is data['probability']:
                figure.colorbar(axes[index, column].images[0], ax=axes[index, column], shrink=.7)
    figure.suptitle(title, fontsize=12)
    figure.savefig(dest / filename, dpi=130)
    plt.close(figure)


def render(kind, run, device):
    rt.CONTRACT = fd.CONTRACT   # share the frozen contract with the training entry
    data = collect(kind, run, device)
    dest = run / 'cases'
    dest.mkdir(exist_ok=True)
    fine_rgb = pca_rgb(data['fine'], data['valid'])
    context_rgb = pca_rgb(data['context'], data['valid'])
    fused_rgb = pca_rgb(data['fused'], data['valid'])
    prediction = [value >= .5 for value in data['probability']]

    draw(data, [(data['rgb'], '原RGB（固定反射率0–0.3）'), (fused_rgb, 'USE融合编码（前3主成分）'),
                (data['truth'], '滑坡真值（白色）'), (data['probability'], '预测概率（固定0–1）'),
                (prediction, '预测掩膜（阈值0.5）')],
         'segmentation_cases_use.png',
         f"{type(data['model'].visual).__name__}：冻结USE v2＋全网络下游；第2列为两组共用的USE编码（六案例统一PCA基，仅用于可视化）\n"
         f"固定案例，非按预测挑选；{device}只读推理", dest)

    draw(data, [(data['rgb'], '原RGB（固定反射率0–0.3）'), (fine_rgb, 'USE细节头Fine（前3主成分）'),
                (context_rgb, 'USE上下文Context（四层压32上采样平均，前3主成分）'), (data['truth'], '滑坡真值（白色）'),
                (data['probability'], '预测概率（固定0–1）'), (prediction, '预测掩膜（阈值0.5）')],
         'segmentation_cases_fine.png',
         f"{type(data['model'].visual).__name__}：冻结USE v2的细节头Fine与上下文Context（同一编码器输出，PCAbasis各自独立）\n"
         f"固定案例，非按预测挑选；{device}只读推理", dest)

    if data['boundary'] != data['model'].use.verify():
        raise ValueError('USE changed during inference')
    rows = [dict(sample_id=case['sample_id'],
                 gt_pixels=int((truth & mask).sum()), predicted_pixels=int((prediction_row & mask).sum()),
                 true_positive=int((prediction_row & truth & mask).sum()),
                 max_probability=float(probability[mask].max()))
            for case, truth, prediction_row, probability, mask in zip(
                data['fixed'], data['truth'], prediction, data['probability'], data['mask2d'])]
    for case, fine, context in zip(data['fixed'], data['fine'], data['context']):
        np.savez_compressed(dest / (case['sample_id'] + '_encodings.npz'), fine=fine.numpy(),
                            context=context.numpy(), fused=(fine + context).numpy())
    for name in ('provenance_use.json', 'provenance_fine.json'):
        path = dest / name
        payload = dict(kind=kind, run=run.relative_to(rt.ROOT).as_posix(), checkpoint_sha256=rt.sha(data['checkpoint']),
                       use_sha256=data['boundary'],
                       panels=('rgb,fused,truth,probability,mask' if 'use' in name
                               else 'rgb,fine,context,truth,probability,mask'),
                       pca="per-panel 3 components from all six cases' valid pixels; 2-98 percentile stretch",
                       cases=rows, training_updates=data['summary']['updates'],
                       columns=5 if 'use' in name else 6)
        path.unlink(missing_ok=True)
        rt.write_json(path, payload, True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=('unet', 'segformer'))
    parser.add_argument('--run')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    run = Path(args.run).resolve() if args.run else (fd.OUT / 'runs' / args.kind).resolve()
    rows = render(args.kind, run, args.device)
    print(f'{args.kind}: {len(rows)} fixed cases -> {run / "cases"} (fused, fine, context panels)')


if __name__ == '__main__':
    main()
