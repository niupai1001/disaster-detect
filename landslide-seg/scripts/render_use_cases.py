"""Six fixed validation cases with one extra panel: the frozen USE encoding both groups share."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import run_use_v2 as rt
import run_use_v2_full_decoders as fd

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']


def render(kind, run, device):
    rt.CONTRACT = fd.CONTRACT   # share the frozen contract with the training entry
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

    samples, encodings = [], []
    for case in fixed:
        sample = dataset[lookup[case['sample_id']]]
        with torch.no_grad():
            encoding = model.fused(rt.collate([sample])['image'].to(device))[0].float().cpu()
        samples.append(sample)
        encodings.append(encoding)

    masks = [torch.from_numpy(sample['valid'].numpy().astype(bool).reshape(-1)) for sample in samples]
    flat = [encoding.reshape(32, -1).T for encoding in encodings]
    stacked = torch.cat([value[mask] for value, mask in zip(flat, masks)])
    stacked = stacked - stacked.mean(0, keepdim=True)
    _, _, basis = torch.linalg.svd(stacked, full_matrices=False)
    basis = basis[:3]
    projected = [(value @ basis.T).reshape(encoding.shape[1], encoding.shape[2], 3).numpy()
                 for value, encoding in zip(flat, encodings)]
    pool = np.concatenate([value.reshape(-1, 3)[mask.numpy()] for value, mask in zip(projected, masks)])
    low, high = np.percentile(pool, 2, axis=0), np.percentile(pool, 98, axis=0)

    dest = run / 'cases'
    dest.mkdir(exist_ok=True)
    figure, axes = plt.subplots(len(fixed), 5, figsize=(15, 3 * len(fixed)), layout='constrained')
    rows = []
    for index, (case, sample, encoding, rgb3, mask) in enumerate(zip(
            fixed, samples, encodings, projected, [m.numpy().reshape(128, 128) for m in masks])):
        with torch.no_grad():
            probability = model(rt.collate([sample])['image'].to(device))[0, 0].sigmoid().cpu().numpy()
        truth = sample['mask'].numpy().astype(bool)
        visible = sample['image'].x.numpy() * np.array(stats['std'])[:, None, None] + np.array(stats['mean'])[:, None, None]
        rgb = np.clip(visible[[2, 1, 0]].transpose(1, 2, 0) * .0001 / .3, 0, 1)
        encoding_rgb = np.clip((rgb3 - low) / np.maximum(high - low, 1e-6), 0, 1)
        prediction = probability >= .5
        panels = [(rgb, '原RGB（固定反射率0–0.3）'),
                  (encoding_rgb, 'USE编码（融合32通道前3主成分）'),
                  (truth, '滑坡真值（白色）'),
                  (probability, '预测概率（固定0–1）'),
                  (prediction, '预测掩膜（阈值0.5）')]
        for ax, (data, title) in zip(axes[index], panels):
            if data.ndim == 3:
                ax.imshow(data, interpolation='nearest')
            else:
                cmap = plt.get_cmap('magma' if data.dtype != bool else 'gray').copy()
                cmap.set_bad('#607d8b')
                ax.imshow(np.ma.masked_where(~mask, data), cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(title if index == 0 else '', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        axes[index, 0].set_ylabel(case['sample_id'], fontsize=9)
        figure.colorbar(axes[index, 3].images[0], ax=axes[index, 3], shrink=.7)
        rows.append(dict(sample_id=case['sample_id'], gt_pixels=int((truth & mask).sum()),
                         predicted_pixels=int((prediction & mask).sum()),
                         true_positive=int((prediction & truth & mask).sum()),
                         max_probability=float(probability[mask].max())))
        np.savez_compressed(dest / (case['sample_id'] + '_use.npz'), prob=probability, mask=truth, valid=mask,
                            rgb=rgb, encoding=encoding.numpy(), encoding_rgb=encoding_rgb)
    if boundary != model.use.verify():
        raise ValueError('USE changed during inference')
    figure.suptitle(f"{type(model.visual).__name__}：冻结USE v2＋全网络下游；第2列为两组共用的USE编码（六案例统一PCA基，仅用于可视化）\n"
                    f"固定案例，非按预测挑选；{device}只读推理", fontsize=12)
    figure.savefig(dest / 'segmentation_cases_use.png', dpi=130)
    plt.close(figure)
    rt.write_json(dest / 'provenance_use.json', dict(
        kind=kind, run=run.relative_to(rt.ROOT).as_posix(), checkpoint_sha256=rt.sha(checkpoint),
        use_sha256=boundary, pca="3 components from all six cases' valid fused pixels; 2-98 percentile stretch",
        cases=rows, training_updates=summary['updates'], columns=5), True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=('unet', 'segformer'))
    parser.add_argument('--run')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    run = Path(args.run).resolve() if args.run else (fd.OUT / 'runs' / args.kind).resolve()
    rows = render(args.kind, run, args.device)
    print(f'{args.kind}: wrote {run / "cases" / "segmentation_cases_use.png"} ({len(rows)} cases)')


if __name__ == '__main__':
    main()
