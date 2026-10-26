"""Summarize the two frozen-USE full-decoder runs: per-event metrics, curves, provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/use_v2_full_decoders'
CONFIG = ROOT / 'configs/use_v2_full_decoders/execution.yaml'
KINDS = ('unet', 'segformer')
NAMES = {'unet': '冻结USE v2＋完整U-Net', 'segformer': '冻结USE v2＋SegFormer B0'}
NETWORKS = {'unet': 'UNetFeatureDecoder', 'segformer': 'SegFormerSeg'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def curve(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def rolling(values, window=50):
    """Per-step loss is dominated by 4-microbatch noise; show a trailing mean."""
    total, out = 0.0, []
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        out.append(total / min(index + 1, window))
    return out


def metrics(results):
    pixel = results['overall']['pixel']
    boundary = results['overall']['boundary']
    objects = results['overall']['objects']
    bins = {name: {'n_objects': value['n_objects'], 'object_recall': value['object_recall_evaluable'],
                   'pixel_recall': value['pixel_recall']} for name, value in objects['bins'].items()}
    per_event = {}
    for event, value in results['by_event'].items():
        per_event[event] = dict(iou=value['pixel']['iou'], dice=value['pixel']['dice'],
                                precision=value['pixel']['precision'], recall=value['pixel']['recall'],
                                ap=value.get('pixel_ap'), boundary_f1=value['boundary']['f1'],
                                object_recall=value['objects']['object_recall_evaluable'],
                                n_tiles=value['n_tiles'])
    return dict(event_mean_iou=results['event_mean_iou'], pixel_ap=results.get('pixel_ap'),
                iou=pixel['iou'], dice=pixel['dice'], precision=pixel['precision'], recall=pixel['recall'],
                boundary_f1=boundary['f1'], boundary_precision=boundary['precision'],
                boundary_recall=boundary['recall'], object_recall=objects['object_recall_evaluable'],
                object_recall_strict=objects['object_recall_strict_evaluable'],
                unmatched_prediction_objects_per_tile=objects['unmatched_prediction_objects_per_tile'],
                bins=bins, per_event=per_event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(BASE / 'comparison.json'))
    parser.add_argument('--figures', default=str(BASE / 'figures'))
    args = parser.parse_args()
    out, figures = Path(args.out), Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    frozen = (BASE / 'releases/engineering_v1/manifest.sha256').read_text(encoding='utf-8').strip()
    report = dict(release_sha256=frozen, groups={}, checks={})

    for kind in KINDS:
        run = BASE / 'runs' / kind
        summary = read(run / 'summary.json')
        rows = curve(run / 'curve.jsonl')
        if not summary['complete'] or summary['updates'] != 5000:
            raise ValueError(f'{kind}: run is not a complete 5000-update run')
        use_hashes = {row['frozen_sha256'] for row in rows}
        if len(use_hashes) != 1:
            raise ValueError(f'{kind}: frozen USE hash changed during training')
        gate = read(run / 'resolved_config.json')['device_gate']
        if not str(gate.get('device', '')).startswith('cuda'):
            raise ValueError(f'{kind}: formal run did not use a CUDA device')
        if summary['identity']['release'] != frozen:
            raise ValueError(f'{kind}: run was not executed under the frozen release')
        improved = [row for row in rows if row.get('selection_improved')]
        config_sha = sha(CONFIG)
        if summary['identity']['config_sha256'] != config_sha:
            raise ValueError(f'{kind}: run did not use the frozen configuration')
        report['groups'][kind] = dict(
            name=NAMES[kind], network=NETWORKS[kind],
            start=dict(frozen_sha256=rows[0]['frozen_sha256'], loss=rows[0]['loss'], step=rows[0]['step']),
            end=dict(loss=rows[-1]['loss'], step=rows[-1]['step']),
            best=dict(score=summary['best_score'], step=improved[-1]['step'] if improved else None,
                      selected=[row['step'] for row in improved]),
            use_frozen_sha256=sorted(use_hashes),
            device=gate['device'], physical_index=gate.get('physical_index'), uuid=gate.get('uuid'),
            display_active=gate.get('display_active'), peak_cuda_mib=round(summary['peak_cuda_bytes'] / 2 ** 20, 1),
            seconds=round(summary['seconds_this_session'], 1),
            seconds_per_update=round(summary['seconds_per_update'], 4),
            parameters=summary['parameters'], trainable_parameters=summary['trainable_parameters'],
            checkpoints=summary['checkpoints'],
            failure=read(run / 'failure.json') if (run / 'failure.json').exists() else None,
            metrics={name: metrics(read(run / 'eval' / name / 'evaluation.json')['results']) for name in ('best', 'last')})

    first, second = report['groups']['unet'], report['groups']['segformer']
    config_sha = sha(CONFIG)
    report['checks'] = dict(
        same_use=first['use_frozen_sha256'] == second['use_frozen_sha256'],
        both_5000=all(report['groups'][k]['end']['step'] == 5000 for k in KINDS),
        both_cuda=all(str(report['groups'][k]['device']).startswith('cuda') for k in KINDS),
        both_frozen_config=all(read(BASE / 'runs' / k / 'summary.json')['identity']['config_sha256'] == config_sha
                               for k in KINDS),
        no_failure=all(report['groups'][k]['failure'] is None for k in KINDS))
    report['checks']['use_never_changed'] = (
        first['use_frozen_sha256'] == [first['start']['frozen_sha256']] ==
        second['use_frozen_sha256'] == [second['start']['frozen_sha256']])
    report['config_sha256'] = config_sha

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    colors = {'unet': '#1f77b4', 'segformer': '#d62728'}
    for kind in KINDS:
        rows = curve(BASE / 'runs' / kind / 'curve.jsonl')
        steps = [row['step'] for row in rows]
        axes[0].plot(steps, rolling([row['loss'] for row in rows]), color=colors[kind], label=NAMES[kind], lw=1.4)
        scored = [(row['step'], row['selection_score']) for row in rows if row.get('selection_score') is not None]
        axes[1].plot([s for s, _ in scored], [v for _, v in scored], color=colors[kind], marker='o', ms=3,
                     label=f"{NAMES[kind]}（最佳 {report['groups'][kind]['best']['score']:.4f}）")
        best = report['groups'][kind]['best']
        if best['step']:
            axes[1].scatter([best['step']], [best['score']], s=60, facecolors='none', edgecolors=colors[kind], linewidths=1.5)
    axes[0].set_title('训练损失（每步，4次微批累积）', fontsize=11)
    axes[0].set_xlabel('更新次数'); axes[0].set_ylabel('BCE+Dice 损失'); axes[0].legend(fontsize=8)
    axes[1].set_title('开发验证：两事件等权IoU（每500步，圈出最佳）', fontsize=11)
    axes[1].set_xlabel('更新次数'); axes[1].set_ylabel('事件等权IoU'); axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    figure.suptitle('同一冻结USE v2与同一融合输入下的两个完整下游网络（seed1234，各5000更新）', fontsize=12)
    figure.savefig(figures / 'training_curves.png', dpi=140)
    plt.close(figure)

    report['provenance'] = {str(path.relative_to(ROOT)): sha(path)
                            for path in [
                                *(BASE / 'runs' / kind / name for kind in KINDS for name in
                                  ('summary.json', 'curve.jsonl', 'process.json', 'resolved_config.json')),
                                *(BASE / 'runs' / kind / 'eval' / name / 'evaluation.json'
                                  for kind in KINDS for name in ('best', 'last')),
                                BASE / 'releases/engineering_v1/manifest.json', CONFIG] if path.exists()}
    figure_path = figures / 'training_curves.png'
    report['provenance'][str(figure_path.relative_to(ROOT))] = sha(figure_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['checks'], ensure_ascii=False))
    for kind in KINDS:
        group = report['groups'][kind]
        for name in ('best', 'last'):
            value = group['metrics'][name]
            print(kind, name, 'event_mean_iou=%.4f' % value['event_mean_iou'],
                  'overall=%.4f' % value['iou'], 'ap=%.4f' % (value['pixel_ap'] or -1),
                  'boundary_f1=%.4f' % value['boundary_f1'],
                  'small_recall=%.4f' % value['bins'].get('small', {}).get('object_recall', -1))
    print('wrote', out, 'and', figure_path)


if __name__ == '__main__':
    main()
