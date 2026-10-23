"""从指定已完成运行读取训练曲线；不改权重，不把回看预算当预注册验证。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets/research'
NAMES = {
    'Single': '单期恒等', 'Mean15': '15期均值', 'Content15': '15期内容',
    'Date15': '15期内容与日期', 'History-Date': '截止参考日期',
    'Date3': '3期内容与日期', 'Date5': '5期内容与日期', 'Date7': '7期内容与日期',
}


def main():
    paths = [ROOT / 'outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234/train_summary.json']
    for folder in ('v4r1_temporal_control', 'v4r1_temporal_frame_count', 's2_10band_v4r1_geo_raw'):
        paths += sorted((ROOT / 'outputs/segformer' / folder).glob('SegFormer*/1234/train_summary.json'))
    result = {'scope': '成熟细节基线、时间控制与固定期数、修正坡度后的六组地形；只纳入已完成3600步训练摘要',
              'metric': '训练时每100步计算的两个验证事件等权IoU；与最终重评小数有数值差异',
              'boundary': '回顾性诊断，同seed与反复使用的两个验证事件；不能保证新结构的最优预算',
              'segformer': [], 'utae': []}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout='constrained')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    # 字体必须在创建文本前设置。
    for path in paths:
        d = json.loads(path.read_text(encoding='utf-8'))
        if not d.get('complete') or d.get('global_step') != 3600:
            continue
        rows = d['history']
        score = lambda r: r.get('dev_event_mean_iou', r['val_iou'])
        best = max(rows, key=score)
        early = max((r for r in rows if r['global_step'] <= 2000), key=score)
        assert best['global_step'] == d['best_step']
        assert abs(score(best) - d['best_val_iou']) < 1e-9
        row = {'run': path.parent.parent.name, 'source': path.relative_to(ROOT).as_posix(),
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'best_step': best['global_step'], 'best_iou': score(best),
               'best_through_2000': score(early), 'last_iou': score(rows[-1]),
               'loss_at_best': best['train_loss'], 'loss_at_last': rows[-1]['train_loss'],
               'seconds': d['seconds']}
        result['segformer'].append(row)
        suffix = row['run'].removeprefix('SegFormer-B0-Temporal-')
        if suffix in NAMES:
            xs = [r['global_step'] for r in rows]
            line, = axes[0].plot(xs, [score(r) for r in rows], label=NAMES[suffix], linewidth=1.5)
            axes[0].scatter(best['global_step'], score(best), color=line.get_color(), s=24)
            axes[1].plot(xs, [r['train_loss'] for r in rows], color=line.get_color(), linewidth=1.5)
    for path in sorted((ROOT / 'outputs/utae_v4r1/v4r1').glob('*/1234/train_summary.json')):
        d = json.loads(path.read_text(encoding='utf-8'))
        if not d.get('complete'):
            continue
        updates = sum(r['updates'] for r in d['history'] if r['epoch'] <= d['best_epoch'])
        result['utae'].append({'mode': path.parent.parent.name, 'source': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'best_epoch': d['best_epoch'],
            'actual_updates_at_best': updates, 'batch': d['meta']['batch_size'],
            'best_iou': d['best_inner_selection'], 'epochs': d['epochs']})
    result['n_completed_segformer'] = len(result['segformer'])
    result['n_best_within_2000'] = sum(r['best_step'] <= 2000 for r in result['segformer'])
    result['nominal_update_saving'] = 1 - 2000 / 3600
    for ax, title, ylabel in zip(axes, ['验证表现：圆点为全程最佳', '训练误差：后段仍在下降'], ['事件平均IoU', '训练损失']):
        ax.axvline(2000, color='black', linestyle='--', linewidth=1)
        ax.set(xlabel='成功参数更新次数', ylabel=ylabel, title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle('SegFormer时间实验：2000步预算的回顾性依据（seed 1234）')
    fig.savefig(OUT / 'training_budget.png', dpi=160)
    plt.close(fig)
    (OUT / 'training_budget.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('n_completed_segformer', 'n_best_within_2000', 'utae')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
