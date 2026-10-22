"""Audit completed runs against their own source snapshots and compare fixed DEV cases."""
from __future__ import annotations
import hashlib
import itertools
import json
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from summarize_frozen_decoder_probes import load, write_csv, rgb, error_mask, plt, np, file_sha256, label_objects, object_table, match_objects

OUT = ROOT / 'reports/v4r3_results'
RUNS = [
    ('segformer_standard', ROOT / 'outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_n100_u3600_s1234/1234', None),
    ('segformer_detail', ROOT / 'outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234', None),
    ('segformer_detail_low_encoder_lr', ROOT / 'outputs/segformer/s2_10band_v4r3_low_encoder_lr/segformer_b0_direct10_detailres_pretrained_encoder_low_lr_n100_u3600_s1234/1234', ROOT / 'outputs/segformer/v4r3_low_encoder_lr_batch'),
    ('dofa_upernet_pyramid', ROOT / 'outputs/segformer/s2_10band_v4r3_dofa/dofa_base_upernet_pyramid_binary_n100_u3600_s1234/1234', ROOT / 'outputs/segformer/v4r3_dofa_batch'),
]


def main():
    OUT.mkdir(exist_ok=True)
    rows, full, curves, audits = [], [], [], []
    protocol = load(ROOT / 'manifests/v4r1/protocol.json')
    expected_stats = file_sha256(ROOT / 'manifests/v4r1/stats_core25.json')
    expected_manifest = file_sha256(ROOT / 'manifests/v4r1/pool_100.jsonl')
    for label, run, batch in RUNS:
        if batch:
            assert load(batch / 'status.json')['state'] == 'complete'
        summary, meta = load(run / 'train_summary.json'), load(run / 'run_meta.json')
        cfg = load(run / 'resolved_config.json')['segformer']
        assert summary['complete'] and summary['global_step'] == 3600
        assert summary['samples_exposed'] == 230400 and summary['skipped_batches'] == 0
        assert summary['n_train'] == 2223 and summary['dev_events'] == {'hiroshima': 448, 'thrissur': 221}
        assert [h['global_step'] for h in summary['history']] == list(range(100, 3601, 100))
        assert meta['stats_sha256'] == expected_stats and meta['protocol']['sha256'] == protocol['sha256']
        assert expected_manifest.startswith(meta['manifest']['sha256'])
        best = max(summary['history'], key=lambda h: h['val_iou'])
        assert best['global_step'] == summary['best_step'] and best['val_iou'] == summary['best_val_iou']
        exposures = load(run / 'sample_exposures.json')
        assert len(exposures) == 2223 and sum(exposures.values()) == 230400
        assert set(exposures.values()) == {103, 104}
        source_verified = False
        if batch:
            with zipfile.ZipFile(batch / 'training_sources.zip') as archive:
                for relative, digest in meta['source_sha256'].items():
                    assert hashlib.sha256(archive.read(relative)).hexdigest() == digest, relative
            source_verified = True
        hashes = {name: file_sha256(run / f'{name}.pt') for name in ('best', 'last')}
        evaluations = {tag: load(run / 'eval' / f'eval_{tag}.json') for tag in ('best_val', 'last_val', 'best_train', 'last_train')}
        for tag, e in evaluations.items():
            checkpoint, split = tag.split('_')
            assert e['checkpoint_sha256'] == hashes[checkpoint]
            assert e['stats_sha256'] == expected_stats and e['manifest_sha256'] == expected_manifest
            assert e['role'] == ('TRAIN' if split == 'train' else 'DEV')
            assert not {'italy', 'newzealand'} & set(e['by_event'])
        for checkpoint in ('best', 'last'):
            e = evaluations[checkpoint + '_val']
            assert set(e['by_event']) == {'hiroshima', 'thrissur'}
            p, o = e['overall']['pixel'], e['overall']['objects']
            mean = float(np.mean([v['pixel']['iou'] for v in e['by_event'].values()]))
            assert abs(e['event_mean_iou'] - mean) < 1e-12
            assert abs(p['iou'] - p['tp'] / (p['tp'] + p['fp'] + p['fn'])) < 1e-12
            assert abs(p['dice'] - 2*p['iou']/(1+p['iou'])) < 1e-12
            assert o['n_gt_evaluable'] == 1679
            assert e['overall']['negative_tiles']['tile_false_alarm_rate'] is None
            row = dict(condition=label, checkpoint=checkpoint,
                       step=summary['best_step'] if checkpoint == 'best' else 3600,
                       train_iou=evaluations[checkpoint + '_train']['overall']['pixel']['iou'],
                       dev_mean_iou=mean, pooled_iou=p['iou'], dice=p['dice'], precision=p['precision'], recall=p['recall'],
                       ap=p['average_precision'], background_pixel_fpr=p['false_positive_rate'],
                       hiroshima_iou=e['by_event']['hiroshima']['pixel']['iou'], thrissur_iou=e['by_event']['thrissur']['pixel']['iou'],
                       object_recall=o['object_recall_evaluable'], object_hits=o['n_matched_evaluable'],
                       boundary_f1=e['overall']['boundary']['f1'], params=summary['params']['total'],
                       trainable_params=summary['params']['trainable'], peak_alloc_mib=summary['peak_alloc_mib'],
                       training_seconds=summary['seconds'], latency_b1_ms=evaluations['best_val']['efficiency']['latency_b1']['mean_ms'],
                       training_precision='bf16_forward_fp32_loss' if cfg['train'].get('amp') else 'fp32', evaluation_precision='fp32',
                       checkpoint_sha256=hashes[checkpoint], negative_image_fpr=None, final_evaluated=False)
            for name, values in o['fine_bins'].items():
                row[f'matched_{name}'] = values['n_matched_evaluable']
                row[f'objects_{name}'] = values['n_objects_evaluable']
                row[f'recall_{name}'] = values['object_recall_evaluable']
            rows.append(row)
        curves.extend(dict(condition=label, **h) for h in summary['history'])
        full.append(dict(condition=label, evaluations=evaluations))
        audits.append(dict(condition=label, run_dir=str(run), updates=3600, exposures=230400, dev_checks=36,
                           source_snapshot_verified=source_verified, historical_source_audited_in_v4r1=batch is None,
                           source_sha256=meta['source_sha256'], checkpoint_sha256=hashes,
                           params=summary['params'], manifest_sha256=expected_manifest, stats_sha256=expected_stats))
    write_csv(OUT / 'comparison.csv', rows)
    for name, payload in [('full_evaluations.json', full), ('curves.json', curves), ('completion_audit.json',
            dict(status='PASS', n_new_formal_runs=2, n_historical_references=2, seed=1234,
                 protocol_sha256=protocol['sha256'], final_evaluated=False, runs=audits))]:
        (OUT / name).write_text(json.dumps(payload, indent=1), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for label, _, _ in RUNS:
        h = [r for r in curves if r['condition'] == label]
        for ax, values in zip(axes, ([r['train_loss'] for r in h], [r['train_clean']['pooled']['iou'] for r in h], [r['val_iou'] for r in h])):
            ax.plot([r['global_step'] for r in h], values, label=label)
    for ax, title in zip(axes, ('TRAIN loss', 'Clean TRAIN pooled IoU', 'DEV event mean IoU')):
        ax.set(title=title, xlabel='Optimizer updates')
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'training_curves.png', dpi=160)
    plt.close(fig)
    selected_runs = RUNS[1:]
    arrays = {}
    for label, run, _ in selected_runs:
        with np.load(run / 'eval/predictions_best_val.npz') as data:
            arrays[label] = {k: data[k].copy() for k in data.files}
    ref = arrays['segformer_detail']
    for data in arrays.values():
        for key in ('sample_ids', 'gt', 'valid'):
            assert np.array_equal(data[key], ref[key])
    records = {r['sample_id']: r for r in [json.loads(line) for line in (ROOT / 'manifests/v4r1/pool_100.jsonl').read_text(encoding='utf-8').splitlines()]}
    indices = {str(sid): i for i, sid in enumerate(ref['sample_ids'])}
    paired = []
    for sid, i in indices.items():
        gt, valid = ref['gt'][i], ref['valid'][i]
        labels, n = label_objects(gt)
        objects = [o for o in object_table(labels, n, valid) if not o['truncated']]
        hits = {}
        for label, data in arrays.items():
            pred, n_pred = label_objects((data['probability'][i] >= .5) & valid)
            hits[label] = match_objects(labels, objects, pred, object_table(pred, n_pred, None), .25)
        for o in objects:
            area = o['area']
            size = '1_4' if area < 5 else '5_9' if area < 10 else '10_19' if area < 20 else '20_49' if area < 50 else '50_199' if area < 200 else 'ge200'
            paired.append(dict(sample_id=sid, event=records[sid]['event_id'], gt_object_id=o['id'], area_px=area, size_bin=size,
                               **{label + '_hit': int(o['id'] in matched) for label, matched in hits.items()}))
    for label in arrays:
        e = next(e for e in full if e['condition'] == label)['evaluations']['best_val']['overall']['objects']
        assert len(paired) == e['n_gt_evaluable']
        assert sum(o[label + '_hit'] for o in paired) == e['n_matched_evaluable']
    write_csv(OUT / 'paired_dev_objects.csv', paired)
    overlap = []
    for left, right in itertools.combinations(arrays, 2):
        for size in ('all', '1_4', '5_9', '10_19', '20_49', '50_199', 'ge200'):
            subset = [o for o in paired if size == 'all' or o['size_bin'] == size]
            overlap.append(dict(left=left, right=right, size_bin=size, n_objects=len(subset),
                                both_hit=sum(o[left+'_hit'] and o[right+'_hit'] for o in subset),
                                left_only=sum(o[left+'_hit'] and not o[right+'_hit'] for o in subset),
                                right_only=sum(not o[left+'_hit'] and o[right+'_hit'] for o in subset),
                                both_missed=sum(not o[left+'_hit'] and not o[right+'_hit'] for o in subset)))
    write_csv(OUT / 'paired_object_overlap.csv', overlap)
    cases = [c for c in load(ROOT / 'reports/v4r1_results/error_atlas/selection.json')
             if c['experiment'] == 'segformer_b0_direct10_detailres_n100_u3600_s1234' and c['available']]
    case_rows = []
    fig, axes = plt.subplots(len(cases), 7, figsize=(21, 3*len(cases)), squeeze=False)
    for i, c in enumerate(cases):
        sid = c['sample_id']
        idx, record = indices[sid], records[sid]
        gt, valid = ref['gt'][idx], ref['valid'][idx]
        with np.load(record['cache']) as data:
            color = rgb(data['x10'])
        panels = [(color, f"{c['event']} {c['category']}\n{sid} {record['date']}"), (gt, 'GT (visibility unreviewed)')]
        for label in arrays:
            probability = arrays[label]['probability'][idx]
            pred = probability >= .5
            tp, fp, fn = int((pred & gt & valid).sum()), int((pred & ~gt & valid).sum()), int((~pred & gt & valid).sum())
            case_rows.append(dict(sample_id=sid, event=c['event'], category=c['category'], condition=label,
                                  tp=tp, fp=fp, fn=fn, iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                                  selection='fixed v4r1 detail model DEV cases', visibility_reviewed=False))
            if label != 'segformer_detail':
                panels.append((probability, label + ' probability'))
            panels.append((error_mask(probability, gt, valid), label + '\nredFP greenTP blueFN'))
        for ax, (arr, title) in zip(axes[i], panels):
            ax.imshow(arr, **({} if arr.ndim == 3 else dict(cmap='gray', vmin=0, vmax=1)))
            ax.set_title(title, fontsize=7)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / 'fixed_dev_cases.png', dpi=110)
    plt.close(fig)
    write_csv(OUT / 'fixed_dev_cases.csv', case_rows)
    (OUT / 'fixed_case_selection.json').write_text(json.dumps(cases, indent=1), encoding='utf-8')
    lines = ['# v4r3：低编码器学习率与 DOFA 完整系统对照', '',
             '执行与整理跨2026-10-19至2026-10-20；文件名沿用本阶段开始日期。', '',
             '本轮已完成两组真实正式训练，以及接口、梯度、重载与回归验证。历史两组 SegFormer 只读作参照，没有重新训练。单 seed1234；TRAIN2223、DEV669（Hiroshima448、Thrissur221）；各3600次成功更新、batch64、230400次曝光、每100步诊断并按双事件平均 IoU 选 best，同时保存 last。不评 Italy/New Zealand。', '',
             '## 实际结果', '',
             'DEV均值IoU为两事件算术均值；Dice/Precision/Recall/AP为合并有效像素指标，不能与事件均值混算。阈值固定0.5；对象匹配沿用既定v2规则。', '',
             '| 系统 | checkpoint/步数 | TRAIN IoU | DEV均值IoU | Hiroshima | Thrissur | Dice | Precision | Recall | AP | 边界F1 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['checkpoint']}/{r['step']} | {r['train_iou']:.4f} | {r['dev_mean_iou']:.4f} | {r['hiroshima_iou']:.4f} | {r['thrissur_iou']:.4f} | {r['dice']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['ap']:.4f} | {r['boundary_f1']:.4f} |")
    lines += ['', '## 可评对象匹配数（best）', '', '排除边界截断对象；下表为命中数/可评对象数，不是像素召回。总计1679个，其中1–4px共77个。', '',
              '| 系统 | 1–4px | 5–9px | 10–19px | 20–49px | 50–199px | ≥200px | 对象召回 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['checkpoint'] == 'best':
            bins = ' | '.join(f"{r['matched_'+b]}/{r['objects_'+b]}" for b in ('1_4', '5_9', '10_19', '20_49', '50_199', 'ge200'))
            lines.append(f"| {r['condition']} | {bins} | {r['object_recall']:.4f} |")
    lines += ['', '## 成本（本机实测）', '',
              '| 系统 | 总参数 | 可训练参数 | 训练秒数 | 峰值allocated MiB | FP32 batch1 ms | 训练精度 |', '|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if r['checkpoint'] == 'best':
            lines.append(f"| {r['condition']} | {r['params']} | {r['trainable_params']} | {r['training_seconds']:.1f} | {r['peak_alloc_mib']:.1f} | {r['latency_b1_ms']:.2f} | {r['training_precision']} |")
    lines += ['', '训练时间包括每100步完整TRAIN/DEV推理和checkpoint保存，不能当纯反传耗时；显存为PyTorch allocated峰值。推理速度为同机实测但受运行环境影响；所有最终与周期评测均FP32。', '',
              '## 本轮修改与复用', '',
              '- 新增 `models/dofa.py` 薄封装，在 `models/factory.py` 增加可选DOFA类型；原SegFormer与训练/损失/评测核心未重写。旧TerraMind、AnySat、语义及粗细分割接口保留。',
              '- DOFA编码器复用官方commit 0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9的密集接口及位置插值；neck复用MMSegmentation1.2.2 Feature2Pyramid；head复用Transformers4.57.6 UperNetHead。单二值头，无辅助头。',
              '- 预训练encoder无缺失键，五个未使用的预训练专用键有明确记录；官方权重SHA256与远端身份一致。波长取自TerraTorch1.2.13映射；输入影像不放大，patch16的原生特征为8×8。',
              '- 两份独立YAML、训练/验证入口、来源与权重准备脚本、汇总脚本和复现说明均已保存；可选依赖仅安装在vendor/dofa_runtime。',
              '- 低LR组保持首层6e-5、其他预训练encoder改6e-6、其余6e-4；全参数训练。DOFA encoder6e-5、neck/head6e-4，除官方固定位置编码外全部训练。', '',
              '## 验证和解释边界', '',
              '- 5项优化器测试及21项适配器/模型/精确更新回归测试通过；低LR真实3步训练、输出和重载检查通过。',
              '- DOFA真实十波段前向、encoder/neck/head有限非零梯度、8样本100步过拟合（IoU约0.787）、保存重载误差0与batch64反传通过。该过拟合结果仅验证链路，不代表泛化。',
              '- 两正式组各36次DEV检查、零跳过更新、曝光与数据/统计/协议身份一致；best/last评测对应checkpoint哈希；源码分别与各组训练快照核对，不拿后加DOFA的factory否定先完成的低LR组。',
              '- DOFA同时改变了编码器、预训练、decoder、容量与训练精度，是完整系统对照；不能把差异单独归因遥感预训练、波长建模或参数量。',
              '- 同一DEV被多次用于选模和研究诊断，未获得独立FINAL泛化结论；单seed不能声称显著性。可靠无滑坡影像误报率仍N/A，背景像素FPR另存CSV，不混用。',
              '- 固定案例沿用v4r1 detailres的8个案例ID，三系统的GT/valid逐像素一致；对象按相同GT身份配对。影像可见性与标签时相仍未逐例人工确认，不据预测声称标签错误。', '',
              '## 产物与复现', '',
              '`reports/v4r3_results/`包含comparison.csv、完整评测JSON、训练曲线、固定案例图与CSV、逐对象配对、完成核验；`docs/v4r3_reproduction.md`包含配置/组件/环境/精度/命令与源码快照使用说明。两正式batch目录保存原日志、status和training_sources.zip。', '',
              f'![训练曲线]({(OUT / "training_curves.png").as_posix()})', '',
              f'![固定DEV案例]({(OUT / "fixed_dev_cases.png").as_posix()})']
    interpretation = OUT / 'interpretation.md'
    if interpretation.exists():
        lines += ['', interpretation.read_text(encoding='utf-8')]
    report = ROOT / 'reports/v4r3_execution_report_2026-10-19.md'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('PASS: 2 new formal runs, 2 historical references; own source snapshots and common evaluation identities verified')
    for row in rows:
        print(row['condition'], row['checkpoint'], 'DEV', row['dev_mean_iou'], 'AP', row['ap'], 'tiny', row['matched_1_4'])


if __name__ == '__main__':
    main()
