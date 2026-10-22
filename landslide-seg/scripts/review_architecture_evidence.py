"""Recompute recorded metric arithmetic and emit architecture-named review evidence.

Reads existing artifacts only; does not train or rerun inference.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads((ROOT / p).read_text(encoding='utf-8'))


def main():
    names = read('configs/experiment_names.json')['experiments']
    evaluation = read('outputs/segformer/single_head/eval_v2.json')
    report = {'verification': 'Recorded confusion-count recomputation, config/log and source review; no new training or inference.', 'models': []}
    for legacy, row in evaluation['experiments'].items():
        run = f'outputs/segformer/s2_10band_v3_p20/{legacy}/1234'
        train = read(f'{run}/train_summary.json')
        item = dict(names[legacy], legacy_artifact_dir=run,
                    parameters=train['params']['total'],
                    used_parameters=train['params']['total'] - (18561 if legacy == 'CF0_NOAUX' else 0),
                    optimizer_steps=train['global_step'], best_epoch=train['best_epoch'],
                    last_val_iou=train['history'][-1]['val_iou'], splits={})
        for split, metrics in row['split'].items():
            counts = metrics['pooled']
            expected = counts['tp'] / (counts['tp'] + counts['fp'] + counts['fn'])
            assert abs(expected - counts['iou']) < 1e-12, (legacy, split)
            item['splits'][split] = {'iou': expected, 'precision': counts['precision'], 'recall': counts['recall'], 'count_check_passed': True}
        report['models'].append(item)
    by_name = {i['canonical_name']: i for i in report['models']}
    residual = by_name[names['CF0_NOAUX']['canonical_name']]
    single = by_name[names['SINGLE_SUM']['canonical_name']]
    report['used_parameter_difference_residual_minus_single'] = residual['used_parameters'] - single['used_parameters']
    assert report['used_parameter_difference_residual_minus_single'] == 65
    coarse = read('outputs/segformer/coarse_localization/coarse_localization.json')
    report['coarse_threshold_audit'] = {
        'announced_threshold': coarse['frozen_threshold'],
        'table_threshold_from_source': 0.5,
        'rows': []}
    for legacy, splits in coarse['experiments'].items():
        for split, data in splits.items():
            report['coarse_threshold_audit']['rows'].append({
                'architecture_name': names[legacy]['architecture_name'], 'split': split,
                'table_iou_at_05': data['coarse']['pooled']['iou'],
                'pr_iou_at_selected_threshold': data['pr_curve'][str(coarse['frozen_threshold'])]['iou']})
    out = ROOT / 'outputs/segformer/research_review_20261019'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'architecture_evidence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
