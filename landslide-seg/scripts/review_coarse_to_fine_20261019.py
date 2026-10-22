"""Read-only checkpoint diagnostics; writes new review artifacts only.

Conditional interventions at inference are sensitivity probes, not retrained ablations.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from landslide_seg.config import resolve_path
from landslide_seg.data.segformer_dataset import S2SegDataset, collate, filter_split, load_stats, read_manifest
from landslide_seg.training.segformer_trainer import load_checkpoint


def main():
    torch.set_num_threads(4)
    root = ROOT / 'outputs/segformer/s2_10band_v3_p20'
    report = {'note': 'Inference interventions, no optimizer updates. Not retrained ablations.', 'device': 'cuda:1', 'curves': {}, 'metadata': {}}
    for exp in ['A1', 'A1_AUX', 'CF1', 'CF0', 'CF1_B', 'U0']:
        d = root / exp / '1234'
        summary = json.loads((d / 'train_summary.json').read_text(encoding='utf-8'))
        hist = summary['history']
        best = max(hist, key=lambda x: x['val_iou'])
        report['curves'][exp] = {'global_step': summary.get('global_step'), 'best': best, 'last': hist[-1]}
        meta = json.loads((d / 'run_meta.json').read_text(encoding='utf-8'))
        report['metadata'][exp] = {k: meta.get(k) for k in ['load_info', 'model_identity', 'seed_info']}
        print(exp, 'best', best['val_iou'], 'last', hist[-1]['val_iou'], 'loss', best['train_loss'], '->', hist[-1]['train_loss'], flush=True)

    report['interventions'] = {}
    for exp in ['CF1', 'A1']:
        model, ckpt = load_checkpoint(root / exp / '1234/best.pt', device='cuda:1')
        cfg = ckpt['config']['segformer']
        records = filter_split(read_manifest(resolve_path(cfg['data']['manifest'])), 'val')
        ds = S2SegDataset(records, load_stats(resolve_path(cfg['data']['stats'])), preload=True)
        variants = ['normal', 'coarse', 'stem_zero'] + (['condition_zero', 'condition_flip'] if exp == 'CF1' else [])
        acc = {name: dict(tp=0, fp=0, fn=0, valid=0, flipped=0, abs_prob_diff=0.) for name in variants}
        shapes = {}
        with torch.inference_mode():
            for i in range(0, len(ds), 16):
                batch = collate([ds[j] for j in range(i, min(i + 16, len(ds)))])
                x = batch['image'].to('cuda:1')
                y = batch['mask'].to('cuda:1').bool().unsqueeze(1)
                v = batch['valid'].to('cuda:1').bool().unsqueeze(1)
                out = model(x, return_features=True)
                d, c, s = out['decoder_feat'], out['base_logits'], model.stem(x)
                shapes = {'input': list(x.shape[1:]), 'decoder': list(d.shape[1:]), 'stem': list(s.shape[1:]), 'output': list(c.shape[1:])}
                p = c.sigmoid()
                if exp == 'CF1':
                    logits = {'normal': out['logits'], 'coarse': c,
                              'stem_zero': c + model.refiner(d, torch.zeros_like(s), p, x.shape[-2:]),
                              'condition_zero': c + model.refiner(d, s, torch.zeros_like(p), x.shape[-2:]),
                              'condition_flip': c + model.refiner(d, s, p.flip([-1, -2]), x.shape[-2:])}
                else:
                    logits = {'normal': out['logits'], 'coarse': c,
                              'stem_zero': c + model.fusion(d, torch.zeros_like(s), x.shape[-2:])}
                normal = logits['normal'].sigmoid()
                for name, logit in logits.items():
                    prob = logit.sigmoid()
                    pred = prob >= 0.5
                    a = acc[name]
                    a['tp'] += int((pred & y & v).sum())
                    a['fp'] += int((pred & ~y & v).sum())
                    a['fn'] += int((~pred & y & v).sum())
                    a['valid'] += int(v.sum())
                    a['flipped'] += int(((pred != (normal >= 0.5)) & v).sum())
                    a['abs_prob_diff'] += float(((prob - normal).abs() * v).sum())
        for a in acc.values():
            a.update(iou=a['tp']/max(1, a['tp']+a['fp']+a['fn']), precision=a['tp']/max(1,a['tp']+a['fp']), recall=a['tp']/max(1,a['tp']+a['fn']), prediction_flip_fraction=a['flipped']/max(1,a['valid']), mean_abs_prob_diff=a['abs_prob_diff']/max(1,a['valid']))
        inactive = sum(p.numel() for p in model.fusion.parameters()) if exp == 'CF1' else 0
        report['interventions'][exp] = {'split': 'val', 'n_tiles': len(ds), 'shapes': shapes, 'variants': acc, 'unused_fusion_parameters': inactive}
        print(exp, json.dumps(report['interventions'][exp]), flush=True)
        del model
        torch.cuda.empty_cache()
    dest = ROOT / 'outputs/segformer/coarse_to_fine/review_20261019'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'checkpoint_diagnostics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved', dest, flush=True)


if __name__ == '__main__':
    main()
