"""Fixed-sample, inference-only USE v4 interface audit. Never creates an optimizer."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src'),
                str(ROOT/'outputs/use_v3_diagnostics'), str(ROOT/'outputs/use_v3/runtime')]
import numpy as np
import torch
from torch.nn import functional as F
import yaml
import run_use_v4 as run
import pretrain_use_v4 as pt
import common as old
from representation_audit import trace
from analysis_card import analysis_card
from landslide_seg.models.frozen_use import state_sha256
from landslide_seg.models.use_v4_mit import load_visual
from landslide_seg.models.use_v4_reconstruction import Segmenter
from landslide_seg.models.use_downstream import build_visual
from landslide_seg.data.optical_observation import collate_optical

MIT = ROOT/'weights/segformer/segformer-b0-finetuned-ade-512-512'
V3 = ROOT/'outputs/use_v3/runs/pretrain/best_unlabeled.pt'
V4 = ROOT/'outputs/use_v4/runs/pretrain/best_unlabeled.pt'
SEG = ROOT/'outputs/use_v4/runs/segment/best.pt'
DIRECT = ROOT/'outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234/best.pt'
PLAN = Path('N:/Users/yzh/Data/USE_v4_MiT89E5818EE5A3_8FE8BBAFE6AD89E88CAEE592.md')
SEED = 1234
CAP = 256


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def seed(key):
    return int(hashlib.sha256(f'{SEED}:{key}'.encode()).hexdigest()[:16], 16)


def quant(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    good = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, nonfinite_fraction=None, mean=None, std=None, rms=None, quantiles=None)
    if not len(good):
        return dict(n=len(x), nonfinite_fraction=1., mean=None, std=None, rms=None, quantiles=None)
    return dict(n=len(x), nonfinite_fraction=float(1-len(good)/len(x)), mean=float(good.mean()),
                std=float(good.std()), rms=float(np.sqrt(np.mean(good**2))),
                quantiles=dict(zip(('min', 'p01', 'p25', 'median', 'p75', 'p99', 'max'),
                                   map(float, np.quantile(good, [0, .01, .25, .5, .75, .99, 1])))))


def spectrum(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return dict(n=len(x), dimensions=x.shape[1], effective_rank=None, eigenvalues=[])
    x = x-x.mean(0)
    eigen = np.maximum(np.linalg.eigvalsh(x.T@x/len(x)), 0)
    # Discard numerical eigenvalues beyond the centered matrix's exact rank bound.
    eigen[:max(0, len(eigen)-min(len(x)-1, x.shape[1]))] = 0
    singular = np.sqrt(eigen)
    singular = singular[singular > max(float(singular.max()), 1.)*1e-9]
    p = singular/singular.sum() if singular.sum() else singular
    return dict(n=len(x), dimensions=x.shape[1], algebraic_rank_cap=min(len(x)-1, x.shape[1]),
                effective_rank=float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.,
                covariance_trace=float(eigen.sum()), eigenvalues=eigen[::-1].tolist())


def grid(x):
    if x.ndim == 3:
        b, n, c = x.shape
        h = int(n**.5)
        assert h*h == n
        return x.transpose(1, 2).reshape(b, c, h, h)
    return x


def mask_at(mask, x):
    return F.interpolate(mask[:, None].float(), size=x.shape[-2:], mode='nearest')[:, 0].bool()


def matrix(x, mask):
    return x[0].permute(1, 2, 0)[mask[0]].float().cpu().numpy()


def chosen(x, key):
    return np.random.default_rng(seed(key)).choice(len(x), min(CAP, len(x)), replace=False)


def spatial(x, mask):
    x = x[0].permute(1, 2, 0).float().cpu().numpy().astype(np.float64)
    valid = mask[0].cpu().numpy()
    center = x[valid].mean(0) if valid.any() else np.zeros(x.shape[-1])
    result = {}
    for distance, name in ((1, 'neighbor'), (min(8, max(1, x.shape[0]//2)), 'distant')):
        aa, bb = [], []
        for axis in (0, 1):
            a = [slice(None), slice(None)]; b = a.copy()
            a[axis] = slice(None, -distance); b[axis] = slice(distance, None)
            ok = valid[tuple(a)] & valid[tuple(b)]
            aa.append(x[tuple(a)][ok]); bb.append(x[tuple(b)][ok])
        a, b = np.concatenate(aa), np.concatenate(bb)
        def cos(u, v):
            return float(np.mean(np.sum(u*v, 1)/np.maximum(np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1), 1e-12))) if len(u) else None
        result[name] = dict(distance_native_pixels=distance, pairs=len(a), raw_cosine=cos(a, b),
                            centered_cosine=cos(a-center, b-center),
                            difference_rms=float(np.sqrt(np.mean((a-b)**2))) if len(a) else None)
    return result


class Collector:
    def __init__(self):
        self.rows = []; self.means = defaultdict(list)

    def add(self, group, sid, x, valid):
        x = grid(x)
        valid = mask_at(valid, x)
        m = matrix(x, valid)
        if not len(m):
            raise ValueError(f'No valid native centers: {group}/{sid}')
        var = m.var(0)
        row = dict(group=group, sample_id=sid, shape=list(x.shape), valid_token_fraction=float(valid.float().mean()),
                   activation_bytes=x.numel()*x.element_size(), values=quant(m), channel_variance=quant(var),
                   tiny_variance_channel_fraction=float(np.mean(var < 1e-8)),
                   token_spectrum=spectrum(m[chosen(m, f'{sid}/grid{x.shape[-2]}x{x.shape[-1]}')]), spatial=spatial(x, valid))
        self.rows.append(row); self.means[group].append(m.mean(0))
        return row

    def export(self):
        groups = {}
        for name, means in self.means.items():
            rows = [r for r in self.rows if r['group'] == name]
            groups[name] = dict(samples=len(rows), image_mean_spectrum=spectrum(means),
                token_effective_rank_mean=float(np.mean([r['token_spectrum']['effective_rank'] for r in rows])),
                rms_mean=float(np.mean([r['values']['rms'] for r in rows])),
                tiny_variance_channel_fraction_mean=float(np.mean([r['tiny_variance_channel_fraction'] for r in rows])),
                neighbor_centered_cosine_mean=float(np.mean([r['spatial']['neighbor']['centered_cosine'] for r in rows])),
                neighbor_difference_rms_mean=float(np.mean([r['spatial']['neighbor']['difference_rms'] for r in rows])))
        return dict(groups=groups, per_sample=self.rows)


def cka(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a -= a.mean(0); b -= b.mean(0)
    return float(np.sum((a.T@b)**2)/max(np.linalg.norm(a.T@a)*np.linalg.norm(b.T@b), 1e-30))


def input_capture(encoder, callback):
    values = {}; handles = []
    def before(m, a):
        values['pre'] = a[0]; values['ln_calls'] = values.get('ln_calls', 0)+1
    handles.append(encoder.patch_embeddings[0].layer_norm.register_forward_pre_hook(before))
    handles.append(encoder.patch_embeddings[0].layer_norm.register_forward_hook(lambda m, a, o: values.__setitem__('post', o)))
    def conv(m, a, o):
        values['rgb_conv_calls'] = values.get('rgb_conv_calls', 0)+1
    handles.append(encoder.patch_embeddings[0].proj.register_forward_hook(conv))
    try:
        result = callback()
    finally:
        for h in handles: h.remove()
    assert values['ln_calls'] == 1
    return result, values


def metrics(prob, target):
    from landslide_seg.training.segformer_metrics import pixel_average_precision
    p = np.concatenate(prob); t = np.concatenate(target).astype(bool); pred = p >= .5
    tp = int((pred&t).sum()); fp = int((pred&~t).sum()); fn = int((~pred&t).sum()); tn = int((~pred&~t).sum())
    return dict(n=len(p), positives=int(t.sum()), AP=pixel_average_precision(p, t),
                IoU=tp/max(tp+fp+fn, 1), Precision=tp/max(tp+fp, 1), Recall=tp/max(tp+fn, 1),
                FPR=fp/max(fp+tn, 1), tp=tp, fp=fp, fn=fn, tn=tn)


def prepare(out):
    split, stats, cache_audit, members = pt.inputs('multisensor')
    previous = read(ROOT/'outputs/use_v3_diagnostics/unlabeled_samples.json')['PRETRAIN_VAL']
    selected = {}
    for sensor, rows in split['PRETRAIN_VAL'].items():
        lookup = {r['file']: r for r in rows}
        first = [lookup[r['file']] for r in previous[sensor]]
        rest = sorted([r for r in rows if r['file'] not in {x['file'] for x in first}], key=lambda r: r['file'])
        random.Random(SEED).shuffle(rest)
        selected[sensor] = (first+rest)[:32]
        assert len(selected[sensor]) == 32
    cfg = yaml.safe_load(run.CONFIG.read_text())
    _, dev = run.rt.datasets(cfg)
    cases = read(ROOT/'outputs/use_v3_diagnostics/manifest.json')['dev_cases']
    assert len(cases) == 22
    lookup = {r['sample_id']: i for i, r in enumerate(dev.records)}
    for case in cases: assert case['sample_id'] in lookup
    sources = [Path(__file__), PLAN, V3, V4, SEG, DIRECT, ROOT/'outputs/use_v4/runs/pretrain/initial.pt',
        run.CONFIG, ROOT/'manifests/v4r1/stats_core25.json', ROOT/'outputs/use/quality/index.json',
        ROOT/'outputs/use_v3_diagnostics/manifest.json', ROOT/'outputs/use_v3_diagnostics/unlabeled_samples.json',
        ROOT/'outputs/use_v3_diagnostics/common.py', ROOT/'outputs/use_v3_diagnostics/representation_audit.py',
        ROOT/'outputs/use_v4/engineering/repair_b1_cpu_tests.xml',
        *[ROOT/f'src/landslide_seg/models/{name}.py' for name in ('use_v3', 'use_v4_mit', 'use_v4_reconstruction', 'segformer', 'spectral_stem')],
        MIT/'config.json', MIT/'model.safetensors', MIT/'preprocessor_config.json']
    source_hashes = {str(p): sha(p) for p in sources}
    source_members = []
    for case in cases:
        r = dev.records[lookup[case['sample_id']]]
        side = dev.quality[case['sample_id']]
        actual_cache = sha(Path(r['cache']))
        actual_quality = sha(dev.quality_root/side['file'])
        assert actual_cache == r['cache_sha256']
        assert actual_quality == side['sha256']
        source_members.append(dict(case=case, record=r, quality=side,
                                   actual_cache_sha256=actual_cache, actual_quality_sha256=actual_quality))
    manifest = dict(status='DRY_RUN', head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        baseline_head='578749d68a46be20bd1e60790a61c4c8798201a3', source_sha256=source_hashes,
        unlabeled=selected, dev=source_members, input_statistics=stats, cache_audit_sha256=cache_audit,
        member_manifest_sha256=members, optimizer_updates=0, lora_updates=0, probe_updates=0, seed=SEED,
        sampling='Preserve old two per sensor in order; sort remaining filenames then random.Random(1234).shuffle; take 30.',
        native_coordinate_cap=CAP, coordinate_key='SHA256(1234:sample_id/gridHxW), identical centers across paths at the same native grid',
        dtype='float32', tf32=False, mode='eval + inference_mode; requires_grad=False',
        masks='Intersect all observed bands at original pixels; nearest native-grid centers, intersect actual encoder support. GT also intersects label valid. This does not assert the whole receptive field is cloud-free.',
        scope='64 PRETRAIN_VAL (32/sensor), fixed historical 22 DEV; descriptive, no selection or training.',
        caveats=['RGB references use physical reflectance RGB and ImageNet mean/std, no 512 resize: same-source RGB, nonstandard pretrained distribution.',
                 'Native RGB/direct10 attention cannot mask keys; invalid input set to physical zero/zscore zero respectively. Quantitative centers excluded; remote influence can remain.',
                 'A2 is supervised direct10 best, not original pretrained weights; S2 reflectance*10000 uses core25 DN stats.',
                 'Token rank uses up to 256 native centers/image; image rank uses 32 means/sensor and cannot exceed 31.',
                 'DEV subset composition intervention is not a trained ablation and not the full 669-tile result.'],
        argv=sys.argv, python=sys.executable, dependencies=dict(torch=torch.__version__, numpy=np.__version__))
    write(out/'manifest.json', manifest)
    return manifest, selected, stats, dev, lookup


def decoder_forward(model, obs):
    d = model.decoder; captured = {}; handles = []
    for name, module in [('fuse', d.linear_fuse), ('bn', d.batch_norm), ('base_native', d.classifier), ('delta', model.residual),
                         *[(f'project{i+1}', m) for i, m in enumerate(d.linear_c)]]:
        handles.append(module.register_forward_hook(lambda m, a, o, n=name: captured.__setitem__(n, o)))
    try: final = model(obs)
    finally:
        for h in handles: h.remove()
    captured['base'] = F.interpolate(captured['base_native'], size=final.shape[-2:], mode='bilinear', align_corners=False)
    captured['final'] = final
    torch.testing.assert_close(captured['base']+captured['delta'], final, rtol=0, atol=0)
    return captured


def run_audit(out, device):
    start = time.perf_counter()
    manifest, selected, stats, dev, lookup = prepare(out)
    torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    interface = Collector(); stages = Collector(); fines = Collector(); decoder = Collector()
    pairs = defaultdict(lambda: [[], []]); changes = []; initial_checks = []; calls = []
    branch_probs = defaultdict(list); targets = []; cases = []; bn_inputs = []; centroids = defaultdict(lambda: [[], []])
    figure_cases = []; edge_rows = []; repeat_checks = []
    with analysis_card(device) as resource, torch.inference_mode():
        v4 = run.load_encoder(V4).requires_grad_(False).eval().to(device)
        selected_v4_hash = state_sha256(v4)
        v3 = old.encoder('v3', device)
        initial = run.load_encoder(ROOT/'outputs/use_v4/runs/pretrain/initial.pt').requires_grad_(False).eval().to(device)
        rgb = load_visual().segformer.encoder.requires_grad_(False).eval().to(device)
        direct, _ = build_visual('bridge10', MIT)
        saved = torch.load(DIRECT, map_location='cpu', weights_only=False)
        direct.load_state_dict(saved['model'], strict=True); direct.requires_grad_(False).eval().to(device)
        direct_encoder = direct.model.segformer.encoder
        segment = Segmenter(v4)
        saved = torch.load(SEG, map_location='cpu', weights_only=False)
        segment.load_state_dict(saved['model'], strict=True); segment.requires_grad_(False).eval().to(device)
        assert state_sha256(segment) == saved['model_sha256']
        assert state_sha256(v4) == selected_v4_hash
        models = dict(v4=v4, v3=v3, initial=initial, rgb=rgb, direct=direct, segment=segment)
        before = {k: state_sha256(v) for k, v in models.items()}
        bn = segment.decoder.batch_norm
        bn_running_mean = bn.running_mean.float().cpu().numpy().copy()
        bn_running_var = bn.running_var.float().cpu().numpy().copy()
        for sensor, rows in selected.items():
            for number, row in enumerate(rows):
                sid = row['file']; obs = pt.load_rows([row], stats, device)
                mask = obs['pixel_valid'].all(1)
                f4, cap = input_capture(v4.mit, lambda: v4(obs, diagnostics=True))
                assert cap.get('rgb_conv_calls', 0) == 0
                for which in ('pre', 'post'): interface.add(f'{sensor}/v4/{which}', sid, cap[which], mask)
                interface.add(f'{sensor}/v4/Q5', sid, f4['queries'][:, :, 4].transpose(1, 2).reshape(1, 32, 64, 64), mask)
                interface.add(f'{sensor}/v4/adapter', sid, f4['mit_input'], mask)
                paths = {'v4': f4['context']}
                if sensor == 's2':
                    mean = torch.tensor(stats[sensor]['mean'], device=device)[None, :, None, None]
                    std = torch.tensor(stats[sensor]['std'], device=device)[None, :, None, None]
                    physical = torch.where(obs['pixel_valid'], obs['x']*std+mean, 0)
                    rgb_input = (physical[:, [2, 1, 0]]-physical.new_tensor([.485, .456, .406])[None, :, None, None])/physical.new_tensor([.229, .224, .225])[None, :, None, None]
                    core = read(ROOT/'manifests/v4r1/stats_core25.json')
                    dn = physical*10000
                    direct_input = (dn-dn.new_tensor(core['mean'])[None, :, None, None])/dn.new_tensor(core['std'])[None, :, None, None]
                    direct_input = torch.where(obs['pixel_valid'], direct_input, 0)
                    for name, encoder, x in (('rgb', rgb, rgb_input), ('direct10', direct_encoder, direct_input)):
                        features, entry = input_capture(encoder, lambda e=encoder, x=x: e(x, output_hidden_states=True).hidden_states)
                        paths[name] = features
                        assert entry['rgb_conv_calls'] == 1
                        for which in ('pre', 'post'): interface.add(f's2/{name}/{which}', sid, entry[which], mask)
                        calls.append(dict(sample_id=sid, path=name, ln_calls=entry['ln_calls'], first_conv_calls=entry['rgb_conv_calls']))
                    interface.add('s2/rgb/physical_input', sid, physical[:, [2, 1, 0]], mask)
                    for stage in (0, 1):
                        valid = mask_at(mask, paths['v4'][stage])
                        for a, b in (('v4', 'rgb'), ('v4', 'direct10'), ('rgb', 'direct10')):
                            ma, mb = matrix(paths[a][stage], valid), matrix(paths[b][stage], valid)
                            pick = chosen(ma, sid+str(stage))
                            pairs[f'stage{stage+1}/{a}--{b}'][0].append(ma[pick]); pairs[f'stage{stage+1}/{a}--{b}'][1].append(mb[pick])
                calls.append(dict(sample_id=sid, path='v4', ln_calls=cap['ln_calls'], first_conv_calls=0,
                                  stage1_block_input='Patch-LN output with unsupported tokens zeroed; no positional embedding module.'))
                for path, values in paths.items():
                    for i, x in enumerate(values):
                        geometry_row = stages.add(f'{sensor}/{path}/stage{i+1}', sid, x, mask)
                        h, w = x.shape[-2:]; n = h*w
                        key_count = (h//[8,4,2,1][i])*(w//[8,4,2,1][i])
                        attention_bytes = [1,2,5,8][i]*n*key_count*4
                        geometry_row['temporary_activation_estimate'] = dict(attention_scores_bytes=attention_bytes,
                            expanded_MLP_bytes=4*x.numel()*4,
                            largest_single_tensor_bytes=max(attention_bytes, 4*x.numel()*4),
                            scope='Static FP32 batch1 single-block tensor sizes, not measured peak allocator/workspace or sum over layers.')
                f3, captured = trace(v3, obs)
                fine_early = captured['fine_pre']
                for name, x in (('v3_early', fine_early), ('v3_final', f3['fine']), ('v4_early', f4['fine'])):
                    fines.add(f'{sensor}/{name}', sid, x, mask)
                delta = f3['fine']-fine_early
                local_valid = mask[:, None].expand_as(delta)
                ratio = float(delta[local_valid].norm()/fine_early[local_valid].norm().clamp_min(1e-12))
                locations = (delta.norm(dim=1)/fine_early.norm(dim=1).clamp_min(1e-12))[mask].cpu().numpy()
                changes.append(dict(sensor=sensor, sample_id=sid, final_minus_early_norm_ratio=ratio, location_ratios=quant(locations)))
                if number == 0:
                    init_early = initial.frontend(obs)[0]
                    torch.testing.assert_close(init_early, fine_early, rtol=1e-5, atol=1e-6)
                    initial_checks.append(dict(sensor=sensor, sample_id=sid, max_abs_difference=float((init_early-fine_early).abs().max()), pass_close=True))
                    repeated = v4(obs)
                    for key in ('fine',): torch.testing.assert_close(repeated[key], f4[key], rtol=0, atol=0)
                    for x, y in zip(repeated['context'], f4['context']): torch.testing.assert_close(x, y, rtol=0, atol=0)
                    repeat_checks.append(dict(sensor=sensor, sample_id=sid, tensors='Fine and Context1–4', max_abs_difference=0.))
                write(out/'progress.json', dict(phase='unlabeled', sensor=sensor, complete=number+1, total=32, elapsed_s=time.perf_counter()-start))
            print(f'Completed {sensor} 32 observations', flush=True)
        # Fixed DEV order; GT stays outside every encoder/model call.
        display_ids = [manifest['dev'][i]['case']['sample_id'] for i in (0, 3, 4)]
        for number, entry in enumerate(manifest['dev']):
            sid = entry['case']['sample_id']; sample = dev[lookup[sid]]
            obs = collate_optical([sample['image']]).to(device)
            valid = sample['valid'].numpy().astype(bool) & sample['image'].pixel_valid.all(0).numpy()
            target = sample['mask'].numpy().squeeze().astype(bool)
            mask = torch.tensor(valid, device=device)[None]
            cap = decoder_forward(segment, obs)
            if number == 0:
                repeated = decoder_forward(segment, obs)
                for key in cap: torch.testing.assert_close(cap[key], repeated[key], rtol=0, atol=0)
                repeat_checks.append(dict(role='DEV', sample_id=sid, tensors=list(cap), max_abs_difference=0.))
            for name, x in cap.items(): decoder.add(f'DEV/{name}', sid, x, mask)
            fm = matrix(cap['fuse'], mask_at(mask, cap['fuse']))
            bn_inputs.append(fm[chosen(fm, sid+'BN')])
            base = cap['base'][0, 0].cpu().numpy(); final = cap['final'][0, 0].cpu().numpy()
            delta = cap['delta'][0, 0].cpu().numpy()
            bp = torch.sigmoid(cap['base'])[0, 0].cpu().numpy(); fp = torch.sigmoid(cap['final'])[0, 0].cpu().numpy()
            branch_probs['base'].append(bp[valid]); branch_probs['final'].append(fp[valid]); targets.append(target[valid])
            bp_mask = bp >= .5; fp_mask = fp >= .5
            changed = bp_mask != fp_mask
            cases.append(dict(sample_id=sid, event=sample['event_id'], n_valid=int(valid.sum()),
                base=metrics([bp[valid]], [target[valid]]), final=metrics([fp[valid]], [target[valid]]),
                delta_to_base_norm_ratio=float(np.linalg.norm(delta[valid])/max(np.linalg.norm(base[valid]), 1e-12)),
                opposing_sign_fraction=float(np.mean((base[valid]*delta[valid]) < 0)),
                changed_predictions=int((changed&valid).sum()),
                removed_fp=int((bp_mask&~fp_mask&~target&valid).sum()), added_fp=int((~bp_mask&fp_mask&~target&valid).sum())))
            f4 = v4(obs)
            f3, traced = trace(v3, obs)
            for name, x in (('v3_early', traced['fine_pre']), ('v3_final', f3['fine']), ('v4_early', f4['fine'])):
                fines.add(f'DEV/{name}', sid, x, mask)
                values = x[0].float().cpu().numpy()
                boundary_values, flat_values = [], []
                for axis in (0, 1):
                    gradient = np.linalg.norm(np.diff(values, axis=axis+1), axis=0)
                    a = [slice(None), slice(None)]; b = a.copy()
                    a[axis] = slice(None, -1); b[axis] = slice(1, None)
                    ok = valid[tuple(a)] & valid[tuple(b)]
                    boundary = target[tuple(a)] != target[tuple(b)]
                    boundary_values.extend(gradient[ok&boundary].tolist())
                    flat_values.extend(gradient[ok&~boundary].tolist())
                edge_rows.append(dict(sample_id=sid, path=name, boundary_gradient=quant(boundary_values),
                                      nonboundary_gradient=quant(flat_values), definition='Neighbor feature L2 change, original128 grid; label boundary descriptive only.'))
            for i, x in enumerate(f4['context']):
                native_mask = mask_at(mask, x)
                fractions = F.adaptive_avg_pool2d(torch.tensor(target, device=device).float()[None, None]*mask[:, None], x.shape[-2:])[0, 0]
                denominator = F.adaptive_avg_pool2d(mask[:, None].float(), x.shape[-2:])[0, 0]
                fractions = (fractions/denominator.clamp_min(1e-12))[native_mask[0]].cpu().numpy()
                values = matrix(x, native_mask).astype(np.float64)
                for label, w in enumerate((1-fractions, fractions)):
                    centroids[f'stage{i+1}'][label].append((values*w[:, None]).sum(0).tolist()+[float(w.sum())])
            if sid in display_ids:
                raw = sample['image'].x.numpy()
                core = dev.stats
                physical = (raw*np.asarray(core['std'])[:, None, None]+np.asarray(core['mean'])[:, None, None])*.0001
                figure_cases.append(dict(sample_id=sid, rgb=np.clip(physical[[2, 1, 0]].transpose(1, 2, 0)*3, 0, 1),
                                         target=target, valid=valid, base=bp, final=fp))
            write(out/'progress.json', dict(phase='DEV', complete=number+1, total=22, elapsed_s=time.perf_counter()-start))
        after = {k: state_sha256(v) for k, v in models.items()}
        assert before == after
        assert all(not p.requires_grad and p.grad is None for model in models.values() for p in model.parameters())
        assert all(not m.training for model in models.values() for m in model.modules())
        bn_values = np.concatenate(bn_inputs).astype(np.float64)
        actual_mean, actual_var = bn_values.mean(0), bn_values.var(0)
        bn_stats = dict(training=bn.training, eps=bn.eps, running_mean=bn_running_mean.tolist(), running_var=bn_running_var.tolist(),
                        input_sample_mean=actual_mean.tolist(), input_sample_var=actual_var.tolist(),
                        normalized_mean_offset=quant((actual_mean-bn_running_mean)/np.sqrt(bn_running_var+bn.eps)),
                        variance_ratio=quant(actual_var/(bn_running_var+bn.eps)),
                        affine_weight=quant(bn.weight.cpu().numpy()), affine_bias=quant(bn.bias.cpu().numpy()),
                        sampling='Up to 256 valid native-center BN inputs/case; 22 cases, no recalibration.')
    for path, value in manifest['source_sha256'].items(): assert sha(path) == value, path
    geometry = stages.export()
    geometry['class_centroids_DEV'] = {}
    for name, classes in centroids.items():
        means = []
        for rows in classes:
            sums = np.sum(rows, axis=0); means.append(sums[:-1]/max(sums[-1], 1e-12))
        geometry['class_centroids_DEV'][name] = dict(background=means[0].tolist(), foreground=means[1].tolist(),
            normalized_distance=float(np.linalg.norm(means[1]-means[0])/max(np.linalg.norm(means[0]), 1e-12)),
            definition='Area-weighted original GT fraction at native cells; descriptive, no classifier/threshold fitting.')
    alignment = interface.export(); alignment['execution_counts'] = calls
    alignment['centered_linear_CKA'] = {k: cka(np.concatenate(a), np.concatenate(b)) for k, (a, b) in pairs.items()}
    alignment['processor'] = read(MIT/'preprocessor_config.json')
    alignment['preprocessing'] = dict(rgb='Physical reflectance B04/B03/B02, subtract [.485,.456,.406], divide [.229,.224,.225], no clipping/no resize; nonstandard pretrained distribution.',
        direct10='Physical reflectance * 10000, core25 DN zscore, invalid band pixels zero; supervised direct10 weights.',
        v4='Actual PRETRAIN_VAL normalized input with physical metadata; Q5→adapter→original patch LN once, first RGB conv bypassed.',
        first_block='Original stage1 patch LN output after unsupported-token zeroing; no extra LayerNorm or positional embedding inserted.',
        equivalent_staged_test='Reuse repair_b1_cpu_tests.xml native/staged all-valid equality tests at 127/128/129; not rerun.')
    fine_decoder = dict(fine=fines.export(), v3_final_minus_early=changes, initial_v4_equals_v3_early=initial_checks,
        decoder=decoder.export(), batch_norm=bn_stats, composition_cases=cases, DEV_fine_edges=edge_rows,
        fixed22_metrics={k: metrics(p, targets) for k, p in branch_probs.items()},
        caveat='Same-checkpoint base/final forward decomposition is an out-of-distribution inference intervention, not trained Fine ablation. Fixed22 differs from full DEV669.')
    write(out/'stage1_alignment.json', alignment); write(out/'stage_geometry.json', geometry); write(out/'fine_decoder.json', fine_decoder)
    make_figures(out, alignment, geometry, fine_decoder, figure_cases)
    manifest.update(status='COMPLETED', elapsed_s=time.perf_counter()-start, resource=resource,
                    model_state_before=before, model_state_after=after, source_hashes_unchanged=True,
                    gradient_tensors=0, models_eval=True, numerical_assertions='LN call counts, initial early equality, strict checkpoint load/hash, exact logit sum, before/after all-state hash')
    manifest['repeat_inference_checks'] = repeat_checks
    manifest['segmentation_USE_equals_selected_pretrain_state'] = True
    write(out/'manifest.json', manifest)
    write(out/'acceptance.json', dict(status='PASS_WITH_LIMITATIONS', optimizer_updates=0, model_parameters_and_buffers_unchanged=before == after,
        sources_unchanged=True, fixed_samples=dict(s2=32, landsat=32, DEV=22), figures=3,
        output_sha256={p.name: sha(p) for p in (out/'manifest.json', out/'stage1_alignment.json', out/'stage_geometry.json', out/'fine_decoder.json')}))
    print(json.dumps(dict(status='COMPLETED', elapsed_s=manifest['elapsed_s'], fixed22=fine_decoder['fixed22_metrics']), ensure_ascii=False), flush=True)


def make_figures(out, a, b, c, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for i, metric in enumerate(('rms_mean', 'token_effective_rank_mean')):
        for j, path in enumerate(('rgb', 'direct10', 'v4')):
            values = [a['groups'][f's2/{path}/{which}'][metric] for which in ('pre', 'post')]
            axes[i].plot([0, 1], values, 'o-', label={'rgb':'同源RGB参考', 'direct10':'监督十波段直入', 'v4':'USE v4注入'}[path])
        axes[i].set_xticks([0, 1], ['Stage1 LN前', 'Stage1 LN后']); axes[i].set_title('有效位置RMS' if i == 0 else '每图原生Token有效秩（最多256位置）'); axes[i].legend()
    fig.suptitle('相同32个S2观测；RGB保留128网格，非标准预训练分布')
    fig.savefig(out/'stage1_alignment.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for path, label in (('s2/v4','v4 S2'), ('landsat/v4','v4 Landsat'), ('s2/rgb','同源RGB参考'), ('s2/direct10','监督十波段直入')):
        groups = [b['groups'][f'{path}/stage{i}'] for i in range(1, 5)]
        axes[0].plot(range(1, 5), [g['token_effective_rank_mean'] for g in groups], 'o-', label=label)
        axes[1].plot(range(1, 5), [g['image_mean_spectrum']['effective_rank'] for g in groups], 'o-', label=label)
    axes[0].set_title('每图原生Token有效秩\n网格32²→16²→8²→4²'); axes[1].set_title('32图均值向量的有效秩\n秩上限31，与Token秩不同')
    for ax in axes[:2]: ax.set_xticks(range(1,5)); ax.set_xlabel('MiT阶段'); ax.legend(fontsize=8)
    for i, sensor in enumerate(('s2', 'landsat')):
        values = [f['final_minus_early_norm_ratio'] for f in c['v3_final_minus_early'] if f['sensor'] == sensor]
        axes[2].boxplot(values, positions=[i], widths=.5)
    axes[2].set_xticks([0, 1], ['S2', 'Landsat']); axes[2].set_title('v3 Fine最终值−早期值\n范数 / 早期Fine范数')
    fig.savefig(out/'stage_geometry_fine.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(len(cases), 6, figsize=(16, 3*len(cases)), constrained_layout=True)
    for row, case in enumerate(cases):
        valid, target = case['valid'], case['target']
        arrays = [case['rgb'], target.astype(float), case['base'], case['final']]
        for key in ('base', 'final'):
            pred = case[key] >= .5
            rgb = np.zeros((*target.shape,3)); rgb[pred&~target] = [1,0,0]; rgb[~pred&target] = [0,.35,1]; rgb[pred&target] = [0,.8,0]
            arrays.append(rgb)
        for col, array in enumerate(arrays):
            if array.ndim == 2:
                image = plt.get_cmap('viridis' if col >= 2 else 'gray')(array)[..., :3]
            else: image = array.copy()
            image[~valid] = .5
            axes[row,col].imshow(image, vmin=0, vmax=1); axes[row,col].axis('off')
            if row == 0: axes[row,col].set_title(['RGB ×3显示','GT','主分支概率','最终概率','主分支差错','最终差错'][col])
        axes[row,0].text(.01, .99, case['sample_id'], transform=axes[row,0].transAxes, fontsize=8,
                         color='white', va='top', bbox=dict(facecolor='black', alpha=.65, edgecolor='none'))
    fig.suptitle('预先固定案例；概率0–1；红=误报，蓝=漏检，绿=正确前景，灰=排除像素；0.5阈值')
    fig.savefig(out/'decoder_composition.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Lock files and 64+22 members without loading model weights onto GPU.')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/use_v4_interface_diagnostics')
    args = parser.parse_args(); out = args.output_dir.resolve()
    if (out/'acceptance.json').exists(): raise FileExistsError('Completed evidence exists; choose another output directory.')
    out.mkdir(parents=True, exist_ok=True)
    (out/('dry_run_command.txt' if args.dry_run else 'command.txt')).write_text(subprocess.list2cmdline([sys.executable, *sys.argv]), encoding='utf-8')
    if args.dry_run:
        manifest, *_ = prepare(out)
        write(out/'dry_run_manifest.json', manifest)
        print(json.dumps(dict(status='DRY_RUN_PASS', s2=32, landsat=32, DEV=22, optimizer_updates=0, head=manifest['head'])))
    else: run_audit(out, args.device)


if __name__ == '__main__': main()
