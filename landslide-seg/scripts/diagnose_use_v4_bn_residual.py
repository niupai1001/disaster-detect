"""Zero-update TRAIN-global BN intervention and fixed 2x4 residual decomposition."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT/'outputs/use_v3/runtime')]
import run_use_v4 as run
from diagnose_use_v4_interface import decoder_forward, sha, read, quant
from analysis_card import analysis_card
from landslide_seg.models.use_v4_reconstruction import Segmenter
from landslide_seg.models.frozen_use import state_sha256
from landslide_seg.data.optical_observation import OpticalObservation, collate_optical
from landslide_seg.data.segformer_dataset import normalize_s2, S2_10_BANDS
from landslide_seg.models.spectral_contract import wavelengths_for
from landslide_seg.training.segformer_metrics import pixel_average_precision as ap

BASE = ROOT/'outputs/use_v4/runs/segment'
ENCODER = ROOT/'outputs/use_v4/runs/pretrain/best_unlabeled.pt'
CHECKPOINT = BASE/'best.pt'
PLAN = Path('N:/Users/yzh/Data/USE_v4_B8E480BDE98DA0E586B8E68BB7E586A7E9B6AEE783AFE6ADAEE592.md')
PARTS = ('BASE', 'MEAN', 'SPATIAL', 'FULL')
ALLOWED_BUFFERS = {'decoder.batch_norm.running_mean', 'decoder.batch_norm.running_var'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def moments(count, total, square):
    if count <= 1: raise ValueError('Unbiased variance needs count > 1')
    mean = total/count
    numerator = square-total*mean
    tolerance = np.maximum(np.abs(square), 1.)*np.finfo(np.float64).eps*64
    if (numerator < -tolerance).any(): raise ValueError('Negative variance beyond FP64 roundoff')
    variance = np.maximum(numerator, 0)/(count-1)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError('Nonfinite/zero BN channel variance')
    return mean, variance


def counts(prob, target):
    pred = np.asarray(prob) >= .5; target = np.asarray(target, dtype=bool)
    tp = int((pred&target).sum()); fp = int((pred&~target).sum())
    fn = int((~pred&target).sum()); tn = int((~pred&~target).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=len(target), positives=int(target.sum()),
        IoU=tp/max(tp+fp+fn, 1), Precision=tp/max(tp+fp, 1), Recall=tp/max(tp+fn, 1),
        FPR=fp/max(fp+tn, 1), predicted_positive_fraction=(tp+fp)/max(len(target), 1))


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        value = p.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_comparison(original, recalibrated):
    params_o = dict(original.named_parameters()); params_r = dict(recalibrated.named_parameters())
    assert params_o.keys() == params_r.keys()
    assert all(torch.equal(v, params_r[k]) for k,v in params_o.items())
    buffers_o = dict(original.named_buffers()); buffers_r = dict(recalibrated.named_buffers())
    assert buffers_o.keys() == buffers_r.keys()
    changed = [name for name,v in buffers_o.items() if not torch.equal(v, buffers_r[name])]
    assert set(changed) == ALLOWED_BUFFERS, changed
    return dict(parameter_tensors_equal=len(params_o), buffer_tensors_checked=len(buffers_o), changed_buffers=changed,
                all_other_buffers_equal=True, BN_affine_equal=True)


def train_optical(record, stats, quality_index):
    """Only x10 is decompressed from the TRAIN cache. Never access mask/valid keys."""
    if sha(Path(record['cache'])) != record['cache_sha256']: raise ValueError('Cache SHA mismatch')
    with np.load(record['cache']) as data:
        raw = data['x10']
    entry = quality_index[record['sample_id']]
    assert entry['cache_sha256'] == record['cache_sha256'] and entry['time_index'] == record['time_index']
    path = ROOT/'outputs/use/quality'/entry['file']
    if sha(path) != entry['sha256']: raise ValueError('Quality SHA mismatch')
    with np.load(path) as data: quality = data['pixel_valid'].copy()
    metadata = torch.tensor([[wave, 0., 10. if band in ('B02','B03','B04','B08') else 20.]
        for band,wave in zip(S2_10_BANDS, wavelengths_for(S2_10_BANDS))])
    known = torch.ones(10,3,dtype=torch.bool); known[:,1] = False
    return OpticalObservation(torch.from_numpy(normalize_s2(raw.astype(np.float32), stats)), S2_10_BANDS,
        metadata, known, torch.ones(10,dtype=torch.bool), torch.from_numpy(quality),
        dict(crs=record['crs'], bbox=record['bbox'], date=record['date'], units='core25_zscore',
             sensor='Sentinel-2', product_level='L2A', processing='S2_harmonized', target_gsd_m=10))


def prepare(out, device):
    config = yaml.safe_load(run.CONFIG.read_text())
    train, dev = run.rt.datasets(config)  # preload=False: no TRAIN arrays are opened here.
    assert not train._x and not train._mask and not dev.augment
    train_records = sorted(train.records, key=lambda r:r['sample_id'])
    dev_ids = {r['sample_id'] for r in dev.records}; train_ids = {r['sample_id'] for r in train_records}
    assert len(train_ids) == 2223 and len(dev_ids) == 669 and not (train_ids&dev_ids)
    assert {r['event_id'] for r in dev.records} == {'hiroshima','thrissur'}
    assert not ({r['event_id'] for r in train_records}&{'hiroshima','thrissur','italy','newzealand'})
    data = config['segformer']['data']
    sources = [Path(__file__), PLAN, run.CONFIG, ENCODER, CHECKPOINT, BASE/'evaluation_best.json',
        ROOT/data['manifest'], ROOT/data['stats'], ROOT/'manifests/v4r1/protocol.json',
        ROOT/'docs/assets/research/research_policy.json', ROOT/'outputs/use/quality/index.json',
        ROOT/'scripts/diagnose_use_v4_interface.py', ROOT/'outputs/use_v4_interface_diagnostics/manifest.json',
        ROOT/'src/landslide_seg/models/use_v4_mit.py', ROOT/'src/landslide_seg/models/use_v4_reconstruction.py',
        ROOT/'src/landslide_seg/models/spectral_stem.py', ROOT/'src/landslide_seg/data/use_dataset.py',
        ROOT/'src/landslide_seg/data/segformer_dataset.py', ROOT/'src/landslide_seg/data/optical_observation.py',
        ROOT/'src/landslide_seg/training/segformer_metrics.py']
    files = {str(p):sha(p) for p in sources}
    def compact(r):
        q = train.quality[r['sample_id']]
        return {**{k:r[k] for k in ('sample_id','event_id','split','cache','cache_sha256','date','time_index')},
                'quality_file':q['file'], 'quality_sha256':q['sha256']}
    manifest = dict(status='DRY_RUN', head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        worktree_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines(),
        source_sha256=files, TRAIN=list(map(compact,train_records)), DEV=list(map(compact,dev.records)),
        TRAIN_order='sample_id ascending', DEV_order='Original dataset order, matches prediction indices 0000–0668.',
        roles=dict(TRAIN='BN input moments only: x10 and image quality, no GT arrays', DEV='All 669, read-only GT evaluation; no calibration/selection'),
        conditions=dict(BN=['BN-O','BN-R'], composition=list(PARTS), threshold=.5),
        mutable_buffer_allowlist=sorted(ALLOWED_BUFFERS), num_batches_tracked_policy='unchanged',
        moments='All native BN input positions, per-channel FP64 count/sum/sumsq; unbiased sample variance.',
        mu_support='Any observed optical band quality-valid at an original pixel; independent of all GT/label-valid masks.',
        evaluation_mask='Exactly sample.valid from the original evaluator, not the stricter common-band mask used in the earlier22 diagnostic.',
        optimizer_updates=0, seed=1234, dtype='FP32', tf32=False, deterministic_algorithms=True, device=device,
        dependencies={name:version(name) for name in ('torch','numpy','transformers','safetensors','matplotlib','scipy','PyYAML')},
        python=platform.python_version(), executable=sys.executable, command=subprocess.list2cmdline([sys.executable,*sys.argv]),
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'],text=True).splitlines(),
        prior_fixed22=read(ROOT/'outputs/use_v3_diagnostics/manifest.json')['dev_cases'])
    write(out/'manifest.json',manifest)
    return manifest,train,train_records,dev


def evaluate(model, dev, condition, out, fixed_ids, baseline=False):
    historic = read(BASE/'evaluation_best.json')
    probs = {name:[] for name in PARTS}; logits = {name:[] for name in PARTS}; targets = []; events = []
    rows = []; visuals = {}; prediction_checks = []; max_identity = 0.; invariance = []
    display_ids = [read(ROOT/'outputs/use_v3_diagnostics/manifest.json')['dev_cases'][i]['sample_id'] for i in (0,3,4)]
    for index in range(len(dev)):
        sample = dev[index]; sid = sample['sample_id']; event = sample['event_id']
        obs = collate_optical([sample['image']]).to(next(model.parameters()).device)
        cap = decoder_forward(model, obs)
        base, delta = cap['base'], cap['delta']
        quality = obs['pixel_valid'].any(1,keepdim=True)
        assert quality.any()
        mu = delta[quality].mean()
        composed = dict(BASE=base, MEAN=base+mu, SPATIAL=base+(delta-mu), FULL=cap['final'])
        error = float((composed['MEAN']+composed['SPATIAL']-base-composed['FULL']).abs().max())
        max_identity = max(max_identity,error)
        torch.testing.assert_close(composed['MEAN']+composed['SPATIAL']-base,composed['FULL'],rtol=1e-5,atol=1e-5)
        valid = sample['valid'].numpy().astype(bool); target = sample['mask'].numpy().astype(bool)
        valid_target = target[valid]; targets.append(valid_target); events.append(event)
        b = base[0,0].cpu().numpy(); d = delta[0,0].cpu().numpy(); q = quality[0,0].cpu().numpy()
        covariance = np.corrcoef(b[q].astype(np.float64),d[q].astype(np.float64))[0,1]
        if not np.isfinite(covariance): covariance = None
        shape_metrics = dict(mu_delta=float(mu), std_delta=float(d[q].std()), rms_delta=float(np.sqrt(np.mean(d[q]**2))),
            rms_base=float(np.sqrt(np.mean(b[q]**2))), corr_base_delta=None if covariance is None else float(covariance),
            quality_support_fraction=float(q.mean()), eval_valid_fraction=float(valid.mean()))
        current = {}
        for part, x in composed.items():
            assert torch.isfinite(x).all()
            probability = x.sigmoid()[0,0].cpu().numpy()
            score = x[0,0].cpu().numpy()
            p = probability[valid].copy(); l = score[valid].copy()
            probs[part].append(p); logits[part].append(l)
            metrics = counts(p,valid_target); metrics['AP'] = ap(p,valid_target); metrics['logit_AP'] = ap(l,valid_target)
            current[part] = (probability,metrics)
            row = dict(BN=condition,composition=part,sample_id=sid,event=event,fixed22=sid in fixed_ids,**shape_metrics,**metrics,
                sigmoid_zero_fraction=float(np.mean(p==0)),sigmoid_one_fraction=float(np.mean(p==1)))
            base_pred = current['BASE'][0][valid] >= .5; pred = p >= .5
            row.update(removed_FP=int((base_pred&~pred&~valid_target).sum()), added_FP=int((~base_pred&pred&~valid_target).sum()),
                       recovered_TP=int((~base_pred&pred&valid_target).sum()), lost_TP=int((base_pred&~pred&valid_target).sum()))
            rows.append(row)
        exact_base = ap(b[valid].astype(np.float64),valid_target)
        exact_mean = ap(b[valid].astype(np.float64)+float(mu),valid_target)
        if exact_base is not None: assert abs(exact_base-exact_mean) < 1e-12
        invariance.append(dict(sample_id=sid,exact_logit_shift_AP_equal=True,
            probability_AP_difference=None if exact_base is None else current['MEAN'][1]['AP']-current['BASE'][1]['AP']))
        if baseline:
            path = BASE/'predictions/best'/f'{index:04d}.npz'
            with np.load(path) as previous:
                assert str(previous['sample_id']) == sid and np.array_equal(previous['valid'],valid)
                assert np.array_equal(previous['mask'],target)
                difference = float(np.max(np.abs(previous['prob']-current['FULL'][0])))
                exact = bool(np.array_equal(previous['prob'],current['FULL'][0]))
            hist_tile = historic['tiles'][index]
            assert hist_tile['sample_id'] == sid
            same_counts = all(current['FULL'][1][key] == hist_tile[key] for key in ('tp','fp','fn','tn'))
            prediction_checks.append(dict(sample_id=sid,path=str(path),sha256=sha(path),max_abs_probability_difference=difference,
                                          probabilities_byte_equal=exact, masks_and_GT_equal=True, confusion_equal=same_counts))
        if sid in display_ids:
            raw = sample['image'].x.numpy(); stats = dev.stats
            rgb = (raw*np.array(stats['std'])[:,None,None]+np.array(stats['mean'])[:,None,None])[[2,1,0]]*.0001
            visuals[sid] = dict(rgb=np.clip(rgb.transpose(1,2,0)*3,0,1),valid=valid,target=target,
                                **{key:value[0] for key,value in current.items()})
        if (index+1)%50 == 0:
            write(out/'progress.json',dict(phase='DEV',condition=condition,completed=index+1,total=len(dev)))
            print(f'{condition}: DEV {index+1}/{len(dev)}',flush=True)
    summaries = []
    for part in PARTS:
        pooled_p = np.concatenate(probs[part]); pooled_l = np.concatenate(logits[part]); pooled_t = np.concatenate(targets)
        result = dict(BN=condition,composition=part,pooled={**counts(pooled_p,pooled_t),'AP':ap(pooled_p,pooled_t), 'logit_AP':ap(pooled_l,pooled_t)}, by_event={})
        for event in sorted(set(events)):
            indices = [i for i,e in enumerate(events) if e == event]
            p = np.concatenate([probs[part][i] for i in indices]); t = np.concatenate([targets[i] for i in indices])
            l = np.concatenate([logits[part][i] for i in indices])
            result['by_event'][event] = {**counts(p,t),'AP':ap(p,t),'logit_AP':ap(l,t),'tiles':len(indices)}
        tile_ap = [r['AP'] for r in rows if r['composition']==part and r['AP'] is not None]
        result.update(event_mean_IoU=float(np.mean([v['IoU'] for v in result['by_event'].values()])),
            event_mean_AP=float(np.mean([v['AP'] for v in result['by_event'].values()])),
            per_image_AP=quant(tile_ap), per_image_AP_undefined=669-len(tile_ap))
        summaries.append(result)
    write(out/f'{condition}_results.json',dict(rows=summaries, per_image_AP_invariance=invariance,max_logit_reconstruction_error=max_identity))
    if baseline:
        full = next(x for x in summaries if x['composition']=='FULL')
        differences = dict(event_mean_IoU=full['event_mean_IoU']-historic['event_mean_iou'],
                           pixel_AP=full['pooled']['AP']-historic['pixel_ap'])
        for event,v in full['by_event'].items():
            differences[event+'_IoU'] = v['IoU']-historic['by_event'][event]['pixel']['iou']
            differences[event+'_AP'] = v['AP']-historic['by_event'][event]['pixel_ap']
        passed = all(abs(v)<1e-12 for v in differences.values()) and all(x['confusion_equal'] and x['probabilities_byte_equal'] for x in prediction_checks)
        reproduction = dict(status='PASS' if passed else 'BLOCKED',best_step=4500,metric_differences=differences,
            historical=dict(event_mean_IoU=historic['event_mean_iou'],pooled_AP=historic['pixel_ap']), actual=full,
            n_tiles=669,valid_pixels=full['pooled']['n'], prediction_checks=prediction_checks,
            probabilities_all_byte_equal=all(x['probabilities_byte_equal'] for x in prediction_checks),
            max_abs_probability_difference=max(x['max_abs_probability_difference'] for x in prediction_checks),
            gate='Counts/masks/GT exact; all IoU/AP differences <1e-12; all full-grid probabilities byte-equal')
        write(out/'baseline_reproduction.json',reproduction)
        if not passed: raise RuntimeError('D0 BLOCKED: original baseline does not reproduce; no BN calibration allowed')
    return summaries,rows,visuals,max_identity


def calibrate(original, records, train, out):
    count=0; total=np.zeros(256,np.float64); square=total.copy(); accepted=[]
    quality_native=0; native_positions=0; full_band_pixels=0; any_band_pixels=0; raw_pixels=0
    captured = {}
    handle=original.decoder.batch_norm.register_forward_pre_hook(lambda m,a:captured.__setitem__('input',a[0]))
    try:
        for index,record in enumerate(records):
            observation=train_optical(record,train.stats,train.quality)
            obs=collate_optical([observation]).to(next(original.parameters()).device)
            original(obs)
            x=captured.pop('input').float().cpu().numpy().astype(np.float64)
            assert x.shape == (1,256,32,32) and np.isfinite(x).all()
            values=x.transpose(0,2,3,1).reshape(-1,256)
            count+=len(values); total+=values.sum(0); square+=(values*values).sum(0)
            quality=obs['pixel_valid'].any(1,keepdim=True)
            quality_native+=int(F.interpolate(quality.float(),size=x.shape[-2:],mode='nearest').sum())
            native_positions+=32*32; raw_pixels+=128*128
            any_band_pixels+=int(quality.sum()); full_band_pixels+=int(obs['pixel_valid'].all(1).sum())
            accepted.append(record['sample_id'])
            if (index+1)%100==0:
                write(out/'progress.json',dict(phase='TRAIN_BN_moments',completed=index+1,total=len(records)))
                print(f'TRAIN moments {index+1}/{len(records)}',flush=True)
    finally: handle.remove()
    mean,variance=moments(count,total,square)
    bn=original.decoder.batch_norm
    result=dict(status='PASS', method='FP64 count/sum/sumsq over all native positions; unbiased sample variance (ddof=1)',
        count_per_channel=count, channels=256, samples=len(accepted),accepted_members=accepted,rejected_members=[],
        mean=mean.tolist(),variance=variance.tolist(),sum=total.tolist(),sumsq=square.tolist(),
        original_running_mean=bn.running_mean.cpu().tolist(),original_running_var=bn.running_var.cpu().tolist(),
        variance_ratio=quant(variance/bn.running_var.cpu().numpy()),
        normalized_mean_shift=quant((mean-bn.running_mean.cpu().numpy())/np.sqrt(bn.running_var.cpu().numpy()+bn.eps)),
        native_image_quality_valid_fraction=quality_native/native_positions,
        raw_any_band_valid_fraction=any_band_pixels/raw_pixels,raw_all_band_valid_fraction=full_band_pixels/raw_pixels,
        actual_grid='All 1024 BN input positions per TRAIN tile, including unsupported/partly invalid receptive fields.',
        TRAIN_npz_arrays_decompressed=['x10'],quality_npz_arrays_decompressed=['pixel_valid'],TRAIN_GT_arrays_opened=[],
        finite=True,nonpositive_variance_channels=[],num_batches_tracked_before=int(bn.num_batches_tracked),
        num_batches_tracked_after=int(bn.num_batches_tracked),limitations='Native-grid quality ratio is a center statistic, not a guarantee of clean full receptive fields; invalid regions contribute to the preregistered all-grid moments.')
    write(out/'bn_train_moments.json',result)
    np.savez(out/'BN-R_diagnostic_buffers.npz',running_mean=mean.astype(np.float32),running_var=variance.astype(np.float32))
    return mean,variance


def flatten_result(result):
    row={k:result[k] for k in ('BN','composition','event_mean_IoU','event_mean_AP','per_image_AP_undefined')}
    row['per_image_AP_mean']=result['per_image_AP']['mean']
    for prefix,values in [('pooled',result['pooled']),*result['by_event'].items()]:
        row.update({prefix+'_'+key:value for key,value in values.items()})
    row.update({'BN_R_minus_O_'+k:v for k,v in result['BN_R_minus_O'].items()})
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/use_v4_bn_residual_diagnostics')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'acceptance.json').exists(): raise FileExistsError('Completed evidence exists; choose another --output-dir')
    (out/('dry_run_command.txt' if args.dry_run else 'command.txt')).write_text(subprocess.list2cmdline([sys.executable,*sys.argv]),encoding='utf-8')
    manifest,train,records,dev=prepare(out,args.device)
    if args.dry_run:
        write(out/'dry_run_manifest.json',manifest)
        print('DRY_RUN_PASS TRAIN2223 DEV669, two BN x four compositions, zero parameter updates');return
    torch.set_num_threads(4);torch.manual_seed(1234);np.random.seed(1234)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    started=time.perf_counter()
    # Tiny independent numerical checks, no learned model or BN training mode.
    toy=np.array([[1.,3.],[2.,7.],[4.,9.]])
    m,v=moments(len(toy),toy.sum(0),(toy*toy).sum(0))
    np.testing.assert_allclose(m,toy.mean(0));np.testing.assert_allclose(v,toy.var(0,ddof=1))
    try:
        with analysis_card(args.device) as resource,torch.inference_mode():
            original=Segmenter(run.load_encoder(ENCODER))
            saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
            assert saved['step']==4500 and sha(CHECKPOINT)==read(BASE/'evaluation_best.json')['checkpoint_sha256']
            original.load_state_dict(saved['model'],strict=True);original.requires_grad_(False).eval().to(args.device)
            assert state_sha256(original)==saved['model_sha256']
            initial_state=state_sha256(original);initial_params=parameter_hash(original)
            bn=original.decoder.batch_norm
            manifest['BN']=dict(path='decoder.batch_norm',position='linear_fuse output → BN → activation',
                num_features=bn.num_features,eps=bn.eps,momentum=bn.momentum,track_running_stats=bn.track_running_stats,
                num_batches_tracked=int(bn.num_batches_tracked),training=bn.training,
                running_mean=bn.running_mean.cpu().tolist(),running_var=bn.running_var.cpu().tolist(),
                affine_weight=bn.weight.cpu().tolist(),affine_bias=bn.bias.cpu().tolist())
            manifest['best_step']=saved['step'];write(out/'manifest.json',manifest)
            # Validate the optical-only path against the existing loader on DEV, never open TRAIN GT.
            first=dev[0]['image'];optical=train_optical(dev.records[0],dev.stats,dev.quality)
            for key in ('x','pixel_valid','band_metadata','known_mask','band_valid'): assert torch.equal(getattr(first,key),getattr(optical,key))
            fixed_ids={x['sample_id'] for x in manifest['prior_fixed22']}
            o,per_o,visual_o,error_o=evaluate(original,dev,'BN-O',out,fixed_ids,baseline=True)
            print('D0 baseline gate PASS; starting TRAIN-only BN statistics',flush=True)
            mean,variance=calibrate(original,records,train,out)
            assert state_sha256(original)==initial_state
            recalibrated=copy.deepcopy(original)
            recalibrated.decoder.batch_norm.running_mean.copy_(torch.as_tensor(mean,device=args.device,dtype=torch.float32))
            recalibrated.decoder.batch_norm.running_var.copy_(torch.as_tensor(variance,device=args.device,dtype=torch.float32))
            comparison_before=tensor_comparison(original,recalibrated)
            r,per_r,visual_r,error_r=evaluate(recalibrated,dev,'BN-R',out,fixed_ids)
            comparison_after=tensor_comparison(original,recalibrated)
            assert comparison_before==comparison_after
            assert parameter_hash(original)==parameter_hash(recalibrated)==initial_params
            assert state_sha256(original)==initial_state
            assert all(not p.requires_grad and p.grad is None for model in (original,recalibrated) for p in model.parameters())
            assert all(not module.training for model in (original,recalibrated) for module in model.modules())
            all_rows=per_o+per_r
            lookup={(row['BN'],row['composition'],row['sample_id']):row for row in all_rows}
            for row in all_rows:
                other_o=lookup['BN-O',row['composition'],row['sample_id']];other_r=lookup['BN-R',row['composition'],row['sample_id']]
                for metric in ('AP','IoU','FPR','Recall'):
                    row['BN_R_minus_O_'+metric]=None if other_o[metric] is None else other_r[metric]-other_o[metric]
            results=o+r
            for row in results:
                reference=next(x for x in o if x['composition']==row['composition'])
                row['BN_R_minus_O']={k:row[k]-reference[k] for k in ('event_mean_IoU','event_mean_AP')}
                for group in ('pooled','hiroshima','thrissur'):
                    current=row['pooled'] if group=='pooled' else row['by_event'][group]
                    previous=reference['pooled'] if group=='pooled' else reference['by_event'][group]
                    row['BN_R_minus_O'].update({group+'_'+key:current[key]-previous[key] for key in ('IoU','AP','Precision','Recall','FPR','tp','fp','fn','predicted_positive_fraction')})
            summary=dict(status='COMPLETED',conditions=results,parameter_updates=0,selection='No condition/model/threshold selected',
                max_logit_reconstruction_error=max(error_o,error_r),tensor_comparison=comparison_after,
                mu_quality_support='Any image-quality-valid observed band, independent of GT.',
                interpretation='Post-training buffers and logit interventions are not trained ablations or training-time BN adaptation.')
            csv_write(out/'factorial_results.csv',[flatten_result(x) for x in results])
            csv_write(out/'per_image_diagnostics.csv',all_rows)
            csv_write(out/'fixed22_error_transitions.csv',[row for row in all_rows if row['fixed22']])
            write(out/'summary.json',summary)
            make_figures(out,manifest,results,visual_o,visual_r)
        for path,value in manifest['source_sha256'].items(): assert sha(Path(path))==value,path
        manifest.update(status='COMPLETED',elapsed_s=time.perf_counter()-started,resource=resource,model_original_state_sha256=initial_state,
            model_parameter_sha256=initial_params,tensor_comparison=comparison_after,BN_R_state_sha256=state_sha256(recalibrated),
            original_state_after=state_sha256(original),calibration_input_parity_with_DEV_loader=True)
        write(out/'manifest.json',manifest)
        acceptance=dict(status='PASS_WITH_LIMITATIONS',optimizer_updates=0,backward_calls=0,source_files_and_checkpoint_bytes_unchanged=True,
            all_model_parameters_unchanged=True,only_BN_R_allowed_buffers_changed=comparison_after['changed_buffers'],
            all_other_buffers_equal=True,num_batches_tracked_unchanged=True,baseline_reproduction='PASS',TRAIN_members=2223,DEV_members=669,
            TRAIN_GT_arrays_opened=[],conditions=8,per_image_rows=len(all_rows),fixed22_transition_rows=22*8,
            all_predictions_finite=True,masks_identical=True,mu_no_GT=True,max_logit_reconstruction_error=max(error_o,error_r),
            model_eval_dropout_disabled=True,fp32_tf32_off=True,failures=[],figures=3,
            limits=['Post-training BN replacement is not training-time adaptation.','One seed, reused development events; no blind/generalization claim.',
                    'All-grid BN statistics include invalid receptive fields by design.','Sigmoid saturation/FP32 ties can affect AP; raw-logit AP and exact constant-shift checks are also reported.'])
        write(out/'acceptance.json',acceptance)
        print(json.dumps(dict(status='COMPLETED',elapsed_s=manifest['elapsed_s'],full=[{k:x[k] for k in ('BN','composition','event_mean_IoU','event_mean_AP')} for x in results if x['composition']=='FULL'])),flush=True)
    except BaseException as error:
        write(out/'failure.json',dict(status='BLOCKED',error=repr(error),optimizer_updates=0,elapsed_s=time.perf_counter()-started))
        raise


def make_figures(out,manifest,results,visual_o,visual_r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    moments_data=read(out/'bn_train_moments.json')
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    axes[0].plot(moments_data['original_running_mean'],label='原始BN');axes[0].plot(moments_data['mean'],label='TRAIN全局均值');axes[0].set_title('256通道均值');axes[0].legend()
    axes[1].plot(moments_data['original_running_var'],label='原始BN');axes[1].plot(moments_data['variance'],label='TRAIN无偏方差');axes[1].set_yscale('log');axes[1].set_title('256通道方差（对数轴）');axes[1].legend()
    ratio=np.array(moments_data['variance'])/np.array(moments_data['original_running_var'])
    axes[2].hist(ratio,bins=32);axes[2].set_title('TRAIN方差 / 原运行方差');axes[2].set_xlabel('仅2223块TRAIN，不读标注')
    fig.savefig(out/'bn_statistics.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(17,7),constrained_layout=True)
    for i,event in enumerate(('hiroshima','thrissur')):
        for j,metric in enumerate(('IoU','AP','FPR','Recall')):
            for condition,offset in (('BN-O',-.18),('BN-R',.18)):
                rows=[next(x for x in results if x['BN']==condition and x['composition']==part) for part in PARTS]
                axes[i,j].bar(np.arange(4)+offset,[r['by_event'][event][metric] for r in rows],width=.35,label='原始BN' if condition=='BN-O' else 'TRAIN重校准BN')
            axes[i,j].set_xticks(range(4),PARTS,rotation=25);axes[i,j].set_title(f'{event}｜{metric}');axes[i,j].legend(fontsize=8)
    fig.suptitle('完整DEV669；八个预先固定推理条件；AP为事件内像素池化，阈值指标用0.5')
    fig.savefig(out/'eight_conditions.png',dpi=150);plt.close(fig)
    ids=list(visual_o)
    fig,axes=plt.subplots(len(ids)*2,9,figsize=(23,4.6*len(ids)),constrained_layout=True)
    for i,sid in enumerate(ids):
        for b,(condition,case) in enumerate((('原始BN',visual_o[sid]),('重校准BN',visual_r[sid]))):
            row=i*2+b;valid=case['valid'];target=case['target']
            arrays=[case['rgb']]
            for part in PARTS:
                p=case[part];pred=p>=.5;error=np.zeros((*target.shape,3));error[pred&~target]=[1,0,0];error[~pred&target]=[0,.35,1];error[pred&target]=[0,.8,0]
                arrays.extend([plt.get_cmap('viridis')(p)[...,:3],error])
            for col,array in enumerate(arrays):
                array=array.copy();array[~valid]=.5;axes[row,col].imshow(array);axes[row,col].axis('off')
                if row==0: axes[row,col].set_title(['RGB',*[v for part in PARTS for v in (part+'概率',part+'差错')]][col])
            axes[row,0].text(.01,.99,sid+'\n'+condition,va='top',color='white',fontsize=8,transform=axes[row,0].transAxes,bbox=dict(facecolor='black',alpha=.7,edgecolor='none'))
    fig.suptitle('原固定三例；每例两行对应两种BN；概率0–1紫低黄高；红误报、蓝漏检、绿真前景、灰排除')
    fig.savefig(out/'fixed_cases_decomposition.png',dpi=150);plt.close(fig)


if __name__=='__main__': main()
