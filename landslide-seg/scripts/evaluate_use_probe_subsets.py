"""Evaluate all input subsets with the same accepted full-input-selected probe weights."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from audit_use_independent_data import read, sha

OUT = ROOT/'outputs/use_independent_frozen'
SUBSETS = ('full','rgb','rgb_nir','six','unseen5')


def evaluation_code(device):
    code="from landslide_seg.config import set_seed; set_seed(1234,True); import runpy; runpy.run_path('scripts/evaluate_segformer.py',run_name='__main__')"
    if device=='cpu':code='import torch; torch.set_num_threads(4); '+code
    return code


def check_report(report, subset, checkpoint_hash, manifest_hash, stats_hash):
    if (report['band_subset'] != subset or report['checkpoint_sha256'] != checkpoint_hash
            or report['manifest_sha256'] != manifest_hash or report['stats_sha256'] != stats_hash):
        raise ValueError('Subset evaluation provenance differs')
    if report['n_samples'] != 669 or report['role'] != 'DEV' or report['threshold'] != .5:
        raise ValueError('Evaluation members/role/threshold differ')
    if {event:value['n_tiles'] for event,value in report['by_event'].items()} != {'hiroshima':448,'thrissur':221}:
        raise ValueError('Evaluation event members differ')
    expected = sum(value['pixel']['iou'] for value in report['by_event'].values())/2
    if abs(expected-report['event_mean_iou']) > 1e-12:
        raise ValueError('Evaluation is not event mean IoU')
    return report['event_mean_iou']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True,choices=['random','mean','attention','physical','random_identity','multisensor','spatial'])
    parser.add_argument('--mode',required=True,choices=['linear','shallow'])
    parser.add_argument('--device',default='cuda:1')
    args = parser.parse_args()
    folder = OUT/'probes/s2_10band_v4r1_p20'/f'use_independent_probe_{args.mode}_{args.candidate}'/'1234'
    target = folder/'subset_acceptance.json'
    if target.exists():raise FileExistsError('Preserve existing subset evaluation acceptance')
    accepted = read(folder/'acceptance.json')
    config = read(folder/'resolved_config.json')['segformer']
    contract = read(OUT/'probe_contracts'/args.candidate/'acceptance.json')
    if accepted.get('passed') is not True or sha(folder/'best.pt') != accepted['checkpoints']['best']['checkpoint_sha256']:
        raise ValueError('Accepted best full-input-selected task weight required')
    for name,digest in contract['source_sha256'].items():
        if name.replace('\\','/').startswith('src/') and sha(ROOT/name) != digest:
            raise ValueError('Frozen inference source changed')
    weights = folder/'best.pt'
    checkpoint_hash = sha(weights)
    manifest_hash,stats_hash = sha(ROOT/config['data']['manifest']),sha(ROOT/config['data']['stats'])
    records = {}
    for subset in SUBSETS:
        tag = 'best_val_'+subset
        report_path = folder/'eval'/f'eval_{tag}.json'
        command = None
        if subset != 'full':
            if report_path.exists():raise FileExistsError('Preserve existing partial evaluation: '+str(report_path))
            code = evaluation_code(args.device)
            command = [sys.executable,'-X','utf8','-c',code,'--checkpoint',str(weights),'--split','val',
                       '--device',args.device,'--skip-efficiency','--band-subset',subset,'--tag',tag]
            log_path = folder/f'evaluate_{tag}.log'
            with log_path.open('x',encoding='utf8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report = read(report_path)
        score = check_report(report,subset,checkpoint_hash,manifest_hash,stats_hash)
        for prefix,suffix in (('per_tile_','.csv'),('objects_','.json')):
            if not (folder/'eval'/f'{prefix}{tag}{suffix}').exists():
                raise ValueError('Missing per-tile or object evaluation')
        records[subset] = dict(event_mean_iou=score,evaluation_sha256=sha(report_path),command=command,
            by_event={event:value['pixel']['iou'] for event,value in report['by_event'].items()})
        if sha(weights) != checkpoint_hash:raise ValueError('Probe weights changed during subset inference')
        print(args.candidate,args.mode,subset,score,flush=True)
    proof = dict(passed=True,candidate=args.candidate,mode=args.mode,checkpoint_sha256=checkpoint_hash,
        use_sha256=accepted['checkpoints']['best']['use_sha256'],training_updates_added=0,
        inference_device=args.device,cpu_threads=4 if args.device=='cpu' else None,
        checkpoint_selection='Reuse best event-mean IoU on full ten-band DEV; no subset-specific weight selection',
        records=records,auditor_sha256=sha(Path(__file__)),
        limitations=['Five-band combination may have occurred during random MBM masking; not SSL-unseen',
                     'All views are Sentinel-2; these are not real RGB or Landsat downstream tasks',
                     'One seed, repeatedly used development events; not blind geographic generalization'])
    target.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Subset evaluation accepted:',args.candidate,args.mode,flush=True)


if __name__ == '__main__':main()
