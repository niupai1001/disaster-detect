"""Evaluate fixed adapter weights on the existing Sentinel-2 band views."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from audit_use_independent_data import read,sha
from evaluate_use_probe_subsets import SUBSETS,check_report,evaluation_code

OUT=ROOT/'outputs/use_independent_frozen'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=['bridge10','rgb3','patch32','fpn','unet'])
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    contract_directory=OUT/('unet_contract' if args.mode=='unet' else 'adapter_contracts')
    contract=read(contract_directory/'acceptance.json');record=contract['records'][args.mode]
    folder=(OUT/'unet_control/s2_10band_v4r1_p20/use_independent_unet/1234' if args.mode=='unet'
        else OUT/'adapters/s2_10band_v4r1_p20'/f'use_independent_adapter_{args.mode}'/'1234')
    accepted=read(folder/'acceptance.json')
    assert accepted['passed'] and not accepted['engineering_only']
    assert accepted['use_sha256']==record['use_sha256']
    target=folder/'subset_acceptance.json'
    if target.exists():raise FileExistsError('Preserve accepted subset evaluation')
    weights=folder/'best.pt';checkpoint_hash=sha(weights)
    assert checkpoint_hash==accepted['checkpoints']['best']['checkpoint_sha256']
    config=read(folder/'resolved_config.json')['segformer']
    manifest_hash,stats_hash=sha(ROOT/config['data']['manifest']),sha(ROOT/config['data']['stats'])
    for name,digest in contract['source_sha256'].items():
        if name.replace('\\','/').startswith('src/'):assert sha(ROOT/name)==digest
    records={}
    for subset in SUBSETS:
        tag='best_val_'+subset;report_path=folder/'eval'/f'eval_{tag}.json';command=None
        if subset!='full':
            if report_path.exists():raise FileExistsError('Preserve partial evaluation')
            command=[sys.executable,'-X','utf8','-c',evaluation_code(args.device),'--checkpoint',str(weights),
                '--split','val','--device',args.device,'--skip-efficiency','--band-subset',subset,'--tag',tag]
            with (folder/f'evaluate_{tag}.log').open('x',encoding='utf8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=read(report_path)
        score=check_report(report,subset,checkpoint_hash,manifest_hash,stats_hash)
        for prefix,suffix in (('per_tile_','.csv'),('objects_','.json')):
            assert (folder/'eval'/f'{prefix}{tag}{suffix}').exists()
        records[subset]=dict(event_mean_iou=score,evaluation_sha256=sha(report_path),command=command,
            by_event={event:value['pixel']['iou'] for event,value in report['by_event'].items()})
        assert sha(weights)==checkpoint_hash
        print(args.mode,subset,score,flush=True)
    proof=dict(passed=True,mode=args.mode,checkpoint_sha256=checkpoint_hash,use_sha256=record['use_sha256'],
        common_use=contract['common_use'],training_updates_added=0,inference_device=args.device,
        checkpoint_selection='Reuse full ten-band DEV best; no subset-specific weight selection',records=records,
        auditor_sha256=sha(Path(__file__)),helper_sha256=sha(ROOT/'scripts/evaluate_use_probe_subsets.py'),
        limitations=['Sentinel-2 band subsets, not additional real RGB or Landsat tasks',
            'Five-band combination may have appeared during MBM masking; not claimed unseen',
            'Single seed and reused development events; no blind geographic generalization claim'])
    target.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':main()
