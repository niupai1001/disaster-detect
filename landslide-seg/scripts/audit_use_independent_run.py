"""Verify completed independent pretraining: budget, exposure, selection and strict reload."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from pretrain_use_independent import initialize, learning_rate, OUT
from audit_use_independent_data import sha, read
from landslide_seg.models.frozen_use import state_sha256


def audit_trace(path, config, members):
    expected={row['file']:row for row in members if row['role']=='TRAIN' and row['sensor'] in config['train_ids']}
    exposures=Counter();sensors=Counter();combinations=Counter();steps=0
    with path.open(encoding='utf8') as handle:
        for steps,line in enumerate(handle,1):
            row=json.loads(line)
            if row['step']!=steps or len(row['exposure'])!=config['batch']:
                raise ValueError('Nonsequential step or wrong observation budget')
            if not math.isfinite(row['loss']) or abs(row['lr']-learning_rate(steps))>1e-12:
                raise ValueError('Nonfinite loss or wrong LR schedule')
            for item in row['exposure']:
                if item['file'] not in expected or expected[item['file']]['sensor']!=item['sensor']:
                    raise ValueError('Exposure outside frozen TRAIN members')
                source=expected[item['file']]
                visible,target=item['visible_bands'],item['target_bands']
                if set(visible)&set(target) or len(set(visible))+len(set(target))!=len(source['band_ids']) or set(visible)|set(target)!=set(source['band_ids']):
                    raise ValueError('Hidden target/input overlap or missing band')
                exposures[item['file']]+=1;sensors[item['sensor']]+=1
                combinations[(item['sensor'],tuple(sorted(visible)))]+=1
    if steps!=config['updates'] or sum(exposures.values())!=config['updates']*config['batch']:
        raise ValueError('Incomplete independent pretraining budget')
    return dict(updates=steps,observations=sum(exposures.values()),by_sensor=dict(sensors),unique_locations=len(exposures),
                location_exposures=dict(exposures),visible_combination_counts=[dict(sensor=s,bands=list(b),exposures=n) for (s,b),n in sorted(combinations.items())],
                extra_spatial_views=steps*config['batch'] if config.get('spatial_objective') else 0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',required=True)
    args=parser.parse_args();folder=OUT/'pretraining'/args.variant
    config,summary=read(folder/'resolved_config.json'),read(folder/'summary.json')
    if summary.get('complete') is not True or summary.get('config_sha256')!=sha(folder/'resolved_config.json'):
        raise ValueError('Training completion/configuration mismatch')
    if config.get('task_labels') is not False or config['updates']!=10000 or config['seed']!=1234:
        raise ValueError('Independent training contract differs')
    for name,expected in config['source_sha256'].items():
        if sha(folder/'source_snapshot'/name)!=expected:
            raise ValueError('Frozen source snapshot changed')
    members=[json.loads(line) for line in (OUT/'cache_audit_v1/members.jsonl').read_text(encoding='utf8').splitlines()]
    if sha(OUT/'cache_audit_v1/members.jsonl')!=config['members_sha256']:
        raise ValueError('Frozen members changed')
    exposure=audit_trace(folder/'sampling.jsonl',config,members)
    history=read(folder/'history.json')
    if [row['step'] for row in history]!=list(range(500,10001,500)):
        raise ValueError('Missing validation/curve checkpoints')
    for row in history:
        sensor_mean=sum(v['standardized_mae'] for v in row['metrics'].values())/len(row['metrics'])
        if not math.isfinite(row['score']) or abs(sensor_mean-row['score'])>1e-10:
            raise ValueError('Unlabeled equal-sensor selection mismatch')
    expected_best=min(history,key=lambda r:r['score'])
    if expected_best['score']!=summary['best_score']:
        raise ValueError('Best weight selection mismatch')
    result={}
    for name in ('best','last'):
        path=folder/f'{name}.pt'
        if sha(path)!=summary[f'{name}_sha256']:
            raise ValueError('Checkpoint file hash mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['config']!=config or saved['step']!=(expected_best['step'] if name=='best' else 10000):
            raise ValueError('Checkpoint configuration or step mismatch')
        model=initialize(args.variant,'cpu');model.load_state_dict(saved['model'],strict=True)
        if set(saved['use'])!=set(model.use.state_dict()) or any(not torch.equal(value,model.use.state_dict()[key]) for key,value in saved['use'].items()):
            raise ValueError('Transferred USE differs from saved model state')
        if not all(key in saved for key in ('optimizer','rng','python_rng','torch_rng','cuda_rng','history')):
            raise ValueError('Incomplete recovery state')
        result[name]=dict(file_sha256=sha(path),use_sha256=state_sha256(model.use),step=saved['step'])
    (folder/'exposure_acceptance.json').write_text(json.dumps(exposure,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    proof=dict(passed=True,variant=args.variant,best_score=summary['best_score'],checkpoints=result,
               config_sha256=summary['config_sha256'],members_sha256=config['members_sha256'],
               exposure_sha256=sha(folder/'exposure_acceptance.json'),auditor_sha256=sha(Path(__file__)),
               limitations=['Self-supervised validation is not downstream migration evidence',
                            'Random MBM views may include the five-band evaluation combination; it is not held out from pretraining'])
    (folder/'acceptance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
