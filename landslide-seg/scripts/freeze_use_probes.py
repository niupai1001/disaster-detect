"""Freeze actual, equal-budget linear/shallow probe configurations and initial states."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import yaml
from landslide_seg.config import load_segformer_config,set_seed
from landslide_seg.models.factory import build_model
from landslide_seg.models.frozen_use import state_sha256
from audit_use_independent_data import sha,read
from check_research_protocol import audit_config

OUT=ROOT/'outputs/use_independent_frozen'


def checkpoint(candidate):
    if candidate=='random':
        folder=OUT/'pretraining/physical'
        acceptance=read(OUT/'frozen_contracts_acceptance.json')
        path=folder/'initialization.pt'
        if sha(path)!=acceptance['records']['physical']['initialization_file_sha256']:
            raise ValueError('Random reference initialization changed')
        stage='independent_random'
    else:
        folder=OUT/'pretraining'/candidate
        summary=read(folder/'summary.json')
        if summary.get('complete') is not True or summary.get('updates')!=10000:
            raise ValueError('Independent pretraining has not finished')
        acceptance=read(folder/'acceptance.json')
        if acceptance.get('passed') is not True:
            raise ValueError('Completed pretraining must pass result acceptance before probes')
        path=folder/'best.pt'
        if sha(path)!=summary['best_sha256']:
            raise ValueError('Selected independent USE weight changed')
        stage='independent_ssl'
    return path,read(folder/'resolved_config.json')['use_config'],stage


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True,choices=['random','mean','attention','physical','random_identity','multisensor','spatial'])
    args=parser.parse_args()
    weights,use_config,stage=checkpoint(args.candidate)
    base=yaml.safe_load((ROOT/'configs/segformer/use/physical_mixed.yaml').read_text())
    directory=OUT/'probe_contracts'/args.candidate
    directory.mkdir(parents=True,exist_ok=False)
    config_directory=ROOT/'configs/segformer/use_independent/probes'
    config_directory.mkdir(parents=True,exist_ok=True)
    records={}
    for mode in ('linear','shallow'):
        raw=copy.deepcopy(base)
        s=raw['segformer']
        s.update(experiment=f'use_independent_probe_{mode}_{args.candidate}',output_dir='outputs/use_independent_frozen/probes',research_stage='use_independent')
        s['data']['mixed_bands']=False
        s['model'].update(use_downstream=mode,freeze_use=True,use_stage=stage,use=use_config,
                          use_pretrained=str(weights.relative_to(ROOT)),use_pretrained_sha256=sha(weights),backbone=None,stem=False)
        draft=directory/f'draft_{mode}.yaml'
        draft.write_text(yaml.safe_dump(raw,sort_keys=False,allow_unicode=True),encoding='utf8')
        resolved=load_segformer_config(draft)
        set_seed(1234,True)
        model=build_model(resolved['segformer'])
        identity=state_sha256(model)
        head_identity=state_sha256(model.visual)
        reference_path=OUT/'probe_contracts'/f'head_reference_{mode}.json'
        if reference_path.exists():
            if read(reference_path)['sha256']!=head_identity:
                raise ValueError('Probe head initialization differs between USE candidates')
        else:
            with reference_path.open('x',encoding='utf8') as handle:
                json.dump(dict(sha256=head_identity,seed=1234,mode=mode),handle,indent=2)
        s['train']['initialization_sha256']=identity
        target=config_directory/f'{mode}_{args.candidate}.yaml'
        with target.open('x',encoding='utf8') as handle:
            yaml.safe_dump(raw,handle,sort_keys=False,allow_unicode=True)
        preflight=audit_config(target)
        (directory/f'preflight_{mode}.json').write_text(json.dumps(preflight,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        torch.save(dict(model=model.state_dict(),state_sha256=identity,head_sha256=head_identity,use_sha256=model.use.verify()),directory/f'initialization_{mode}.pt')
        records[mode]=dict(config=str(target.relative_to(ROOT)),config_sha256=sha(target),initialization_sha256=identity,
                           head_sha256=head_identity,use_sha256=model.use.verify(),use_checkpoint_sha256=sha(weights),
                           command=[sys.executable,'-X','utf8','scripts/train_segformer.py','--config',str(target.relative_to(ROOT)),'--device','cuda:0'])
    source_files=[Path(__file__),ROOT/'scripts/check_research_protocol.py',*sorted((ROOT/'src/landslide_seg').rglob('*.py'))]
    proof=dict(passed=True,candidate=args.candidate,seed=1234,updates=2000,schedule_updates=3600,warmup_updates=180,
               batch=8,precision='FP32',selection='DEV event mean IoU',train=2223,dev=669,records=records,
               source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_files})
    for source in source_files:
        target=directory/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        if sha(target)!=proof['source_sha256'][str(source.relative_to(ROOT))]:
            raise ValueError('Probe source snapshot changed during freezing')
    (directory/'acceptance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(records,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
