"""Freeze the user-added U-Net control after the original source-frozen chain."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import yaml
from landslide_seg.config import load_segformer_config,set_seed
from landslide_seg.models.factory import build_model
from landslide_seg.models.frozen_use import state_sha256
from audit_use_independent_data import read,sha
from check_research_protocol import audit_config

OUT=ROOT/'outputs/use_independent_frozen'

def require_finished_original_pipeline():
    state=read(OUT/'common_use_adapter_status.json')
    if state.get('status')!='finished_requires_final_study_publication_and_acceptance':
        raise ValueError('Original source-frozen adapters must finish before U-Net integration')
    if any(row.get('status')!='finished' or row.get('exit_code')!=0 for row in state['jobs']):
        raise ValueError('Original adapter commands are unresolved')
    for mode in ('bridge10','rgb3','patch32','fpn'):
        folder=OUT/'adapters/s2_10band_v4r1_p20'/f'use_independent_adapter_{mode}'/'1234'
        accepted=read(folder/'acceptance.json');subset=read(folder/'subset_acceptance.json')
        if not accepted['passed'] or not subset['passed']:raise ValueError('Original adapter acceptance missing')
        for name in ('best','last'):
            if sha(folder/f'{name}.pt')!=accepted['checkpoints'][name]['checkpoint_sha256']:
                raise ValueError('Original accepted adapter changed')

def source_for():
    original=read(OUT/'adapter_contracts/acceptance.json')
    if original.get('passed') is not True:raise ValueError('Common USE not frozen')
    candidate=original['common_use']
    config=read(OUT/'pretraining'/candidate/'resolved_config.json')['use_config']
    weights=OUT/'pretraining'/candidate/'best.pt'
    if sha(weights)!=original['common_checkpoint_sha256']:raise ValueError('Common USE changed')
    return candidate,weights,config,'independent_ssl',dict(rule=original['selection'],
        basis='unlabeled_validation',source='outputs/use_independent_frozen/adapter_contracts/acceptance.json',
        source_sha256=sha(OUT/'adapter_contracts/acceptance.json'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    require_finished_original_pipeline()
    from landslide_seg.models.use_downstream import MODES
    if 'unet' not in MODES:raise ValueError('U-Net source integration is not active; preserve old contracts')
    candidate,weights,use_config,use_stage,selection=source_for()
    directory=OUT/'unet_contract';directory.mkdir(exist_ok=False)
    config_directory=ROOT/'configs/segformer/use_independent/unet';config_directory.mkdir(exist_ok=False)
    raw=yaml.safe_load((ROOT/'configs/segformer/use/physical_mixed.yaml').read_text())
    s=raw['segformer'];s.update(experiment='use_independent_unet',output_dir='outputs/use_independent_frozen/unet_control',research_stage='use_independent')
    s['data']['mixed_bands']=False
    s['model'].update(type='use',use_downstream='unet',freeze_use=True,use_stage=use_stage,use=use_config,
        use_pretrained=str(weights.relative_to(ROOT)),use_pretrained_sha256=sha(weights),backbone=None,stem=False)
    draft=directory/'draft.yaml';draft.write_text(yaml.safe_dump(raw,sort_keys=False),encoding='utf8')
    resolved=load_segformer_config(draft);set_seed(1234,True);model=build_model(resolved['segformer'])
    digest=state_sha256(model);use_digest=model.use.verify()
    s['train']['initialization_sha256']=digest
    config=config_directory/'unet.yaml'
    with config.open('x',encoding='utf8') as f:yaml.safe_dump(raw,f,sort_keys=False)
    preflight=audit_config(config)
    if preflight['counts']!={'train':2223,'dev':669}:raise ValueError('U-Net members differ')
    (directory/'preflight_unet.json').write_text(json.dumps(preflight,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (directory/'initialization_unet.pt').open('xb') as f:
        torch.save(dict(model=model.state_dict(),state_sha256=digest,use_sha256=use_digest),f)
    smoke=copy.deepcopy(raw);smoke_s=smoke['segformer']
    smoke_s['experiment']='use_independent_unet_engineering';smoke_s['output_dir']='outputs/use_independent_frozen/unet_engineering'
    smoke_s['train'].update(max_steps=2,stop_updates=2,eval_every_updates=2,report_steps=[2])
    smoke_path=config_directory/'smoke_unet.yaml'
    with smoke_path.open('x',encoding='utf8') as f:yaml.safe_dump(smoke,f,sort_keys=False)
    audit_config(smoke_path)
    record=dict(config=str(config.relative_to(ROOT)),config_sha256=sha(config),
        smoke_config=str(smoke_path.relative_to(ROOT)),smoke_config_sha256=sha(smoke_path),
        initialization_sha256=digest,use_sha256=use_digest,parameters=model.count_params(),
        formal_command=[sys.executable,'-X','utf8','scripts/train_segformer.py','--config',str(config.relative_to(ROOT)),'--device',args.device],
        smoke_command=[sys.executable,'-X','utf8','scripts/train_segformer.py','--config',str(smoke_path.relative_to(ROOT)),'--device',args.device])
    source_files=[Path(__file__),ROOT/'scripts/check_research_protocol.py',*sorted((ROOT/'src/landslide_seg').rglob('*.py'))]
    source_hashes={}
    for source in source_files:
        name=source.relative_to(ROOT);target=directory/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        source_hashes[str(name)]=sha(source)
        if sha(target)!=source_hashes[str(name)]:raise ValueError('Source changed while freezing')
    proof=dict(passed=True,meaning='Configuration freeze only; real-data engineering must pass before fresh formal training',
        common_use=candidate,common_checkpoint_sha256=sha(weights),selection=selection,records=dict(unet=record),source_sha256=source_hashes,
        updates=2000,batch=8,precision='FP32',seed=1234,schedule_updates=3600,warmup_updates=180,
        task_learning_rate=0.0006,task_loss='0.5 BCE + 0.5 Dice',downstream_selection='DEV event mean IoU',
        architecture='Frozen USE32 -> four-level randomly initialized U-Net; GroupNorm, skip concatenation')
    (directory/'acceptance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,candidate=candidate,selection='unlabeled_common_use',config=str(config)),ensure_ascii=False))

if __name__=='__main__':main()
