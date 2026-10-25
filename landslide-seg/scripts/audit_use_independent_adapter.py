"""Accept engineering or formal downstream runs against the frozen common USE."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from landslide_seg.config import load_segformer_config
from landslide_seg.models.factory import build_model
from landslide_seg.models.frozen_use import state_sha256
from landslide_seg.training.segformer_trainer import _make_optimizer
from audit_use_independent_data import read,sha
from audit_use_independent_probe import check_history
from check_research_protocol import audit_config
from evaluate_use_probe_subsets import check_report

OUT=ROOT/'outputs/use_independent_frozen'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=['bridge10','rgb3','patch32','fpn','unet'])
    parser.add_argument('--engineering',action='store_true')
    args=parser.parse_args()
    contract_directory=OUT/('unet_contract' if args.mode=='unet' else 'adapter_contracts')
    contract=read(contract_directory/'acceptance.json')
    record=contract['records'][args.mode]
    key='smoke_config' if args.engineering else 'config'
    config_path=ROOT/record[key]
    assert contract['passed'] and sha(config_path)==record[key+'_sha256']
    config=load_segformer_config(config_path);s=config['segformer']
    preflight=audit_config(config_path)
    assert preflight['counts']=={'train':2223,'dev':669}
    frozen_preflight=read(contract_directory/f'preflight_{args.mode}.json')
    assert preflight['files']==frozen_preflight['files']
    assert s['seed']==1234 and not s['train']['amp'] and not s['data']['mixed_bands']
    assert s['model']['freeze_use'] and s['train']['freeze_bn_running_stats']
    assert s['model']['use_downstream']==args.mode
    assert s['model']['use_pretrained_sha256']==contract['common_checkpoint_sha256']
    expected=2 if args.engineering else 2000
    assert s['train']['stop_updates']==expected and s['train']['schedule_updates']==3600
    if not args.engineering:
        gate=read(contract_directory/'engineering_acceptance.json')
        assert gate['passed'] and set(gate['records'])==set(contract['records'])
        for mode,evidence in gate['records'].items():
            assert sha(ROOT/evidence['path'])==evidence['sha256']
    folder=ROOT/s['output_dir']/s['data']['version']/s['experiment']/'1234'
    target=folder/'acceptance.json'
    if target.exists():raise FileExistsError('Preserve completed acceptance')
    assert read(folder/'resolved_config.json')==config
    summary,meta=read(folder/'train_summary.json'),read(folder/'run_meta.json')
    canonical=read(OUT/'probes/s2_10band_v4r1_p20/use_independent_probe_linear_random/1234/run_meta.json')
    assert meta['train_samples']==canonical['train_samples'] and meta['val_samples']==canonical['val_samples']
    assert summary['complete'] and summary['global_step']==expected
    assert (summary['n_train'],summary['n_val'],summary['samples_exposed'],summary['skipped_batches'])==(2223,669,expected*8,0)
    assert meta['seed_info']['tf32'] is False
    if args.mode in ('fpn','unet'):assert meta['weight'] is None
    for name,digest in contract['source_sha256'].items():
        assert sha(contract_directory/'source_snapshot'/name)==digest
        if name.replace('\\','/').startswith('src/'):assert sha(ROOT/name)==digest
    history=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    if args.engineering:
        assert len(history)==1 and history[0]['global_step']==2
        best=history[0]
    else:best=check_history(history,record['use_sha256'])
    for row in history:
        integrity=row['probe_integrity']
        assert integrity['use_sha256']==record['use_sha256'] and integrity['encoder_unchanged']
        assert not integrity['encoder_has_grad'] and not integrity['encoder_training']
        events=row['dev']['by_event'];assert set(events)=={'hiroshima','thrissur'}
        assert math.isfinite(row['val_iou']) and abs(row['val_iou']-sum(v['iou'] for v in events.values())/2)<1e-12
    assert summary['best_step']==best['global_step'] and summary['best_val_iou']==best['val_iou']
    trace=[json.loads(line) for line in (folder/'per_update.jsonl').read_text().splitlines()]
    assert len(trace)==expected
    for step,row in enumerate(trace,1):
        assert row['global_step']==step and row['batch_size']==8 and math.isfinite(row['train_loss'])
        factor=(step+1)/180 if step<180 else 1-(step-180)/3420
        expected_lrs=[s['optim']['encoder_lr']*factor,s['optim']['head_lr']*factor]
        assert len(row['learning_rates'])==len(expected_lrs)
        assert all(abs(a-b)<1e-12 for a,b in zip(row['learning_rates'],expected_lrs))
    exposures=read(folder/'sample_exposures.json')
    assert set(exposures)==set(meta['train_samples']) and len(exposures)==2223 and sum(exposures.values())==expected*8
    model=build_model(s)
    initial=torch.load(folder/'initialization.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(initial['model'],strict=True)
    assert state_sha256(model)==record['initialization_sha256']
    assert initial['identity']==meta['initialization']
    assert model.use.verify()==record['use_sha256']
    optimizer=_make_optimizer(model,config);model.use.verify(optimizer)
    assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for g in optimizer.param_groups for p in g['params']}
    assert not any(n.startswith('use.') for n in meta['initialization']['trainable_names'])
    initial_bn={n:b.clone() for n,b in model.named_buffers() if n.endswith(('running_mean','running_var','num_batches_tracked'))}
    checkpoints={}
    for name in ('best','last'):
        path=folder/f'{name}.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
        step=best['global_step'] if name=='best' else expected
        assert saved['config']==config and saved['global_step']==step and saved['skipped_total']==0
        model.load_state_dict(saved['model'],strict=True)
        assert model.use.verify(optimizer)==record['use_sha256']
        assert all(torch.equal(b,initial_bn[n]) for n,b in model.named_buffers() if n in initial_bn)
        optimizer.load_state_dict(saved['optimizer']);model.use.verify(optimizer)
        assert sum(saved['stream_state']['exposures'])==step*8
        assert all(k in saved for k in ('rng','scheduler','history'))
        if name=='last':assert saved['stream_state']['exposures']==[exposures[sid] for sid in meta['train_samples']]
        checkpoints[name]=dict(step=step,checkpoint_sha256=sha(path),use_sha256=model.use.verify())
        if not args.engineering:
            evaluation_path=folder/f'eval/eval_{name}_val_full.json';evaluation=read(evaluation_path)
            score=check_report(evaluation,'full',sha(path),sha(ROOT/s['data']['manifest']),sha(ROOT/s['data']['stats']))
            expected_score=next(row['val_iou'] for row in history if row['global_step']==step)
            assert abs(score-expected_score)<=1e-5
            checkpoints[name].update(event_mean_iou=score,evaluation_sha256=sha(evaluation_path))
    proof=dict(passed=True,mode=args.mode,engineering_only=args.engineering,updates=expected,observations=expected*8,
        checkpoints=checkpoints,common_use=contract['common_use'],config_sha256=sha(config_path),
        initialization_sha256=record['initialization_sha256'],use_sha256=record['use_sha256'],
        auditor_sha256=sha(Path(__file__)),helper_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [ROOT/'scripts/audit_use_independent_probe.py',ROOT/'scripts/evaluate_use_probe_subsets.py']},
        limitations=['Protocol, initialization and immutable USE acceptance; does not imply improved task performance'])
    target.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(proof,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
