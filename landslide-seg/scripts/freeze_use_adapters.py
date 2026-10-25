"""Freeze D1-D4 only after independent pretraining and uniform probes are accepted."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
import yaml
from landslide_seg.config import load_segformer_config, set_seed
from landslide_seg.models.factory import build_model
from landslide_seg.models.frozen_use import state_sha256
from audit_use_independent_data import read, sha
from check_research_protocol import audit_config

OUT = ROOT/'outputs/use_independent_frozen'
MODES = ('bridge10', 'rgb3', 'patch32', 'fpn')


def select_common(records):
    if set(records) != {'multisensor', 'spatial'}:
        raise ValueError('Both authorized multi-sensor candidates are required')
    for proof in records.values():
        if proof.get('passed') is not True or not math.isfinite(proof['best_score']):
            raise ValueError('Accepted finite unlabeled scores required')
    # Selection was fixed in the stage ledger before either run; ties favor MBM alone.
    return min(('multisensor', 'spatial'), key=lambda name:records[name]['best_score'])


def prerequisites():
    proofs = {}
    for candidate in ('random', 'mean', 'attention', 'physical', 'random_identity', 'multisensor', 'spatial'):
        for mode in ('linear', 'shallow'):
            folder = OUT/'probes/s2_10band_v4r1_p20'/f'use_independent_probe_{mode}_{candidate}'/'1234'
            acceptance = read(folder/'acceptance.json')
            if acceptance.get('passed') is not True or acceptance['updates'] != 2000 or acceptance['observations'] != 16000:
                raise ValueError('Uniform probe acceptance required: '+candidate+'/'+mode)
            for checkpoint in ('best', 'last'):
                if sha(folder/f'{checkpoint}.pt') != acceptance['checkpoints'][checkpoint]['checkpoint_sha256']:
                    raise ValueError('Accepted probe checkpoint changed')
        if candidate in ('multisensor', 'spatial'):
            folder = OUT/'pretraining'/candidate
            proof = read(folder/'acceptance.json')
            interface = read(folder/'interface_acceptance.json')
            config, summary = read(folder/'resolved_config.json'), read(folder/'summary.json')
            if (proof.get('passed') is not True or interface.get('passed') is not True
                    or not summary['complete'] or summary['updates'] != 10000
                    or config['task_labels'] is not False or config['seed'] != 1234):
                raise ValueError('Independent multi-sensor pretraining and interface acceptance required')
            if sha(folder/'best.pt') != proof['checkpoints']['best']['file_sha256']:
                raise ValueError('Accepted USE checkpoint changed')
            if interface['use_sha256'] != proof['checkpoints']['best']['use_sha256']:
                raise ValueError('Interface acceptance is not for selected USE')
            proofs[candidate] = proof
    reference = read(OUT/'reference_acceptance.json')
    if reference.get('passed') is not True or reference['training_updates_added'] != 0:
        raise ValueError('Reused raw-ten-band reference acceptance required')
    winner = select_common(proofs)
    return winner, proofs


def adapter_config(base, mode, winner, weights, use_config):
    if mode not in MODES:
        raise ValueError('Unsupported downstream adapter')
    raw = copy.deepcopy(base)
    s = raw['segformer']
    s.update(experiment=f'use_independent_adapter_{mode}', output_dir='outputs/use_independent_frozen/adapters',
             research_stage='use_independent')
    s['data']['mixed_bands'] = False
    s['model'].update(type='use', use_downstream=mode, freeze_use=True, use_stage='independent_ssl',
        use=use_config, use_pretrained=str(weights.relative_to(ROOT)), use_pretrained_sha256=sha(weights),
        backbone=None if mode == 'fpn' else 'b0', stem=mode != 'fpn')
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    winner, proofs = prerequisites()
    weights = OUT/'pretraining'/winner/'best.pt'
    use_config = read(OUT/'pretraining'/winner/'resolved_config.json')['use_config']
    directory = OUT/'adapter_contracts'
    directory.mkdir(exist_ok=False)
    config_directory = ROOT/'configs/segformer/use_independent/adapters'
    config_directory.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load((ROOT/'configs/segformer/use/physical_mixed.yaml').read_text())
    records = {}
    for mode in MODES:
        raw = adapter_config(base, mode, winner, weights, use_config)
        draft = directory/f'draft_{mode}.yaml'
        draft.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf8')
        resolved = load_segformer_config(draft)
        set_seed(1234, True)
        model = build_model(resolved['segformer'])
        digest, use_digest = state_sha256(model), model.use.verify()
        if use_digest != proofs[winner]['checkpoints']['best']['use_sha256']:
            raise ValueError('Downstream adapters do not load the same selected USE')
        s = raw['segformer']
        s['train']['initialization_sha256'] = digest
        config = config_directory/f'{mode}.yaml'
        with config.open('x', encoding='utf8') as handle:
            yaml.safe_dump(raw, handle, sort_keys=False, allow_unicode=True)
        preflight = audit_config(config)
        if preflight['counts'] != {'train':2223, 'dev':669}:
            raise ValueError('Actual downstream members differ')
        (directory/f'preflight_{mode}.json').write_text(json.dumps(preflight, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        torch.save(dict(model=model.state_dict(), state_sha256=digest, use_sha256=use_digest), directory/f'initialization_{mode}.pt')
        smoke = copy.deepcopy(raw)
        smoke_s = smoke['segformer']
        smoke_s['experiment'] += '_engineering'
        smoke_s['output_dir'] = 'outputs/use_independent_frozen/engineering_adapters'
        smoke_s['train'].update(max_steps=2, stop_updates=2, eval_every_updates=2, report_steps=[2])
        smoke_path = config_directory/f'smoke_{mode}.yaml'
        with smoke_path.open('x', encoding='utf8') as handle:
            yaml.safe_dump(smoke, handle, sort_keys=False, allow_unicode=True)
        audit_config(smoke_path)
        records[mode] = dict(config=str(config.relative_to(ROOT)), config_sha256=sha(config),
            smoke_config=str(smoke_path.relative_to(ROOT)), smoke_config_sha256=sha(smoke_path),
            initialization_sha256=digest, use_sha256=use_digest, parameters=model.count_params(),
            formal_command=[sys.executable, '-X', 'utf8', 'scripts/train_segformer.py', '--config', str(config.relative_to(ROOT)), '--device', args.device],
            smoke_command=[sys.executable, '-X', 'utf8', 'scripts/train_segformer.py', '--config', str(smoke_path.relative_to(ROOT)), '--device', args.device])
        del model
    source_files = [Path(__file__), ROOT/'scripts/check_research_protocol.py', *sorted((ROOT/'src/landslide_seg').rglob('*.py'))]
    source_hashes = {}
    for source in source_files:
        name = source.relative_to(ROOT)
        target = directory/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[str(name)] = sha(source)
        if sha(target) != source_hashes[str(name)]:
            raise ValueError('Source changed during freeze')
    proof = dict(passed=True, meaning='Configuration and initialization freeze only; formal training additionally requires all four real-data smokes',
        common_use=winner, common_checkpoint_sha256=sha(weights),
        selection='Lowest equal-sensor unlabeled MBM validation MAE between multisensor and spatial; ties multisensor',
        unlabeled_scores={name:p['best_score'] for name,p in proofs.items()}, records=records,
        source_sha256=source_hashes, updates=2000, batch=8, schedule_updates=3600, warmup_updates=180,
        seed=1234, precision='FP32', task_loss='0.5 BCE + 0.5 Dice', downstream_selection='DEV event mean IoU')
    (directory/'acceptance.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(common_use=winner, configurations=len(records), passed=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
