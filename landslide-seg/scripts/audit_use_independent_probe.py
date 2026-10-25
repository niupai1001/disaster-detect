"""Accept uniform frozen probes without changing weights or selection rules."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
from landslide_seg.config import load_segformer_config
from landslide_seg.models.factory import build_model
from landslide_seg.models.frozen_use import state_sha256
from landslide_seg.training.segformer_trainer import _make_optimizer
from audit_use_independent_data import read, sha
from check_research_protocol import audit_config

OUT = ROOT/'outputs/use_independent_frozen'


def check_history(history, use_digest):
    if [r['global_step'] for r in history] != list(range(100, 2001, 100)):
        raise ValueError('Missing uniform probe validation points')
    for row in history:
        integrity = row['probe_integrity']
        if (integrity['use_sha256'] != use_digest or not integrity['encoder_unchanged']
                or integrity['encoder_has_grad'] or integrity['encoder_training']):
            raise ValueError('Frozen USE integrity violation')
        events = row['dev']['by_event']
        if set(events) != {'hiroshima', 'thrissur'}:
            raise ValueError('Wrong development events')
        expected = sum(r['iou'] for r in events.values())/2
        if not math.isfinite(row['val_iou']) or abs(row['val_iou']-expected) > 1e-12:
            raise ValueError('Selection is not event mean IoU')
        if row['samples_exposed'] != row['global_step']*8 or row['skipped_batches'] != 0:
            raise ValueError('Incorrect observation budget')
    # Python max keeps the first occurrence, matching strictly-greater trainer selection.
    return max(history, key=lambda row: row['val_iou'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--mode', required=True, choices=['linear', 'shallow'])
    args = parser.parse_args()
    contract = read(OUT/'probe_contracts'/args.candidate/'acceptance.json')
    record = contract['records'][args.mode]
    config_path = ROOT/record['config']
    if contract['passed'] is not True or sha(config_path) != record['config_sha256']:
        raise ValueError('Frozen actual configuration changed')
    audit_config(str(config_path))
    config = load_segformer_config(config_path)
    s = config['segformer']
    folder = OUT/'probes/s2_10band_v4r1_p20'/s['experiment']/'1234'
    target = folder/'acceptance.json'
    if target.exists():
        raise FileExistsError('Preserve completed probe acceptance: '+str(target))
    if read(folder/'resolved_config.json') != config:
        raise ValueError('Actual resolved configuration differs from frozen configuration')
    summary, meta = read(folder/'train_summary.json'), read(folder/'run_meta.json')
    if (summary['complete'] is not True or summary['global_step'] != 2000 or summary['n_train'] != 2223
            or summary['n_val'] != 669 or summary['samples_exposed'] != 16000 or summary['skipped_batches'] != 0):
        raise ValueError('Incomplete uniform probe training budget')
    if s['seed'] != 1234 or s['data']['mixed_bands'] or s['train']['amp'] or not s['model']['freeze_use']:
        raise ValueError('Uniform probe protocol differs')
    if s['train']['schedule_updates'] != 3600 or s['train']['stop_updates'] != 2000:
        raise ValueError('Probe schedule differs')
    if meta['seed_info'].get('tf32') is not False or meta['weight'] is not None:
        raise ValueError('Probe precision or pretrained visual weight provenance differs')
    for name, digest in contract['source_sha256'].items():
        snapshot = OUT/'probe_contracts'/args.candidate/'source_snapshot'/name
        if sha(snapshot) != digest:
            raise ValueError('Frozen source snapshot differs: '+name)
    history = [json.loads(line) for line in (folder/'metrics.jsonl').read_text(encoding='utf8').splitlines()]
    best = check_history(history, record['use_sha256'])
    if best['global_step'] != summary['best_step'] or best['val_iou'] != summary['best_val_iou']:
        raise ValueError('Best checkpoint selection mismatch')
    trace = [json.loads(line) for line in (folder/'per_update.jsonl').read_text().splitlines()]
    if len(trace) != 2000:
        raise ValueError('Missing per-update curve')
    for step, row in enumerate(trace, 1):
        factor = (step+1)/180 if step < 180 else 1-(step-180)/3420
        expected = [s['optim']['encoder_lr']*factor, s['optim']['head_lr']*factor]
        if row['global_step'] != step or row['batch_size'] != 8 or not math.isfinite(row['train_loss']):
            raise ValueError('Wrong update trace')
        if any(abs(a-b) > 1e-12 for a, b in zip(row['learning_rates'], expected)):
            raise ValueError('Wrong learning rate schedule')
    exposures = read(folder/'sample_exposures.json')
    if set(exposures) != set(meta['train_samples']) or len(exposures) != 2223 or sum(exposures.values()) != 16000:
        raise ValueError('Wrong frozen TRAIN exposure members')
    model = build_model(s)
    initial = torch.load(folder/'initialization.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(initial['model'], strict=True)
    if state_sha256(model) != record['initialization_sha256'] or state_sha256(model.visual) != record['head_sha256']:
        raise ValueError('Formal fresh initialization differs from pinned initialization')
    if initial['identity'] != meta['initialization']:
        raise ValueError('Formal initialization provenance differs')
    model.use.verify()
    optimizer = _make_optimizer(model, config)
    model.use.verify(optimizer)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    grouped = {id(p) for group in optimizer.param_groups for p in group['params']}
    if trainable != grouped or any(name.startswith('use.') for name in meta['initialization']['trainable_names']):
        raise ValueError('USE leaked into optimizer')
    results = {}
    for name in ('best', 'last'):
        path = folder/f'{name}.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        step = best['global_step'] if name == 'best' else 2000
        if saved['config'] != config or saved['global_step'] != step or saved['skipped_total'] != 0:
            raise ValueError('Checkpoint identity/step differs')
        model.load_state_dict(saved['model'], strict=True)
        if model.use.verify(optimizer) != record['use_sha256']:
            raise ValueError('Checkpoint USE changed')
        optimizer.load_state_dict(saved['optimizer'])
        model.use.verify(optimizer)
        if sum(saved['stream_state']['exposures']) != step*8 or not all(k in saved for k in ('rng', 'scheduler', 'history')):
            raise ValueError('Incomplete recovery state or sampling budget')
        if name == 'last' and saved['stream_state']['exposures'] != [exposures[sid] for sid in meta['train_samples']]:
            raise ValueError('Checkpoint exposure counts differ')
        evaluation = read(folder/f'eval/eval_{name}_val_full.json')
        if (evaluation['checkpoint_sha256'] != sha(path) or evaluation['n_samples'] != 669
                or evaluation['stats_sha256'] != sha(ROOT/s['data']['stats'])
                or evaluation['manifest_sha256'] != sha(ROOT/s['data']['manifest'])):
            raise ValueError('Independent evaluation provenance differs')
        if {event: value['n_tiles'] for event, value in evaluation['by_event'].items()} != {'hiroshima':448, 'thrissur':221}:
            raise ValueError('Evaluation development members differ')
        expected = next(r['val_iou'] for r in history if r['global_step'] == step)
        if abs(evaluation['event_mean_iou']-expected) > 1e-5:
            raise ValueError('Independent inference differs from training validation')
        results[name] = dict(step=step, checkpoint_sha256=sha(path), use_sha256=model.use.verify(),
                             event_mean_iou=evaluation['event_mean_iou'], evaluation_sha256=sha(folder/f'eval/eval_{name}_val_full.json'))
    proof = dict(passed=True, candidate=args.candidate, mode=args.mode, updates=2000, observations=16000,
                 checkpoints=results, initialization_sha256=record['initialization_sha256'], head_sha256=record['head_sha256'],
                 config_sha256=record['config_sha256'], auditor_sha256=sha(Path(__file__)),
                 limitations=['Acceptance confirms protocol and frozen state, not a performance improvement',
                              'One seed and one Sentinel-2 task; no general transfer claim'])
    target.write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
