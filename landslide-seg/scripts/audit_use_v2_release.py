"""Independent C audit of the USE v2 engineering release.

Read-only towards the release: every check is recomputed from code, caches and
checkpoint bytes rather than trusting the producer's `passed` fields. Only
`outputs/use_v2/audit` is written, and the training marker appears only when
every check passes.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_use_v2 as runtime
from pretrain_use_v2 import (initialize, initialize_pair, RecoveryProbe, load_rows, sample_rows,
                            task_view, ssl_update, masked_mae, lr, deterministic)
from audit_use_independent_data import members, audit_cache, EXTERNAL, sha, read
from evaluate_use_v2 import calibrate, area30
from landslide_seg.models.use_v2 import USEv2
from landslide_seg.models.frozen_use import FrozenUSE, state_sha256

RELEASE = ROOT / 'outputs/use_v2/releases/engineering_v1'
AUDIT = ROOT / 'outputs/use_v2/audit'
MARKER = AUDIT / 'READY_FOR_TRAINING.json'
SUITE = ROOT / 'outputs/use_v2/engineering/suite_v5'
ACCEPTANCE = ROOT / 'outputs/use_v2/engineering/acceptance_v2.json'
DEVICE = 'cpu'
TASKS = ('spectral', 'fine', 'context')


def observation(c=3, h=16, w=16):
    return dict(x=torch.randn(2, c, h, w), pixel_valid=torch.ones(2, c, h, w, dtype=torch.bool))


def check_release():
    digest = runtime.verify_release(RELEASE)
    manifest = read(RELEASE / 'manifest.json')
    assert manifest['status'] == 'READY_FOR_AUDIT' and manifest['formal_updates'] == 0
    # A new release may not have trained anything itself; runs from an accepted earlier release stay untouched.
    started = [path.parent.name for path in (ROOT / 'outputs/use_v2/runs').glob('*/resolved_config.json')
               if read(path)['identity'].get('release') == digest]
    assert not started, started
    assert len(manifest['commands']) == 30
    if 'allocation' in manifest:
        plan = manifest['allocation']
        assert set(plan) == set(manifest['commands']) == set(runtime.matrix())
        for job, argv in manifest['commands'].items():
            assert argv[-2] == '--device' and argv[-1] == plan[job], (job, argv[-2:])
        assert {plan[j] for j in plan} == {'cuda:0', 'cuda:1'}
        assert [wave['wave'] for wave in manifest['waves']] == ['pretrain', 'probe', 'downstream']
        flat = [j for wave in manifest['waves'] for jobs in wave['jobs'].values() for j in jobs]
        assert sorted(flat) == sorted(manifest['commands']) and len(flat) == 30
        for wave in manifest['waves']:
            for card, jobs in wave['jobs'].items():
                assert all(plan[j] == card for j in jobs)
        steps = {card: sum(runtime.matrix()[j]['steps'] for j in plan if plan[j] == card)
                 for card in ('cuda:0', 'cuda:1')}
        assert steps == {'cuda:0': 46000, 'cuda:1': 46000}, steps
        assert 'cuda:1' in plan['pretrain_attention'] and 'cuda:0' in plan['pretrain_mean']
    else:
        assert all(argv[-2] == '--device' and argv[-1] == 'cuda:1' for argv in manifest['commands'].values())
    return dict(release_sha256=digest, bound_files=len(manifest['files']), status=manifest['status'],
                formal_updates=manifest['formal_updates'], environment=manifest['environment'],
                jobs_started_under_this_release=started, commands=len(manifest['commands']),
                allocation=manifest.get('allocation'), waves=[w['wave'] for w in manifest.get('waves', [])])


def check_gate():
    assert not MARKER.exists(), 'approval marker exists before the independent audit'
    temporary = AUDIT / 'gate_probe.json'
    temporary.write_text(json.dumps(dict(status='READY_FOR_TRAINING', release_sha256='0' * 64)), encoding='utf8')
    try:
        runtime.verify_release(RELEASE, temporary)
        raise AssertionError('mismatched release hash was accepted')
    except ValueError:
        pass
    temporary.write_text(json.dumps(dict(status='READY_FOR_TRAINING',
                                        release_sha256=sha(RELEASE / 'manifest.json'))), encoding='utf8')
    runtime.verify_release(RELEASE, temporary)
    temporary.unlink()
    try:
        runtime.verify_release(RELEASE, MARKER)
        raise AssertionError('missing marker was accepted')
    except (FileNotFoundError, ValueError):
        pass
    return dict(without_marker_rejected=True, wrong_hash_rejected=True, matching_hash_accepted=True,
                marker_absent_before_audit=True)


def check_structure():
    mean, attention = initialize_pair()
    keys_mean, keys_attention = mean.state_dict(), attention.state_dict()
    assert set(keys_mean) < set(keys_attention) and set(keys_attention) - set(keys_mean) == {
        'spectral.in_proj_weight', 'spectral.in_proj_bias', 'spectral.out_proj.weight', 'spectral.out_proj.bias'}
    assert all(torch.equal(keys_mean[k], keys_attention[k]) for k in keys_mean)
    assert mean.spectral is None and isinstance(attention.spectral, nn.MultiheadAttention)
    assert isinstance(mean.readout, nn.MultiheadAttention) and isinstance(attention.readout, nn.MultiheadAttention)
    difference = sum(p.numel() for p in attention.parameters()) - sum(p.numel() for p in mean.parameters())
    assert difference == 3 * 32 * 32 + 3 * 32 + 32 * 32 + 32
    assert [len(s) for s in attention.stages] == [1, 1, 2, 1]
    assert [[b.shifted for b in s] for s in attention.stages] == [[False], [False], [False, True], [False]]
    # Real training artifacts: shared encoder, FFN, queries and readout must match.
    left = torch.load(SUITE / 'pretrain_mean/initialization.pt',
                      map_location='cpu', weights_only=False)['model']
    right = torch.load(SUITE / 'pretrain_attention/initialization.pt',
                       map_location='cpu', weights_only=False)['model']
    extra = set(right) - set(left)
    shared = [k for k in left if k in right and k != 'model_sha256']
    unequal = [k for k in shared if not torch.equal(left[k], right[k])]
    assert extra and all('use.spectral' in k for k in extra), extra
    assert unequal == [], unequal
    return dict(shared_tensors=len(shared), attention_only_extra_tensors=sorted(extra),
                attention_only_extra_parameters=difference, shared_unequal=0,
                query_cross_attention_in_both=True, stage_depth=[1, 1, 2, 1])


def check_fine_positions():
    torch.manual_seed(0)
    result = {}
    for flag in (False, True):
        model = USEv2(flag).eval()
        base = observation(1, 16, 16)
        base['x'].zero_()
        moved = dict(base, x=base['x'].clone())
        moved['x'][:, :, 5, 6] = 3.
        with torch.no_grad():
            plain, changed = model(base)['fine'], model(moved)['fine']
        support = {(int(r), int(c)) for _, r, c in ((changed - plain).abs().amax(1) > 0).nonzero()}
        assert support == {(4, 6), (4, 7), (5, 6), (5, 7)}, support
        single = observation(1, 8, 8)
        single['x'][:, :, 0, 1] = 2.
        with torch.no_grad():
            fine = model(single)['fine']
        assert not torch.allclose(fine[:, :, 0, 0], fine[:, :, 0, 1])
        result['attention' if flag else 'mean'] = dict(changed_pixels=len(support), patch='rows4-5 cols6-7')
    return result


def check_masks():
    model = USEv2(True).eval()
    invalid = observation(3, 10, 10)
    invalid['pixel_valid'][:, :, :5] = False
    with torch.no_grad():
        output = model(invalid)
    assert all(torch.isfinite(t).all() for t in [output['fine']] + output['context'])
    assert output['fine'][:, :, :5].abs().sum() == 0
    assert torch.equal(output['valid']['fine'], invalid['pixel_valid'].any(1))
    odd = observation(2, 9, 11)
    with torch.no_grad():
        output = model(odd)
    assert output['fine'].shape[-2:] == (9, 11)
    assert [t.shape[-2:] for t in output['context']] == [(3, 3), (2, 2), (1, 1), (1, 1)]
    empty = observation(3, 8, 8)
    empty['pixel_valid'][:] = False
    for bad, reason in ((empty, 'empty observation'), (dict(observation(), label=1), 'label key')):
        try:
            model(bad)
        except ValueError:
            continue
        raise AssertionError('not rejected: ' + reason)
    nan = observation(3, 8, 8)
    nan['pixel_valid'][:, 1] = False
    finite = dict(nan, x=nan['x'].clone())
    finite['x'][:, 1] = 0.
    nan = dict(finite, x=finite['x'].clone())
    nan['x'][:, 1] = float('nan')
    with torch.no_grad():
        assert torch.isfinite(model(nan)['fine']).all()
        assert torch.equal(model(nan)['fine'], model(finite)['fine'])
    return dict(odd_grid_context=[(3, 3), (2, 2), (1, 1), (1, 1)], empty_rejected=True, label_rejected=True,
                nonfinite_invalid_ignored=True)


def check_metadata_zero():
    torch.manual_seed(1)
    for flag in (False, True):
        model = USEv2(flag).eval()
        plain = observation(4, 16, 16)
        with torch.no_grad():
            reference = model(plain)['fine']
            for parameter in model.physical.parameters():
                parameter.add_(100.)
            shifted = model(plain)['fine']
        assert torch.equal(reference, shifted), 'unknown metadata still contributes'
    strict = observation(4, 8, 8)
    strict['physical'] = torch.zeros(2, 4, 3)
    strict['known_mask'] = torch.ones(2, 4, 3, dtype=torch.bool)
    try:
        USEv2(True)(strict)
        raise AssertionError('known zero metadata accepted')
    except ValueError:
        pass
    partial = observation(4, 8, 8)
    partial['physical'] = torch.tensor([.5, 0., 0.]).expand(2, 4, 3).clone()
    partial['known_mask'] = torch.zeros(2, 4, 3, dtype=torch.bool)
    partial['known_mask'][:, :, 0] = True
    with torch.no_grad():
        assert torch.isfinite(USEv2(True)(partial)['fine']).all()
    nonzero_unknown = observation(4, 8, 8)
    nonzero_unknown['physical'] = torch.tensor([.5, .1, 10.]).expand(2, 4, 3).clone()
    nonzero_unknown['known_mask'] = torch.zeros(2, 4, 3, dtype=torch.bool)
    nonzero_unknown['known_mask'][:, :, 0] = True
    try:
        USEv2(True)(nonzero_unknown)
        raise AssertionError('non-zero placeholder for an unknown field accepted')
    except ValueError:
        pass
    return dict(exactly_zero_when_all_unknown=True, known_zero_rejected=True, partial_known_finite=True)


def check_views(splits, stats):
    rows = {'s2': splits['TRAIN']['s2'][:2], 'landsat': splits['TRAIN']['landsat'][:2]}
    model = initialize('mean')
    report = {}
    for sensor, batch in rows.items():
        original = load_rows(batch, stats, DEVICE)
        visible = original['pixel_valid'] & original['band_valid'][:, :, None, None]
        hidden_fraction = []
        for seed in range(16):
            view, hidden, summary = task_view(original, 'fine', seed)
            assert not (hidden & ~visible).any()
            hidden_fraction.append(float(hidden.sum() / visible.sum()))
            tampered = dict(original, x=original['x'].clone())
            tampered['x'][hidden] = 999.
            other, other_hidden, other_summary = task_view(tampered, 'fine', seed)
            assert torch.equal(hidden, other_hidden) and torch.equal(view['x'], other['x'])
            assert torch.equal(summary, other_summary), 'hidden values reached the target summary'
            recomputed = []
            seen = visible & ~hidden
            for band in range(visible.shape[1]):
                values = torch.where(seen[:, band], view['x'][:, band], 0.)
                count = seen[:, band].sum((-1, -2)).clamp_min(1)
                mean = values.sum((-1, -2)) / count
                std = (torch.where(seen[:, band], values - mean[..., None, None], 0.).square()
                       .sum((-1, -2)) / count).sqrt()
                recomputed.append(torch.stack((mean, std), -1))
            assert torch.allclose(torch.stack(recomputed, 1), summary, atol=1e-4, rtol=1e-4), 'summary reads hidden pixels'
        view, hidden, _ = task_view(original, 'context', 3)
        per_sample = []
        for i in range(hidden.shape[0]):
            plane = hidden[i].any(0)
            block = plane.nonzero()
            rows = {int(v) for v in block[:, 0]}
            cols = {int(v) for v in block[:, 1]}
            assert len(rows) <= 32 and len(cols) <= 32, (len(rows), len(cols))
            assert int(plane.sum()) <= 32 * 32
            per_sample.append(int(plane.sum()))
        if bool(original['pixel_valid'].all()):
            assert set(per_sample) == {32 * 32}, per_sample
        spectral, spectral_hidden, _ = task_view(original, 'spectral', 5)
        known = original['known_mask'][:, :, 0]
        assert not (spectral_hidden & ~known[:, :, None, None]).any(), 'spectral target lacks physical identity'
        for task in TASKS:
            one = load_rows([batch[0]], stats, DEVICE)
            view, hidden, _ = task_view(one, task, 700000, metadata_mode='known')
            valid_one = one['pixel_valid'] & one['band_valid'][:, :, None, None]
            assert hidden.any() and not (hidden & ~valid_one).any(), task
            with torch.no_grad():
                assert torch.isfinite(model.use(view)['fine']).all(), task
        fractions = [float(task_view(original, 'fine', 11, metadata_mode=m)[0]['known_mask'].float().mean())
                     for m in ('known', 'partial', 'unknown')]
        modes = [task_view(original, 'fine', 11, metadata_mode=m)[1] for m in ('known', 'partial', 'unknown')]
        assert all(torch.equal(modes[0], m) for m in modes), 'spatial masks differ across metadata modes'
        report[sensor] = dict(fine_fraction=sum(hidden_fraction) / len(hidden_fraction),
                              metadata_known_fraction=fractions, context_block=32)
    assert 0.2 < report['s2']['fine_fraction'] < 0.3, report['s2']['fine_fraction']
    assert report['s2']['metadata_known_fraction'][0] > .9
    assert report['s2']['metadata_known_fraction'][2] == 0 and report['landsat']['metadata_known_fraction'][2] == 0
    return report


def check_streams(splits):
    batches = [sample_rows(splits['TRAIN'], step) for step in range(1, 201)]
    train_files = {r['file'] for sensor in splits['TRAIN'] for r in splits['TRAIN'][sensor]}
    counts = Counter()
    for batch in batches:
        assert len(batch) == 8, len(batch)
        assert all(r['file'] in train_files for r in batch), 'sampling left TRAIN'
        counts.update(r['sensor'] for r in batch)
    assert counts['s2'] == 1200 and counts['landsat'] == 400, counts
    torch.manual_seed(999)
    assert [r['file'] for r in sample_rows(splits['TRAIN'], 7)] == [r['file'] for r in sample_rows(splits['TRAIN'], 7)]
    return dict(updates=200, per_update=8, sensor_counts=dict(counts), train_only=True,
                independent_of_torch_rng=True)


def check_calibration(splits, stats, configuration):
    path = ROOT / 'configs/use_v2/calibration.json'
    computed = calibrate(splits, stats)
    assert json.dumps(computed, indent=2, ensure_ascii=False) == path.read_text(encoding='utf8'), 'calibration not reproducible'
    calibration = read(path)
    assert sha(path) == configuration['use_v2']['calibration_sha256']
    train = {s: {r['file'] for r in splits['TRAIN'][s]} for s in splits['TRAIN']}
    for sensor, files in calibration['members'].items():
        assert len(files) == 32 and set(files) <= train[sensor]
    assert all(calibration['coefficients'][s][t] > 0 for s in calibration['coefficients'] for t in TASKS)
    return dict(recomputed_sha256=sha(path), members_per_sensor=32, train_only=True,
                coefficients=calibration['coefficients'])


def check_data(configuration):
    rows, _ = members()
    audited = ROOT / 'outputs/use_independent_frozen/cache_audit_v1/members.jsonl'
    assert sha(audited) == configuration['use_v2']['members_sha256']
    stored = [json.loads(line) for line in audited.read_text(encoding='utf8').splitlines()]
    assert [r['file'] for r in rows] == [r['file'] for r in stored]
    failures = []
    for row in rows:
        try:
            audit_cache(EXTERNAL, row)
        except (ValueError, OSError, KeyError) as error:
            failures.append(dict(file=row['file'], error=str(error)))
    roles = Counter(r['role'] for r in rows)
    assert not failures, failures[:3]
    assert roles == Counter({'TRAIN': 11132, 'PRETRAIN_VAL': 599, 'GUARD': 239}), roles
    assert max(Counter(r['file'] for r in rows).values()) == 1
    return dict(roles=dict(roles), sensors={f'{role}/{sensor}': count for (role, sensor), count
                                           in Counter((r['role'], r['sensor']) for r in rows).items()},
                failures=len(failures), duplicate_files=0,
                geographic_isolation='inherited from the accepted source product/geographic audits')


def check_budget(configuration):
    jobs = runtime.matrix()
    planned = configuration['use_v2']['jobs']
    assert len(jobs) == 30 and set(jobs) == set(planned)
    assert all(jobs[name]['steps'] == planned[name]['steps'] for name in jobs)
    assert sum(j['steps'] for j in jobs.values()) == 92000
    modes = Counter(job['mode'] for job in jobs.values())
    assert modes == Counter({'probe': 22, 'downstream': 6, 'pretrain': 2}), modes
    assert all(jobs[n]['steps'] == 20000 for n in ('pretrain_mean', 'pretrain_attention'))
    assert all(jobs[n]['steps'] == 5000 for n in jobs if n.startswith('downstream_'))
    assert all(jobs[n]['steps'] == 1000 for n in jobs if n.startswith('probe_'))
    breakdown = dict(pretrain=sum(j['steps'] for j in jobs.values() if j['mode'] == 'pretrain'),
                     probe=sum(j['steps'] for j in jobs.values() if j['mode'] == 'probe'),
                     downstream=sum(j['steps'] for j in jobs.values() if j['mode'] == 'downstream'))
    assert breakdown == dict(pretrain=40000, probe=22000, downstream=30000), breakdown
    assert lr(1000, 20000, 1000) == 6e-4 and lr(20000, 20000, 1000) < 1e-9 and lr(50, 1000, 50) == 6e-4
    return dict(jobs=30, modes=dict(modes), breakdown=breakdown, schedule_peak=6e-4,
                cosine_end=float(lr(20000, 20000, 1000)))


def check_fresh_initialization(configuration):
    acceptance = read(ACCEPTANCE)
    reference = state_sha256(initialize('mean'))
    attention = state_sha256(initialize('attention'))
    assert reference == acceptance['suite']['summaries']['pretrain_mean']['identity']['initialization_sha256']
    assert attention == acceptance['suite']['summaries']['pretrain_attention']['identity']['initialization_sha256']
    for job, summary in acceptance['suite']['summaries'].items():
        assert summary['initialization_reset_sha256'] == summary['identity']['initialization_sha256'], job
    downstream = runtime.build(configuration['use_v2']['jobs']['downstream_mean_context_fine'], DEVICE,
                               engineering=True)
    assert state_sha256(downstream) == acceptance['suite']['summaries']['downstream_mean_context_fine'][
        'identity']['initialization_sha256']
    assert acceptance['actual_formal_updates'] == 0 and acceptance['formal_planned_updates'] == 92000
    return dict(mean_reference=reference, attention_reference=attention,
                downstream_reference=state_sha256(downstream), reset_equals_identity_for_all_suite_jobs=True,
                engineering_formal_updates=0)


def check_freeze_and_resume(splits, stats, calibration):
    deterministic()
    probe = RecoveryProbe(initialize('mean').use, 'fine')
    frozen = state_sha256(probe.use.encoder)
    optimizer = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=.0006, weight_decay=.01)
    assert not {id(p) for p in probe.use.encoder.parameters()} & {
        id(p) for group in optimizer.param_groups for p in group['params']}
    record = ssl_update(probe, optimizer, splits['TRAIN'], stats, 1, calibration, task='fine', device=DEVICE)
    assert state_sha256(probe.use.encoder) == frozen, 'frozen USE state changed during an update'
    assert probe.use.verify(optimizer) == frozen and record['views']
    try:
        FrozenUSE(USEv2()).forward(dict(observation(), mask=torch.zeros(2, 16, 16)))
        raise AssertionError('labels entered the frozen boundary')
    except ValueError:
        pass
    for name in ('resume_whole', 'resume_split'):
        shutil.rmtree(AUDIT / name, ignore_errors=True)
    whole = runtime.train('pretrain_mean', AUDIT / 'resume_whole', DEVICE, engineering=True, engineering_steps=2)
    runtime.train('pretrain_mean', AUDIT / 'resume_split', DEVICE, engineering=True, engineering_steps=2, stop_at=1)
    split = runtime.train('pretrain_mean', AUDIT / 'resume_split', DEVICE, engineering=True, engineering_steps=2,
                          resume=True)
    left = torch.load(AUDIT / 'resume_whole' / 'last.pt', map_location='cpu', weights_only=False)
    right = torch.load(AUDIT / 'resume_split' / 'last.pt', map_location='cpu', weights_only=False)
    assert left['model_sha256'] == right['model_sha256'] and left['history'] == right['history']
    assert runtime.states_equal(left['optimizer'], right['optimizer'])
    assert runtime.states_equal(left['rng'], right['rng'])
    assert split['updates'] == whole['updates'] == 2
    return dict(frozen_sha256=frozen, optimizer_excludes_encoder=True, labels_rejected=True,
                resume_updates=2, resume_model_sha256=left['model_sha256'], resume_optimizer_exact=True,
                resume_rng_exact=True)


def check_preflight_and_area(splits, stats, configuration):
    from check_research_protocol import audit_config
    proof = audit_config(ROOT / 'configs/use_v2/execution.yaml')
    stored = read(ROOT / 'configs/use_v2/protocol_preflight.json')
    assert proof['ok'] and proof['counts'] == dict(train=2223, dev=669)
    assert proof['effective_config_sha256'] == stored['effective_config_sha256']
    assert sha(ROOT / 'configs/use_v2/execution.yaml') == proof['config_sha256']
    assert configuration['segformer']['data']['mixed_bands'] is False
    row = splits['PRETRAIN_VAL']['s2'][0]
    batch, provenance = area30(row, stats['s2'], DEVICE)
    assert provenance['target_gsd_m'] == 30 and provenance['minimum_valid_area'] == .8
    assert provenance['crs'].startswith('EPSG:32') and provenance['source_crs'] == row['crs']
    assert provenance['footprint_area_m2'] > 0 and batch['pixel_valid'].any()
    return dict(preflight_ok=True, counts=proof['counts'], normalization_fit_n=proof['normalization_fit_n'],
                source_gsd_m=abs(row['transform'][0]), target_gsd_m=provenance['target_gsd_m'],
                target_crs=provenance['crs'], valid_fraction_mean=provenance['valid_fraction_mean'],
                note='30m is a derived diagnostic; no same-place cross-sensor pairing is claimed')


def check_reconstruction_target(splits, stats):
    original = load_rows([splits['PRETRAIN_VAL']['s2'][0]], stats, DEVICE)
    view, hidden, _ = task_view(original, 'context', 42)
    valid = original['pixel_valid'] & original['band_valid'][:, :, None, None]
    assert hidden.any() and not (hidden & ~valid).any()
    predicted = torch.zeros_like(original['x'])
    loss = masked_mae(predicted, original['x'], hidden)
    manual = (predicted - original['x']).abs()[hidden].mean()
    assert torch.allclose(loss, manual), (float(loss), float(manual))
    # The target mask is the hidden set of originally valid pixels; output support never replaces it.
    assert masked_mae(predicted, original['x'], hidden & valid) == loss
    return dict(hidden_inside_original_valid=True, loss_equals_hidden_mean_abs_error=True,
                hidden_pixels=int(hidden.sum()), support_pixels=int(view['pixel_valid'].sum()))


def main():
    global RELEASE, MARKER, SUITE, ACCEPTANCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', default=str(RELEASE))
    parser.add_argument('--marker', default=str(MARKER))
    parser.add_argument('--suite', default=str(SUITE))
    parser.add_argument('--acceptance', default=str(ACCEPTANCE))
    parser.add_argument('--out', default=str(AUDIT / 'release_audit_v1.json'))
    parser.add_argument('--tests-xml', default=str(AUDIT / 'tests_audit_v1.xml'))
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    RELEASE, MARKER = Path(args.release).resolve(), Path(args.marker).resolve()
    SUITE, ACCEPTANCE = Path(args.suite).resolve(), Path(args.acceptance).resolve()
    out_path = Path(args.out).resolve()
    torch.set_num_threads(4)
    AUDIT.mkdir(parents=True, exist_ok=True)
    configuration, calibration, splits, stats = runtime.load_configuration()
    checks = []

    def run(name, function, *arguments):
        began = time.perf_counter()
        try:
            checks.append(dict(name=name, passed=True, seconds=round(time.perf_counter() - began, 2),
                               detail=function(*arguments)))
        except Exception as error:  # Audit records the failure instead of hiding it.
            checks.append(dict(name=name, passed=False, seconds=round(time.perf_counter() - began, 2),
                               error=repr(error)))
        print(json.dumps({k: checks[-1][k] for k in ('name', 'passed')}), flush=True)

    run('release_integrity', check_release)
    run('formal_gate', check_gate)
    run('two_group_structure', check_structure)
    run('fine_position_locality', check_fine_positions)
    run('spatial_and_quality_masks', check_masks)
    run('unknown_metadata_zero', check_metadata_zero)
    run('masking_target_identity', check_views, splits, stats)
    run('sampling_streams', check_streams, splits)
    run('calibration_lock', check_calibration, splits, stats, configuration)
    run('data_roles', check_data, configuration)
    run('budget_and_schedule', check_budget, configuration)
    run('fresh_initialization', check_fresh_initialization, configuration)
    run('freeze_and_resume', check_freeze_and_resume, splits, stats, calibration)
    run('protocol_and_area30', check_preflight_and_area, splits, stats, configuration)
    run('reconstruction_target', check_reconstruction_target, splits, stats)
    tests = None
    xml = Path(args.tests_xml).resolve()
    if xml.exists():
        import xml.etree.ElementTree as ElementTree
        suite = ElementTree.parse(xml).getroot().find('testsuite')
        tests = dict(tests=int(suite.get('tests')), failures=int(suite.get('failures')),
                     errors=int(suite.get('errors')), skipped=int(suite.get('skipped')),
                     sha256=sha(xml), file=str(xml.relative_to(ROOT)))
    passed = all(check['passed'] for check in checks)
    report = dict(status='PASS' if passed else 'FAIL', release=str(RELEASE),
                  release_sha256=sha(RELEASE / 'manifest.json'), checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  python=sys.version, executable=sys.executable, cwd=str(ROOT), device=DEVICE, checks=checks,
                  independent_tests=tests, failing=[c['name'] for c in checks if not c['passed']],
                  evidence=dict(config='configs/use_v2/execution.yaml',
                                calibration='configs/use_v2/calibration.json',
                                members='outputs/use_independent_frozen/cache_audit_v1/members.jsonl',
                                engineering=str(ACCEPTANCE.relative_to(ROOT))))
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(dict(status=report['status'], failing=report['failing'])), flush=True)
    if not passed:
        raise SystemExit(1)
    if args.publish:
        marker = dict(status='READY_FOR_TRAINING', release_sha256=report['release_sha256'],
                      release=RELEASE.name, audit=str(out_path.relative_to(ROOT)),
                      audit_sha256=sha(out_path), published_at=report['checked_at'], published_by='Agent C',
                      checks=len(checks), independent_tests=tests,
                      conditions=['B starts each of the 30 manifest commands only after verify_release accepts this marker',
                                  'Any source, config, data or environment change invalidates this approval',
                                  'Formal runs re-initialize; engineering weights are never inherited',
                                  'Each wave runs one process per card on the device the manifest assigns; no DDP and no batch change'],
                      notes=['engineering_v1 snapshot and its approval marker stay byte-identical and are not reused for new runs',
                             'pretrain_mean was started under engineering_v1 and is not restarted; later jobs consume its checkpoint through the recorded source hashes'],
                      limits=['Single seed 1234; no significance claim',
                              'Project event-held-out protocol, not the official split',
                              'Second labeled task and true cross-sensor pairing not completed'])
        MARKER.write_text(json.dumps(marker, indent=2, ensure_ascii=False), encoding='utf8')
        print(json.dumps(dict(marker=str(MARKER.relative_to(ROOT)), sha256=sha(MARKER))), flush=True)


if __name__ == '__main__':
    main()






