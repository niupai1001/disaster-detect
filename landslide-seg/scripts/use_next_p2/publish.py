"""Seal P2 without changing any P1-bound file; dependent stages get separate releases."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from use_next_p2.protocol import audit,sha


def main():
    parent=ROOT/'outputs/use_next_v2/releases/p0_engineering_v2'
    inherited=json.loads((parent/'release_manifest.json').read_text())
    output=ROOT/'outputs/use_next_v2/p2_engineering'
    results={}
    for mode in ('pixel','physical'):
        for task in ('pretrain','head'):
            name=f'release_{mode}_{task}';r=json.loads((output/name/'summary.json').read_text())
            assert r['exact_disk_replay'] and r['frozen_unchanged'] and r['formal_updates']==0
            assert r['identity']['code_sha256']==sha(ROOT/'scripts/use_next_p2/run.py')
            assert r['identity']['config_sha256']==sha(ROOT/f'configs/use_next_p2/{mode}_{task}.yaml')
            results[mode+'/'+task]=r
    assert results['pixel/pretrain']['initialization_sha256']==results['physical/pretrain']['initialization_sha256']
    assert results['pixel/head']['initialization_sha256']==results['physical/head']['initialization_sha256']
    receipts={mode+'/'+task:audit(ROOT/f'configs/use_next_p2/{mode}_{task}.yaml') for mode in ('pixel','physical') for task in ('pretrain','head')}
    (output/'protocol_receipts.json').write_text(json.dumps(receipts,indent=2))
    # Earlier CPU failed comparisons happened after their third successful optimizer update.
    counts={p.parent.name:json.loads(p.read_text()).get('successful_updates_including_replay',0) for p in output.glob('*/summary.json')}
    counts['pixel_head_trial_failed']=3;counts['pixel_head_trial_v2_failed_and_diagnostic_replay']=4
    (output/'budget_ledger.json').write_text(json.dumps(dict(formal_updates=0,optimizer_updates=sum(counts.values()),runs=counts,
        lock_wait_attempts_optimizer_updates=0,maximum_per_configuration=100,engineering_protocol='All successful optimizer updates including failures/replays; no budget reallocation'),indent=2))
    paths=set(ROOT/name for name in inherited['files'])
    for directory in ('src/landslide_seg/models/use_next_p2','scripts/use_next_p2','tests/use_next_p2','configs/use_next_p2'):
        paths.update(p for p in (ROOT/directory).rglob('*') if p.suffix in ('.py','.yaml','.json'))
    for name in ('calibration.json','u0.pt','initialization.json','tests_final.xml','protocol_receipts.json','budget_ledger.json'):
        paths.add(output/name)
    for name in ('release_pixel_pretrain','release_physical_pretrain','release_pixel_head','release_physical_head'):
        paths.update(output/name/f for f in ('summary.json','curve.jsonl','replay_comparison.json'))
    paths.add(ROOT/'outputs/use_next_v2/p2_requests/WORKFLOW_AUTHORIZATION.json')
    target=ROOT/'outputs/use_next_v2/p2_releases/engineering_v1'
    target.mkdir(parents=True,exist_ok=False);files={}
    for path in sorted(paths):
        relative=path.relative_to(ROOT);dest=target/'snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);files[relative.as_posix()]=sha(dest)
    env=dict(python=sys.version,dependencies={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','PyYAML','rasterio','pyproj','shapely','timm','transformers')})
    manifest=dict(status='READY_FOR_INDEPENDENT_AUDIT',stage='P2',files=files,environment=env,weights=inherited['weights'],
        parent_release=parent.relative_to(ROOT).as_posix(),parent_release_sha256=sha(parent/'release_manifest.json'))
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2));digest=sha(target/'manifest.json')
    marker=dict(status='READY_P2',release_sha256=digest,data_manifest_sha256=results['pixel/pretrain']['identity']['data_manifest_sha256'],
                u0_sha256=sha(output/'u0.pt'),schedule_sha256=sha(ROOT/'outputs/use_next_v2/data/schedule_seed1234.jsonl'),
                previous_acceptance_sha256='REPLACE_WITH_VERIFIED_P1_FINAL_ACCEPTANCE_SHA256',
                config_sha256={f'configs/use_next_p2/{mode}_{task}.yaml':sha(ROOT/f'configs/use_next_p2/{mode}_{task}.yaml') for mode in ('pixel','physical') for task in ('pretrain','head')})
    (target/'READY_TEMPLATE_NOT_APPROVAL.json').write_text(json.dumps(marker,indent=2))
    acceptance=dict(status='READY_FOR_INDEPENDENT_AUDIT',release_sha256=digest,formal_updates=0,
        engineering_updates=sum(counts.values()),tests='8 unit/geometry/masking/rotation/metric-schema tests, 4 real CUDA two-update exact disk replays',
        reconstruction='Same120m TRAIN-calibrated low-frequency targets; 4x4 source-pixel versus30m physical encoding only',
        source_resolution='Unknown effective resolution retained; unsupported high-frequency and SAR radiometric reconstruction unavailable and reported; SAR still exposed, no resampling',
        cpu_limit='CPU4 optimized consumer replay showed 1e-9 parameter differences; CPU1 noMKLDNN/noMHAfastpath and formal CUDA replay pass exactly; no tolerance relaxation',
        frozen_endpoint='Each head requires own same-mode nonengineering u6000; best/last select head only on full669 every500',
        gate='READY_P2 signed by independent owner and previous P1_ACCEPTED SHA mandatory; no formal work started here',
        p1_preserved=True,backend='FP32 TF32off, explicit AdamW foreachFalse/fusedFalse; common microbatch2/effective8')
    (target/'ACCEPTANCE.json').write_text(json.dumps(acceptance,indent=2))
    handoff=dict(status='READY_FOR_INDEPENDENT_AUDIT',release=target.relative_to(ROOT).as_posix(),release_sha256=digest,
        verify=f'.venv-use-independent/Scripts/python.exe scripts/use_next_p2/verify.py {target.relative_to(ROOT).as_posix()}',
        marker_template=(target/'READY_TEMPLATE_NOT_APPROVAL.json').relative_to(ROOT).as_posix(),
        prerequisites='P1_FINAL_ACCEPTANCE.json statusP1_ACCEPTED; exact previous_acceptance_sha256, u0/schedule/data/config bindings',
        commands={mode+'/'+task:f'.venv-use-independent/Scripts/python.exe scripts/use_next_p2/run.py --config configs/use_next_p2/{mode}_{task}.yaml --device cuda:{i} --out outputs/use_next_v2/runs/p2_{mode}{"_head" if task=="head" else ""} --release {target.relative_to(ROOT).as_posix()} --marker outputs/use_next_v2/audit/READY_P2.json' for i,mode in enumerate(('pixel','physical')) for task in ('pretrain','head')},
        allocation='Two6000 pretrains, then corresponding two3000 heads; total18000. Automatic only after independent gates.',
        workflow='P3/P5 isolated engineering remains active; full77k workflow authorized, no repeated user authorization needed')
    (ROOT/'outputs/use_next_v2/p2_requests/HANDOFF.json').write_text(json.dumps(handoff,indent=2))
    print(digest)


if __name__=='__main__':main()
