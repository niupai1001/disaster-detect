import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import torch
from use_next_p2.protocol import sha
from use_next_p2.verify import verify
from use_next_recovery.compare import equal


def main():
    parent=ROOT/'outputs/use_next_v2/p3_releases/engineering_v1';verify(parent)
    inherited=json.loads((parent/'manifest.json').read_text());base=ROOT/'outputs/use_next_v2/cached_p3'
    comparisons={}
    for objective in ('scale','latent','joint'):
        a=base/f'{objective}_original';b=base/f'{objective}_cached'
        for p in (a,b):
            r=json.loads((p/'summary.json').read_text());assert r['exact_disk_replay'] and r['frozen_unchanged'] and r['successful_updates_including_replay']==3
        equal(torch.load(a/'last.pt',map_location='cpu',weights_only=False),torch.load(b/'last.pt',map_location='cpu',weights_only=False))
        comparisons[objective]=dict(complete_checkpoint_bitwise=True,model_optimizer_ema_rng_history_exact=True)
    acceptance=ROOT/'outputs/use_next_v2/audit/P2_USER_STOP_ACCEPTANCE.json'
    assert json.loads(acceptance.read_text())['status']=='P2_USER_STOPPED'
    budget={p.parent.name:json.loads(p.read_text())['successful_updates_including_replay'] for p in base.glob('*/summary.json')}
    report=dict(status='READY_FOR_INDEPENDENT_AUDIT',comparisons=comparisons,engineering_updates=sum(budget.values()),engineering_update_receipts=budget,formal_updates=0,
        numpy_seed_policy='New P3 startup explicitly seeds global NumPy1234; old unseeded receipts preserved; actual fixed default_rng sample schedule unchanged',
        previous_stage_status='P2_USER_STOPPED',previous_acceptance_sha256=sha(acceptance),
        limitation='Shared short probes do not establish mixed-stream throughput; independent timing and release acceptance required')
    (base/'compatibility.json').write_text(json.dumps(report,indent=2))
    files={ROOT/n for n in inherited['files']}
    files.update((ROOT/'scripts/use_next_cached').glob('*.py'))
    files.update(ROOT/n for n in ('scripts/use_next_perf/geometry_cache.py','scripts/use_next_perf/disk_geometry.py',
        'scripts/use_next_recovery/compare.py',
        'outputs/use_next_v2/performance/tests_disk_retry.xml','outputs/use_next_v2/performance/disk_gradient_check/result.json',
        'outputs/use_next_v2/performance/schedule_reuse.json'))
    files.update((acceptance,base/'compatibility.json'))
    for objective in comparisons:
        for variant in ('original','cached'):
            for name in ('summary.json','curve.jsonl','replay_comparison.json'):files.add(base/f'{objective}_{variant}'/name)
    target=base/'release_v1';target.mkdir(exist_ok=False)
    hashes={}
    for path in sorted(files):
        relative=path.relative_to(ROOT);copy=target/'snapshot'/relative;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,copy);hashes[relative.as_posix()]=sha(copy)
    manifest=dict(status='READY_FOR_INDEPENDENT_AUDIT',files=hashes,environment=inherited['environment'],weights=inherited['weights'],
        parent_release=parent.relative_to(ROOT).as_posix(),parent_manifest_name='manifest.json',parent_release_sha256=sha(parent/'manifest.json'))
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2));digest=verify(target)
    marker=json.loads((parent/'P3_READY_TEMPLATE_NOT_APPROVAL.json').read_text())
    marker.update(status='READY_P3_CACHED',release_sha256=digest,previous_stage_status='P2_USER_STOPPED',previous_acceptance_sha256=sha(acceptance))
    (target/'READY_TEMPLATE_NOT_APPROVAL.json').write_text(json.dumps(marker,indent=2))
    report.update(release_sha256=digest,release=target.relative_to(ROOT).as_posix(),commands={o:f'.venv-use-independent/Scripts/python.exe -X utf8 -X faulthandler -u scripts/use_next_cached/run.py --config configs/use_next_p3/{o}_pretrain.yaml --device CUDA_DEVICE --out outputs/use_next_v2/runs/p3_{o} --release {target.relative_to(ROOT).as_posix()} --marker outputs/use_next_v2/audit/READY_P3_CACHED.json' for o in comparisons})
    (base/'HANDOFF.json').write_text(json.dumps(report,indent=2));print(digest)


if __name__=='__main__':main()
