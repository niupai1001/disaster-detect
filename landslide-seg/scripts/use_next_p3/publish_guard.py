"""Narrow frozen-consumer guard release; no change to the executing P2 release."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from use_next_p2.protocol import sha


def main():
    parent=ROOT/'outputs/use_next_v2/p2_releases/engineering_v1';old=json.loads((parent/'manifest.json').read_text())
    target=ROOT/'outputs/use_next_v2/p3_releases/consumer_guard_v1';target.mkdir(parents=True,exist_ok=False)
    evidence=ROOT/'outputs/use_next_v2/p3_engineering';results={}
    for mode in ('pixel','physical'):
        result=json.loads((evidence/f'guard_{mode}_head_trial/summary.json').read_text())
        baseline=json.loads((ROOT/f'outputs/use_next_v2/p2_engineering/release_{mode}_head/summary.json').read_text())
        assert result['exact_disk_replay'] and result['frozen_unchanged'] and result['successful_updates_including_replay']==3
        assert result['initialization_sha256']==baseline['initialization_sha256']
        assert result['final_local_sha256']==baseline['final_local_sha256']
        assert result['identity']['config_sha256']==baseline['identity']['config_sha256']
        results[mode]=result
    paths=set(ROOT/name for name in old['files'])
    for name in ('src/landslide_seg/models/use_next_p3/__init__.py','src/landslide_seg/models/use_next_p3/readout.py',
                 'scripts/use_next_p3/run_guarded_p2_head.py','scripts/use_next_p3/publish_guard.py',
                 'tests/use_next_p3/test_guard.py','tests/use_next_p3/test_guard_lineage.py',
                 'outputs/use_next_v2/p3_engineering/tests_guard_lineage.xml',
                 'outputs/use_next_v2/audit/READY_P2.json','outputs/use_next_v2/audit/P1_FINAL_ACCEPTANCE.json'):
        paths.add(ROOT/name)
    for mode in ('pixel','physical'):
        for name in ('summary.json','curve.jsonl','replay_comparison.json'):paths.add(evidence/f'guard_{mode}_head_trial'/name)
    files={}
    for path in sorted(paths):
        relative=path.relative_to(ROOT);dest=target/'snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);files[relative.as_posix()]=sha(dest)
    manifest=dict(status='READY_FOR_INDEPENDENT_AUDIT',stage='P2_HEAD_GUARD',files=files,environment=old['environment'],weights=old['weights'],
        parent_release=parent.relative_to(ROOT).as_posix(),parent_manifest_name='manifest.json',parent_release_sha256=sha(parent/'manifest.json'))
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2));digest=sha(target/'manifest.json')
    ready=ROOT/'outputs/use_next_v2/audit/READY_P2.json';previous=ROOT/'outputs/use_next_v2/audit/P1_FINAL_ACCEPTANCE.json'
    base=json.loads((parent/'READY_TEMPLATE_NOT_APPROVAL.json').read_text())
    base.update(status='READY_P2_HEAD',release_sha256=digest,parent_release_sha256=sha(parent/'manifest.json'),
                ready_p2_sha256=sha(ready),previous_acceptance_sha256=sha(previous))
    (target/'READY_TEMPLATE_NOT_APPROVAL.json').write_text(json.dumps(base,indent=2))
    commands=[dict(mode=mode,device=f'cuda:{i}',out=f'outputs/use_next_v2/runs/p2_{mode}_head',
        command=f'.venv-use-independent/Scripts/python.exe -X utf8 -u scripts/use_next_p3/run_guarded_p2_head.py --config configs/use_next_p2/{mode}_head.yaml --device cuda:{i} --out outputs/use_next_v2/runs/p2_{mode}_head --release {target.relative_to(ROOT).as_posix()} --marker outputs/use_next_v2/audit/READY_P2_HEAD.json') for i,mode in enumerate(('pixel','physical'))]
    report=dict(status='READY_FOR_INDEPENDENT_AUDIT',release=target.relative_to(ROOT).as_posix(),release_sha256=digest,
        parent_release_sha256=sha(parent/'manifest.json'),new_optimizer_updates=6,formal_updates=0,
        mathematical_equivalence='GuardedReadout directly inherits the exact ScaleReadout.forward/train and all tensors; only adds read-only post-optimizer SHA validation',
        initialization_sha256={k:v['initialization_sha256'] for k,v in results.items()},u0_sha256=base['u0_sha256'],
        cpu_proof='Both configurations real2 updates plus exact disk step2 replay; CPU1, noMKLDNN/noMHAfastpath, no tolerance relaxation',
        cuda_proof='Original CUDA mathematical paths passed; independent guarded CUDA verification still required before READY_P2_HEAD',
        source_constraint='Configured same-mode formal6000, FD release identity, original pretrain config SHA, complete summary/checkpoint identity and local tensor SHA required',
        source_files_unchanged=True,commands=commands,
        guard_scope='Complete frozen USE including local/teacher/native parameters and registered buffers after each optimizer step; failure before any new checkpoint commit')
    (target/'ACCEPTANCE.json').write_text(json.dumps(report,indent=2))
    (ROOT/'outputs/use_next_v2/p3_requests/P2_HEAD_GUARD.json').write_text(json.dumps(report,indent=2))
    print(digest)


if __name__=='__main__':main()
