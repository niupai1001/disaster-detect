"""Freeze the six checked P3/P5 paths without changing inherited releases."""
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from use_next_p2.protocol import sha
from use_next_p2.verify import verify
from use_next_p3.protocol import audit


def main():
    parent=ROOT/'outputs/use_next_v2/p3_releases/consumer_guard_v1'
    parent_sha=verify(parent)
    inherited=json.loads((parent/'manifest.json').read_text(encoding='utf8'))
    evidence=ROOT/'outputs/use_next_v2/p3_engineering'
    results={};configs={};audits={}
    for objective in ('scale','latent','joint'):
        for task in ('pretrain','head'):
            name=f'{objective}_{task}';config=ROOT/f'configs/use_next_p3/{name}.yaml'
            probe=ROOT/f'outputs/use_next_v2/audit/p3_shared_cuda_{name}'
            result=json.loads((probe/'summary.json').read_text(encoding='utf8'))
            receipt=json.loads((probe/'independent_probe_receipt.json').read_text(encoding='utf8'))
            assert receipt['status']=='PREPUBLICATION_ENGINEERING_PASS' and receipt['production_locks_unchanged']
            for source,expected in receipt['source_sha256'].items():assert sha(ROOT/source)==expected
            assert result['complete'] and result['updates']==2 and result['formal_updates']==0
            assert result['exact_disk_replay'] and result['frozen_unchanged']
            assert result['successful_updates_including_replay']==3
            assert result['identity']['config_sha256']==sha(config)
            assert result['identity']['objective']==objective
            assert result['identity']['stage']==('P3' if task=='pretrain' else 'P5')
            results[name]=result;configs[config.relative_to(ROOT).as_posix()]=sha(config)
            audits[name]=audit(config)
    for task in ('pretrain','head'):
        assert len({results[f'{o}_{task}']['initialization_sha256'] for o in ('scale','latent','joint')})==1
    diagnostic=json.loads((evidence/'diagnostic_engineering.json').read_text(encoding='utf8'))
    assert diagnostic['status']=='P3_REPRESENTATION_GATE_PASS' and not diagnostic['formal_fixed_endpoint']
    assert diagnostic['teacher_covariance_trace']>1e-8
    gate=json.loads((ROOT/'outputs/use_next_v2/conditional_gates/P4_ELIGIBILITY.json').read_text(encoding='utf8'))
    assert sha(ROOT/gate['source'])==gate['source_sha256']
    pairs=[json.loads(line) for line in (ROOT/gate['source']).read_text(encoding='utf8').splitlines()]
    assert not any(p['eligible'] for p in pairs)
    assert gate['verified_train_geographic_groups']==gate['verified_validation_geographic_groups']==0
    (evidence/'protocol_receipts.json').write_text(json.dumps(audits,indent=2),encoding='utf8')
    # Count every retained optimizer receipt, including earlier superseded smoke runs.
    budget={p.parent.name:json.loads(p.read_text(encoding='utf8'))['successful_updates_including_replay']
            for p in evidence.glob('*/summary.json')}
    failed={p.parent.name:json.loads(p.read_text(encoding='utf8'))['successful_updates_this_session']
            for p in evidence.glob('*/failure.json')}
    unit_updates={p.name:sum('test_objective_and_ema_boundaries' in t.get('name','')
                  for t in ET.parse(p).iter('testcase')) for p in evidence.glob('*.xml')}
    (evidence/'budget_ledger.json').write_text(json.dumps(dict(formal_updates=0,successful_smoke_updates=budget,
        failed_smoke_successful_optimizer_updates=failed,independent_cuda_six_path_updates=18,
        unit_optimizer_updates_by_retained_receipt=unit_updates,
        limitation='CPU engineering timings are not CUDA cost comparisons; superseded runs are not formal models'),indent=2),encoding='utf8')
    paths={ROOT/p for p in inherited['files']}
    for pattern in ('src/landslide_seg/models/use_next_p3/*.py','scripts/use_next_p3/*.py',
                    'configs/use_next_p3/*.yaml','tests/use_next_p3/*.py','outputs/use_next_v2/conditional_gates/*.json'):
        paths.update(ROOT.glob(pattern))
    for name in ('u0.pt','tests_release.xml','diagnostic_engineering.json','protocol_receipts.json','budget_ledger.json'):
        paths.add(evidence/name)
    paths.update(evidence.glob('*.xml'))
    for name in results:
        for filename in ('summary.json','curve.jsonl','replay_comparison.json','failure.json'):
            path=evidence/f'final_{name}'/filename
            if path.exists():paths.add(path)
        probe=ROOT/f'outputs/use_next_v2/audit/p3_shared_cuda_{name}'
        for filename in ('summary.json','curve.jsonl','replay_comparison.json','independent_probe_receipt.json'):
            paths.add(probe/filename)
    target=ROOT/'outputs/use_next_v2/p3_releases/engineering_v1'
    target.mkdir(parents=True,exist_ok=False)
    files={}
    for path in sorted(paths):
        relative=path.relative_to(ROOT);dest=target/'snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        files[relative.as_posix()]=sha(dest)
    manifest=dict(status='READY_FOR_INDEPENDENT_AUDIT',stage='P3_P5',files=files,
        environment=inherited['environment'],weights=inherited['weights'],
        parent_release=parent.relative_to(ROOT).as_posix(),parent_manifest_name='manifest.json',parent_release_sha256=parent_sha)
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    digest=verify(target)
    base=json.loads((parent/'READY_TEMPLATE_NOT_APPROVAL.json').read_text(encoding='utf8'))
    for stage,previous in (('P3','P2'),('P5','P3')):
        marker={k:base[k] for k in ('data_manifest_sha256','u0_sha256','schedule_sha256')}
        marker.update(status=f'READY_{stage}',release_sha256=digest,config_sha256=configs,
            previous_acceptance_sha256=f'REPLACE_WITH_VERIFIED_{previous}_FINAL_ACCEPTANCE_SHA256')
        if stage=='P5':
            marker.update(p4_decision_sha256='REPLACE_BEFORE_REMAINING_HEADS',
                          joint_head_acceptance_sha256='REPLACE_BEFORE_REMAINING_HEADS')
        (target/f'{stage}_READY_TEMPLATE_NOT_APPROVAL.json').write_text(json.dumps(marker,indent=2),encoding='utf8')
    commands=[]
    for task in ('pretrain','head'):
        for objective in ('joint','scale','latent'):
            stage='P3' if task=='pretrain' else 'P5'
            out=f'outputs/use_next_v2/runs/p3_{objective}' if task=='pretrain' else f'outputs/use_next_v2/runs/p5_{objective}_head'
            commands.append(dict(stage=stage,objective=objective,
                command=f'.venv-use-independent/Scripts/python.exe -X utf8 -u scripts/use_next_p3/run.py --config configs/use_next_p3/{objective}_{task}.yaml --device CUDA_DEVICE --out {out} --release {target.relative_to(ROOT).as_posix()} --marker outputs/use_next_v2/audit/READY_{stage}.json'))
    report=dict(status='READY_FOR_INDEPENDENT_AUDIT',release=target.relative_to(ROOT).as_posix(),release_sha256=digest,
        parent_release_sha256=parent_sha,formal_updates=0,independent_cuda_six_path_updates=18,
        evidence='All six actual CUDA paths:2 updates plus exact disk replay; frozen checks;15 unit tests; source hashes rebound to this release',
        cpu_boundary='Three pretrain paths passed exact CPU replay. Retained CPU head attempts can differ by last floating-point bits despite CPU1/noMKLDNN/noMHAfastpath; failed strict checks are preserved, never relabeled as passes.',
        cuda_evidence='Independent final manifest verification and stage prerequisites required before READY_P3/P5; mixed-load probe timing is not a cost comparison',
        initialization_sha256={task:results[f'joint_{task}']['initialization_sha256'] for task in ('pretrain','head')},
        sequence=['P2 full acceptance','P3 three fresh same-initialization10000 runs and fixed-endpoint diagnostics',
                  'P3 full acceptance','P5 joint frozen head3000 and acceptance','P4 eligibility decision',
                  'P5 scale/latent frozen heads3000 each','Independent external-label eligibility decision','Final acceptance/report'],
        scientific_boundaries=['P3 strict integer20%-40% band removal is an additional disclosed change from P2 rounding',
            'P3 scale/latent/joint share structure, initialization, data and schedule; raw scale and EMA latent losses use unit weights',
            'Latent target uses detached local EMA, LayerNorm and true valid ground-area weighting; anchors excluded from EMA',
            'No verified SAR scale/noise target; unknown effective resolution never supplies high-frequency supervision',
            'Engineering representation gate only checks nondegeneracy/content dependence, not semantic transfer',
            'P4 pairs0 vs32 TRAIN/8 validation; external legal independent label manifest absent: independent skip decisions required; no budget reallocation'],
        commands=commands,diagnostic_command='.venv-use-independent/Scripts/python.exe -X utf8 -u scripts/use_next_p3/diagnose.py --checkpoint outputs/use_next_v2/runs/p3_OBJECTIVE/last.pt --out outputs/use_next_v2/audit/p3_OBJECTIVE_representation.json --device cpu')
    (target/'ACCEPTANCE.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (ROOT/'outputs/use_next_v2/p3_requests/HANDOFF.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(digest)


if __name__=='__main__':main()
