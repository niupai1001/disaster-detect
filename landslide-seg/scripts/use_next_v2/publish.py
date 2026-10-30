"""Immutable engineering candidate; independent audit alone can sign READY_P0."""
import importlib.metadata
import json
import platform
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

from audit_data import ROOT,sha


def main():
    target=ROOT/'outputs/use_next_v2/releases/p0_engineering_v2'
    target.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(ROOT/'scripts/use_next_v2'),str(ROOT/'scripts'),str(ROOT/'src')]
    runpy.run_path(str(ROOT/'scripts/use_next_v2/run_p1.py'),run_name='release_dependency_scan')
    from check_research_protocol import audit_config
    receipts={name:audit_config(ROOT/f'configs/use_next_v2/p1_{name}.yaml') for name in ('dofa','anysat')}
    engineering=ROOT/'outputs/use_next_v2/engineering'
    native=[json.loads((engineering/name).read_text()) for name in ('native_dofa.json','native_anysat.json')]
    assert all(r['strict_load'] and r['direct_adapter_max_abs']==0 and r['unchanged'] for r in native)
    combined=json.loads((engineering/'pipeline_cuda_0_final/acceptance.json').read_text())
    assert combined['exact_disk_replay'] and combined['hidden_input_perturbation_all_paths_exact'] and combined['teacher_spatial_variance']>1e-8
    for name in ('dofa','anysat'):
        r=json.loads((engineering/f'p1_{name}_trial_v2/summary.json').read_text())
        assert r['exact_disk_replay'] and r['microbatch']==2 and r['formal_updates']==0
    (engineering/'protocol_receipts.json').write_text(json.dumps(receipts,indent=2))
    (engineering/'budget_ledger.json').write_text(json.dumps(dict(formal_updates=0,successful_cpu_optimizer_updates=30,
        successful_cuda_optimizer_updates=19,total_engineering_optimizer_updates=49,
        cpu_runs={'first_5_tests':3,'second_6_tests':3,'first_8_tests':6,'tests_v5':6,'tests_v6':6,'tests_final':6},
        cuda_runs={'p1_dofa_trial':2,'p1_anysat_trial':2,'p1_dofa_trial_v2_including_replay':3,'p1_anysat_trial_v2_including_replay':3,
                   'pipeline_failed_optimizer_comparison':3,'pipeline_v2':3,'pipeline_final':3},
        maximum_successful_updates_per_configuration_below=100,
        accounting='Repeated tests and successful updates before the CPU/GPU optimizer-comparison assertion failure included; zero-step native/modality/geometry checks and backward-only capacity separate.'),indent=2))
    paths=set()
    for directory in ('scripts/use_next_v2','src/landslide_seg/models/use_next_v2','configs/use_next_v2','tests/use_next_v2'):
        paths.update(p for p in (ROOT/directory).rglob('*') if p.suffix in ('.py','.json','.yaml'))
    for module in tuple(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value:
            p=Path(value).resolve()
            if p.is_file() and p.is_relative_to(ROOT) and p.relative_to(ROOT).parts[0] in ('src','scripts') and p.suffix=='.py':paths.add(p)
    paths.add(ROOT/'tests/test_use_next_v2_protocol.py')
    for directory in ('vendor/dofa-official-0cfb7e1','vendor/AnySat-main'):
        paths.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and (p.suffix=='.py' or p.name=='LICENSE'))
    for directory in ('outputs/use_next_v2/data/data_release_v1','outputs/use_next_v2/planning/data_snapshot'):
        paths.update(p for p in (ROOT/directory).iterdir() if p.is_file())
    paths.add(ROOT/'outputs/use_next_v2/data/schedule_seed1234.jsonl')
    for name in ('native_dofa.json','native_anysat.json','modalities.json','tests_final.xml','u0.pt','protocol_receipts.json','budget_ledger.json',
                 'pipeline_cuda_0_final/acceptance.json','p1_dofa_trial_v2/summary.json','p1_anysat_trial_v2/summary.json',
                 'p1_dofa_trial_v2/curve.jsonl','p1_anysat_trial_v2/curve.jsonl'):
        paths.add(engineering/name)
    hashes={}
    for path in sorted(paths):
        relative=path.relative_to(ROOT);dest=target/'snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        hashes[relative.as_posix()]=sha(dest)
    environment=dict(python=sys.version,platform=platform.platform(),dependencies={name:importlib.metadata.version(name) for name in
        ('torch','numpy','scipy','PyYAML','rasterio','pyproj','shapely','timm','transformers')},
        gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv'],text=True))
    acceptance=dict(status='READY_FOR_INDEPENDENT_AUDIT',independent_audit_ready=True,formal_updates=0,blockers=[],
        tests='10 CPU tests plus real native numerical checks, eight modality/subset forwards, two P1 trials/replays, combined CUDA replay/mask/capacity',
        independently_pending=['cuda:1 same release reproduction and capacity','data radiometry/provenance and 3829 conservative TRAIN exclusions review','complete source and environment binding verification'],
        limitations=['Effective instrument resolution remains explicitly unknown; no high-frequency target may be inferred from interpolated caches.',
                     'DOFA loads all173 published encoder keys; official dense inference intentionally does not consume mask/norm/projector. No MAE decoder is claimed.',
                     'AnySat native SAR/Landsat paths blocked by missing product/channel compatibility; independent metric paths remain valid.',
                     '256 capacity uses synthetic legal 10m tensors; not a real 256 crop or research result.',
                     'P2/P3/P5 training entrypoints NOT_IMPLEMENTED, P4 skipped for unverified pairs, external task generalization unverified.'],
        engineering_optimizer_updates=49,p1_commands={name:f'.venv-use-independent/Scripts/python.exe scripts/use_next_v2/run_p1.py --config configs/use_next_v2/p1_{name}.yaml --device cuda:{i} --out outputs/use_next_v2/runs/p1_{name} --release {target.relative_to(ROOT).as_posix()} --ready-p0 outputs/use_next_v2/audit/READY_P0.json' for i,name in enumerate(('dofa','anysat'))},
        audit_cuda_command='.venv-use-independent/Scripts/python.exe scripts/use_next_v2/check_pipeline.py cuda:1 outputs/use_next_v2/audit/pipeline_cuda_1')
    (target/'P0_ACCEPTANCE.json').write_text(json.dumps(acceptance,indent=2))
    weights={p:d for p,d in receipts['dofa']['use_next_v2'].get('weights',{}).items()}
    weights={p:sha(ROOT/p) for p in ('weights/dofa/XShadow_DOFA_base/DOFA_ViT_base_e100.pth','weights/AnySat_base.pth')}
    manifest=dict(status='READY_FOR_INDEPENDENT_AUDIT',files=hashes,weights=weights,environment=environment,
                  acceptance_sha256=sha(target/'P0_ACCEPTANCE.json'))
    (target/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
    (target/'release_manifest.sha256').write_text(sha(target/'release_manifest.json'))
    print(target);print(sha(target/'release_manifest.json'))


if __name__=='__main__':main()
