"""Seal incomplete engineering evidence without granting training readiness."""
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from audit_data import ROOT, sha


def main():
    target=ROOT/'outputs/use_next_v2/releases/p0_partial_v4'
    target.mkdir(parents=True,exist_ok=False)
    paths=[]
    for directory in ('src/landslide_seg/models/use_next_v2','scripts/use_next_v2','configs/use_next_v2','tests/use_next_v2'):
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.suffix in ('.py','.json'))
    paths.extend(ROOT/p for p in ('outputs/use_next_v2/engineering/tests_v3.xml','outputs/use_next_v2/engineering/native_anysat.json','outputs/use_next_v2/data/audit_summary.json',
                                 'outputs/use_next_v2/data/asset_audit.jsonl','outputs/use_next_v2/data/pair_eligibility.jsonl',
                                 'outputs/use_next_v2/planning/data_snapshot/manifest.jsonl',
                                 'outputs/use_next_v2/planning/data_snapshot/pairs.jsonl',
                                 'outputs/use_next_v2/planning/data_snapshot/snapshot_identity.json'))
    hashes={}
    for path in paths:
        relative=path.relative_to(ROOT)
        dest=target/'snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        hashes[relative.as_posix()]=sha(dest)
    weights={}
    for path in (ROOT/'weights/dofa/XShadow_DOFA_base/DOFA_ViT_base_e100.pth',ROOT/'weights/AnySat_base.pth'):
        weights[path.relative_to(ROOT).as_posix()]=sha(path)
    environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,platform=platform.platform(),
                     gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv'],text=True))
    blockers=[
        'Native/effective resolution and resampling lineage not yet recovered from source metadata; no v2 data admission seal issued.',
        'DOFA local full MAE constructor auto-downloads teacher and requires absent weight_generator checkpoint. Available downstream encoder drops norm/projector/mask pretrained keys; cannot replace complete native strict loading.',
        'Shared metric grid and conservative pixel-mask/EMA helpers tested, but end-to-end combined native/local teacher-student pipeline and label-grid reprojection still unverified.',
        'No isolated UUID-locked dual-CUDA 128/256 capacity measurement. CPU local AdamW/scheduler/RNG/cursor disk-resume replay passes, but is not complete production resume acceptance.',
        'Shared protocol preflight and formal stage entrypoints not adapted; formal training must remain blocked.'
    ]
    acceptance=dict(status='BLOCKED',formal_updates=0,engineering_cpu_successful_updates_cumulative=12,
                    engineering_budget_basis='first 5-test suite:3; second 6-test suite:3; current 8-test suite:3 in-memory plus 3 disk replay. Manual EMA parameter perturbation is not an optimizer update. Earlier partial counters undercounted reruns; this supersedes them. All CPU disposable tests; no training scores.',
                    tests='8 passed; CPU local synthetic contracts plus separate real AnySat native equivalence',blockers=blockers,
                    data='13956 caches hash/array checked; no v2-admitted release',
                    pair_eligibility='0 qualified under conservative v2 checks; screening evidence absent',
                    independent_audit_ready=False)
    (target/'P0_ACCEPTANCE.json').write_text(json.dumps(acceptance,indent=2))
    vendor_hashes={}
    for directory in ('vendor/AnySat-main','vendor/dofa-official-0cfb7e1'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and (path.suffix=='.py' or path.name=='LICENSE'):
                vendor_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    manifest=dict(status='BLOCKED',files=hashes,weights=weights,environment=environment,vendor_hashes=vendor_hashes,
                  acceptance_sha256=sha(target/'P0_ACCEPTANCE.json'))
    (target/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(target)
    print(sha(target/'release_manifest.json'))


if __name__=='__main__': main()
