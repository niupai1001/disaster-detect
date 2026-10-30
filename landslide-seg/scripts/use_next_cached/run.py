"""Isolated P3/P5 pure-geometry caching and explicit user-stop prerequisite."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from use_next_cached import engine
from use_next_p3.protocol import audit
from landslide_seg.models.use_next_p3.tasks import LatentTask
from landslide_seg.models.use_next_p3.readout import GuardedReadout
from landslide_seg.models.use_next_p2 import encoder
from use_next_perf.geometry_cache import GeometryMemo
from use_next_perf.disk_geometry import DiskGeometry
import yaml


if __name__=='__main__':
    engine.np.random.seed(1234)
    config=sys.argv[sys.argv.index('--config')+1]
    values=yaml.safe_load(Path(config).read_text(encoding='utf8'))['use_next_p3']
    engineering='--engineering' in sys.argv
    no_cache='--no-cache-probe' in sys.argv
    if no_cache:
        if not engineering:raise ValueError('Formal workflow requires accepted geometry cache')
        sys.argv.remove('--no-cache-probe')
    cache=ROOT/'outputs/use_next_v2/cached_p3/geometry_cache'
    if engineering:probe_out=Path(sys.argv[sys.argv.index('--out')+1]);cache=probe_out.parent/(probe_out.name+'_cache')
    if not no_cache:
        original=encoder.area_grid
        digest=hashlib.sha256((ROOT/'src/landslide_seg/models/use_next_p2/geometry.py').read_bytes()).hexdigest()
        encoder.area_grid=GeometryMemo(DiskGeometry(original,cache,digest))
    if engineering:
        out=Path(sys.argv[sys.argv.index('--out')+1])
        @contextmanager
        def probe_claim(device):
            lock=out/'engineering_probe.lock'
            with lock.open('x') as f:json.dump(dict(pid=os.getpid(),device=device),f)
            try:yield subprocess.check_output(['nvidia-smi','--id='+device.split(':')[1],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            finally:lock.unlink()
        engine.claim=probe_claim
    if values['task']=='head':
        if not engineering:
            from use_next_p3.run import check_source
            check_source(config,values,sys.argv[sys.argv.index('--release')+1])
            if values['source_objective']!='joint':
                marker=json.loads(Path(sys.argv[sys.argv.index('--marker')+1]).read_text())
                for name,field,statuses in [('P4_DECISION','p4_decision_sha256',('P4_SKIPPED','P4_ACCEPTED')),('P5_JOINT_HEAD_ACCEPTANCE','joint_head_acceptance_sha256',('P5_JOINT_HEAD_ACCEPTED',))]:
                    p=ROOT/f'outputs/use_next_v2/audit/{name}.json'
                    if json.loads(p.read_text()).get('status') not in statuses or engine.sha(p)!=marker.get(field):raise ValueError('P5 prerequisite mismatch')
        engine.ScaleReadout=GuardedReadout
    engine.main(task_factory=lambda root,v:LatentTask(root,v['mode'],v['objective']),phase_key='use_next_p3',audit_fn=audit,
                marker_status='READY_P3_CACHED' if values['task']=='pretrain' else 'READY_P5_CACHED',source_steps=10000)
