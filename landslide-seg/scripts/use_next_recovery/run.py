"""Explicit, hash-bound geometry overlay; original checkpoint identity is retained."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import os
import subprocess
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from use_next_p2 import run as engine
from landslide_seg.models.use_next_p2 import encoder
from use_next_recovery.geometry import area_grid


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--geometry-manifest',required=True)
    ap.add_argument('--recovery-acceptance')
    ap.add_argument('--probe-stop',type=int)
    ap.add_argument('--backend',choices=('original','candidate'),default='candidate')
    args,remaining=ap.parse_known_args()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=json.loads(Path(args.geometry_manifest).read_text(encoding='utf8'))
    for name,digest in manifest['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Overlay source changed: '+name)
    if '--resume' not in remaining or '--engineering' in remaining:raise ValueError('Complete formal checkpoint resume required')
    out=Path(remaining[remaining.index('--out')+1]).resolve()
    saved=engine.torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    if saved['identity']['stage']!='P2' or saved['identity']['mode']!='pixel' or saved['identity']['task']!='pretrain':
        raise ValueError('Only P2 pixel pretrain recovery supported')
    receipt=dict(overlay_manifest_sha256=sha(args.geometry_manifest),backend=args.backend,
                 initial_step=saved['step'],disposable_probe=args.probe_stop is not None,
                 scientific_identity_preserved=True,root_cause='UNCONFIRMED')
    if args.probe_stop is not None:
        if out.parent!=ROOT/'outputs/use_next_v2/geometry_recovery' or not saved['step']<args.probe_stop<=1680:
            raise ValueError('Probe must be a disposable copy ending by u1680')
        original_save=engine.atomic_save
        def bounded_save(value,path):
            original_save(value,path)
            if Path(path).name=='last.pt' and value['step']>=args.probe_stop:
                (out/'overlay_probe_completed.json').write_text(json.dumps(dict(receipt,final_step=value['step']),indent=2))
                raise SystemExit(0)
        engine.atomic_save=bounded_save
        @contextmanager
        def probe_claim(device):
            lock=out/'disposable_probe.lock'
            with lock.open('x') as stream:json.dump(dict(pid=os.getpid(),device=device),stream)
            try:
                yield subprocess.check_output(['nvidia-smi','--id='+device.split(':')[1],
                    '--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            finally:lock.unlink()
        engine.claim=probe_claim
    else:
        if args.backend!='candidate' or not args.recovery_acceptance:raise ValueError('Independent overlay acceptance required')
        acceptance=json.loads(Path(args.recovery_acceptance).read_text(encoding='utf8'))
        if acceptance.get('status')!='GEOMETRY_OVERLAY_ACCEPTED' or acceptance.get('overlay_manifest_sha256')!=receipt['overlay_manifest_sha256']:
            raise ValueError('Independent recovery binding mismatch')
        if out!=ROOT/'outputs/use_next_v2/runs/p2_pixel':raise ValueError('Wrong formal output')
    (out/'geometry_overlay_provenance.json').write_text(json.dumps(receipt,indent=2))
    if args.backend=='candidate':encoder.area_grid=area_grid
    encoder._geometry.clear()
    sys.argv=[sys.argv[0]]+remaining
    engine.main()


if __name__=='__main__':main()
