"""Isolated optional P2 consumer repair entry; never replaces a live job or source."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import yaml
import json
import torch
from use_next_p2 import run as engine
from use_next_p2.protocol import sha
from landslide_seg.models.use_next_p3.readout import GuardedReadout


def check_dependencies(values,marker,release):
    manifest=json.loads((Path(release)/'manifest.json').read_text(encoding='utf8'))
    parent=manifest['parent_release_sha256']
    if marker.get('parent_release_sha256')!=parent:raise ValueError('Guard parent release mismatch')
    ready=ROOT/'outputs/use_next_v2/audit/READY_P2.json'
    previous=json.loads(ready.read_text(encoding='utf8'))
    if previous.get('status')!='READY_P2' or previous.get('release_sha256')!=parent or sha(ready)!=marker.get('ready_p2_sha256'):
        raise ValueError('Original P2 acceptance not bound')
    source=ROOT/values['frozen_source'];saved=torch.load(source,map_location='cpu',weights_only=False)
    summary=json.loads((source.parent/'summary.json').read_text(encoding='utf8'))
    if saved['step']!=6000 or not summary['complete'] or summary['updates']!=6000 or saved['identity']!=summary['identity']:
        raise ValueError('Incomplete formal6000 source')
    identity=saved['identity']
    if identity['engineering'] or identity.get('release_sha256')!=parent or identity['mode']!=values['mode'] or identity['stage']!='P2' or identity['task']!='pretrain':
        raise ValueError('Wrong source lineage')
    expected=ROOT/f"configs/use_next_p2/{values['mode']}_pretrain.yaml"
    if identity['config_sha256']!=sha(expected):raise ValueError('Source config differs')
    local={k[len('use.local.'):]:v for k,v in saved['model'].items() if k.startswith('use.local.')}
    if engine.tensor_digest(local)!=summary['final_local_sha256']:raise ValueError('Frozen source parameters changed')


if __name__=='__main__':
    config=sys.argv[sys.argv.index('--config')+1]
    values=yaml.safe_load(Path(config).read_text(encoding='utf8'))['use_next_p2']
    if values['task']!='head':raise ValueError('Guarded repair entry is for frozen P2 consumers only')
    if '--engineering' not in sys.argv:
        if '--source' in sys.argv:raise ValueError('Formal guard uses only the configured same-mode endpoint')
        flag='--marker' if '--marker' in sys.argv else '--ready-p2'
        marker=json.loads(Path(sys.argv[sys.argv.index(flag)+1]).read_text(encoding='utf8'))
        check_dependencies(values,marker,sys.argv[sys.argv.index('--release')+1])
    engine.ScaleReadout=GuardedReadout
    engine.main(marker_status='READY_P2_HEAD')
