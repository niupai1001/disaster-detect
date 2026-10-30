"""Verify code, data, environment and inherited immutable releases without mutation."""
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))

from use_next_p2.protocol import ROOT,sha


def verify(release):
    release=Path(release);path=release/'manifest.json';manifest=json.loads(path.read_text())
    if manifest['status']!='READY_FOR_INDEPENDENT_AUDIT':raise ValueError('Incomplete engineering release')
    if manifest['environment']['python']!=sys.version:raise ValueError('Python changed')
    for name,version in manifest['environment']['dependencies'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Environment changed: '+name)
    for name,expected in manifest['files'].items():
        if sha(ROOT/name)!=expected or sha(release/'snapshot'/name)!=expected:raise ValueError('Release-bound file changed: '+name)
    parent=ROOT/manifest['parent_release']
    parent_path=parent/manifest.get('parent_manifest_name','release_manifest.json')
    if sha(parent_path)!=manifest['parent_release_sha256']:raise ValueError('Inherited release changed')
    previous=json.loads(parent_path.read_text())
    for name,expected in previous['files'].items():
        if sha(ROOT/name)!=expected or sha(parent/'snapshot'/name)!=expected:raise ValueError('Inherited dependency changed: '+name)
    for name,expected in manifest['weights'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Weight changed: '+name)
    return sha(path)


if __name__=='__main__':
    # Namespace execution also works when this script is invoked directly.
    print(verify(sys.argv[1]))
