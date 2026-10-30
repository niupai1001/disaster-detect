"""Fail closed: partial engineering artifacts cannot become training authorization."""
import json
import sys
import importlib.metadata
from pathlib import Path

from audit_data import ROOT,sha


def verify(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'release_manifest.json').read_text())
    environment=manifest['environment']
    if environment.get('python')!=sys.version:raise ValueError('Python environment changed')
    for name,expected in environment.get('dependencies',{}).items():
        if importlib.metadata.version(name)!=expected:raise ValueError('Dependency environment changed: '+name)
    for name,expected in manifest['files'].items():
        if sha(directory/'snapshot'/name)!=expected: raise ValueError('Snapshot changed: '+name)
        if sha(ROOT/name)!=expected: raise ValueError('Working release file changed: '+name)
    for name,expected in manifest['weights'].items():
        if sha(ROOT/name)!=expected: raise ValueError('Weight changed: '+name)
    for name,expected in manifest.get('vendor_hashes',{}).items():
        if sha(ROOT/name)!=expected: raise ValueError('Vendor changed: '+name)
    acceptance=directory/'P0_ACCEPTANCE.json'
    if sha(acceptance)!=manifest['acceptance_sha256']: raise ValueError('Acceptance changed')
    a=json.loads(acceptance.read_text())
    return dict(integrity_pass=True,training_allowed=False,independent_audit_ready=a.get('independent_audit_ready',False),
                status=a['status'],blockers=a.get('blockers',[]),formal_updates=0)


if __name__=='__main__':
    result=verify(sys.argv[1]); print(json.dumps(result,indent=2))
    sys.exit(0 if result['training_allowed'] else 2)
