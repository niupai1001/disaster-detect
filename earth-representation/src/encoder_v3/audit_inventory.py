"""Read-only metadata inventory; never converts candidate pairs into admission."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def audit(manifest):
    before=digest(manifest); counts=Counter(); duplicates=0; ids=set()
    missing_geometry=0; photographic=0; sources=Counter()
    with manifest.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip(): continue
            row=json.loads(line); counts[(row.get('role','UNKNOWN'),row.get('sensor','UNKNOWN'))]+=1
            identifier=row.get('id')
            if identifier in ids: duplicates+=1
            ids.add(identifier)
            units=str(row.get('units','UNKNOWN'));sources[units]+=1
            photographic+=int('photographic' in units.lower() or 'not_reflectance' in units.lower())
            missing_geometry+=int(not row.get('crs') or not row.get('transform'))
    if digest(manifest)!=before: raise RuntimeError('Inventory changed during audit; no stable snapshot')
    return dict(path=str(manifest.resolve()),sha256=before,role_sensor_counts={'/'.join(k):v for k,v in counts.items()},units=dict(sources),duplicate_ids=duplicates,missing_geometry=missing_geometry,photographic_not_reflectance=photographic,status='METADATA_ONLY_NOT_ADMITTED',remaining=['source calibration verification','real SRF join and file hashes','QA and per-pixel validity','geographic/acquisition split leakage','pair registration and temporal change risk','native arrays and cache source equivalence'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=audit(a.manifest);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
