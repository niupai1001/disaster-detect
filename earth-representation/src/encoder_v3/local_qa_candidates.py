"""Discover local QA candidates by exact product aliases, without opening imagery."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse
from .data import sha256


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--members',type=Path,required=True);parser.add_argument('--roots',type=Path,nargs='+',required=True);parser.add_argument('--aerosol-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    before=sha256(args.members)
    with args.members.open(encoding='utf-8-sig') as stream:members=[json.loads(line) for line in stream if line.strip()]
    members=[r for r in members if r['role'] in ('TRAIN','VAL') and r['provenance']=='native_lattice_cache_channels_only']
    aliases=defaultdict(set);by_id={r['id']:r for r in members}
    for row in members:
        aliases[row['product']].add(row['id'])
        for band in row['bands']:
            url=band.get('source_asset')
            if url:aliases[urlparse(url).path.rstrip('/').rsplit('/',2)[-2]].add(row['id'])
    existing=[str(p.resolve()) for p in args.roots if p.is_dir()];missing=[str(p) for p in args.roots if not p.is_dir()]
    command=['rg','--files','--no-ignore','--hidden']
    for pattern in ('*scl*.tif','*scl*.jp2','*qa_pixel*.tif','*qa_radsat*.tif','*qa_aerosol*.tif'):command+=['--iglob',pattern]
    found=subprocess.run(command+existing,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
    if found.returncode not in (0,1,2):raise RuntimeError('Unexpected inventory scan error')
    paths={str(Path(line).resolve()) for line in found.stdout.splitlines() if line.strip()}
    paths.update(str(p.resolve()) for p in args.aerosol_root.glob('*.tif'))
    matching=defaultdict(lambda:defaultdict(list));index=[];errors=[]
    for text in sorted(paths):
        path=Path(text);name=path.name.lower();kind=next((key.upper() for key in ('qa_pixel','qa_radsat','qa_aerosol') if key in name),None)
        if kind is None and 'scl' in name:kind='SCL'
        ids=set()
        for component in path.parts:ids.update(aliases.get(component,()))
        if path.parent.resolve()==args.aerosol_root.resolve() and path.stem in by_id:kind='QA_AEROSOL';ids.add(path.stem)
        if kind is None:continue
        try:stat=path.stat()
        except OSError as exc:errors.append(dict(path=text,error=str(exc)));continue
        index.append(dict(path=text,kind=kind,size=stat.st_size,mtime_ns=stat.st_mtime_ns,matched_member_ids=sorted(ids)))
        for identifier in ids:matching[identifier][kind].append(text)
    rows=[];counts=Counter()
    for member in members:
        required={'SCL'} if member['sensor']=='S2' else {'QA_PIXEL','QA_RADSAT','QA_AEROSOL'}
        present=dict(matching[member['id']]);missing_kinds=sorted(required-set(present))
        counts['candidate_kind_set_complete' if not missing_kinds else 'candidate_kind_set_incomplete']+=1
        rows.append(dict(id=member['id'],role=member['role'],sensor=member['sensor'],product=member['product'],candidate_paths=present,missing_required_kinds=missing_kinds))
    if sha256(args.members)!=before:raise ValueError('Member inventory changed during scan')
    report=dict(status='LOCAL_QA_CANDIDATES_ONLY_NOT_ADMISSION',members_sha256=before,cache_parent_members=len(members),roots=existing,missing_roots=missing,scan_exit_code=found.returncode,scan_warnings=found.stderr,unique_QA_paths=len(index),file_errors=errors,counts=dict(counts),members=rows,files=index,imagery_or_header_reads=0,optimizer_updates=0,limitations=['Exact product/path aliases do not prove matching processing version or spatial coverage','QA grid, role provenance, calibration applicability and mask differences must be checked before using candidates','No source cache, mask, data role, task or admitted member has changed'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as target:json.dump(report,target,ensure_ascii=False,indent=2)
    print(json.dumps({key:report[key] for key in ('status','cache_parent_members','unique_QA_paths','counts','scan_exit_code')},indent=2))
