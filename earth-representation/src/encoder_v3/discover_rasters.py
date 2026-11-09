"""Read-only mounted-drive filename discovery; never optical admission."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
from .data import sha256


RASTERS=('.tif','.tiff','.jp2','.hdf','.h5','.he5','.hdf5','.nc','.img','.hdr','.bsq','.bil','.bip','.npy','.npz')
ARCHIVES=('.zip','.tar','.tgz','.7z','.gz')


def scan(root,directory,*,timeout_seconds=900,max_depth=None):
    raw=directory/(root[0]+'_filenames.nul');errors=directory/(root[0]+'_scan_errors.txt');started=time.perf_counter()
    command=['rg','--files','--hidden','--no-ignore','--null','--threads','1']
    if max_depth is not None:command+=['--max-depth',str(max_depth)]
    for suffix in RASTERS+ARCHIVES:command+=['--iglob','*'+suffix]
    command+=[root]
    with raw.open('xb') as output,errors.open('xb') as stderr:
        child=subprocess.Popen(command,stdout=output,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        try:code=child.wait(timeout=timeout_seconds);status='FINISHED_WITH_ACCESS_ERRORS' if code==2 else 'FINISHED' if code in (0,1) else 'SCAN_FAILED'
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            code=child.returncode;status='PARTIAL_SCAN_TIMEOUT'
    counts=Counter();parents=Counter();files=0
    with raw.open('rb') as stream:
        pending=b''
        while block:=stream.read(1024*1024):
            parts=(pending+block).split(b'\0');pending=parts.pop()
            for value in parts:
                path=Path(value.decode('utf-8',errors='replace'));counts[path.suffix.lower()]+=1;parents[str(path.parent)]+=1;files+=1
        if pending:raise ValueError('Incomplete filename record')
    return dict(root=root,status=status,exit_code=code,command=command,seconds=time.perf_counter()-started,files=files,extensions=dict(counts),parent_directories=dict(parents),filename_list=str(raw.resolve()),filename_list_sha256=sha256(raw),error_log=str(errors.resolve()),error_bytes=errors.stat().st_size)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--roots',nargs='+',required=True);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--chunked',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if len(set(x[0].upper() for x in args.roots))!=len(args.roots) or any(len(x)!=3 or x[1:]!=':\\' for x in args.roots):raise ValueError('Unique Windows drive roots required')
    if args.resume and not args.chunked:raise ValueError('Only durable chunked scans support resume')
    args.output_dir.mkdir(exist_ok=args.resume,parents=True)
    report=dict(status='SCANNING_NAMES_ONLY',roots=[],raster_extensions=RASTERS,archive_extensions=ARCHIVES,imagery_reads=0,admitted_members=0,limitations=['Filenames are candidates only; archives are not opened or extracted','Access errors, unmounted volumes and unfollowed links remain explicit coverage gaps','Generic photographs and package formats outside the listed extensions are not enumerated','No reflectance, SRF, QA, role or split eligibility inferred from extension/name'])
    for root in args.roots:
        if args.chunked:
            try:children=sorted(Path(root).iterdir())
            except OSError as exc:
                report['roots'].append(dict(root=root,status='ROOT_ENUMERATION_ERROR',error=str(exc)));continue
            work=[(root,1)]
            for child in children:
                try:
                    if child.is_symlink() or child.is_junction():report['roots'].append(dict(root=str(child),status='LINK_NOT_FOLLOWED'));continue
                    if child.is_dir():work.append((str(child),None))
                except OSError as exc:report['roots'].append(dict(root=str(child),status='CHILD_METADATA_ERROR',error=str(exc)))
            for index,(path,depth) in enumerate(work):
                directory=args.output_dir/(root[0]+'_'+str(index).zfill(4));detail=directory/'scan.json'
                if directory.exists():
                    if not args.resume or not detail.exists():raise ValueError('Uncommitted scan artifacts; inspect before replay')
                    row=json.loads(detail.read_text(encoding='utf-8'))
                    if row['root']!=path or '--iglob' not in row['command'] or sha256(row['filename_list'])!=row['filename_list_sha256']:raise ValueError('Chunk identity changed; no silent reuse')
                else:
                    directory.mkdir();row=scan(path,directory,timeout_seconds=180,max_depth=depth)
                    detail.write_text(json.dumps(row,indent=2),encoding='utf-8')
                report['roots'].append(dict({key:value for key,value in row.items() if key!='parent_directories'},detail_report=str(detail.resolve())))
                if index%25==0 or index==len(work)-1:(args.output_dir/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(json.dumps(dict(path=path,status=row['status'],files=row['files']),ensure_ascii=True),flush=True)
        else:
            row=scan(root,args.output_dir);report['roots'].append(row)
            (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(root,row['status'],row['files'],flush=True)
    report['status']='MOUNTED_DRIVE_FILENAME_DISCOVERY_NOT_ADMISSION'
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
