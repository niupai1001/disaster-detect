"""Refine only timed-out filename subtrees, retaining all partial/error evidence."""
import argparse
from collections import deque,Counter
import hashlib,json
from pathlib import Path
from .discover_rasters import scan
from .data import sha256


def children_for_refinement(root):
    work=[];skipped=[]
    for path in sorted(Path(root).iterdir()):
        try:
            if path.is_symlink() or path.is_junction():skipped.append(dict(root=str(path),status='LINK_NOT_FOLLOWED'))
            elif path.is_dir():work.append(str(path))
        except OSError as e:skipped.append(dict(root=str(path),status='CHILD_METADATA_ERROR',error=str(e)))
    return work,skipped


def main(args):
    args.output_dir.mkdir(parents=True,exist_ok=True);pending=deque((str(Path(p).resolve()),0,None) for p in args.targets);rows=[]
    report=dict(status='REFINING_NAMES_ONLY',targets=[str(Path(p).resolve()) for p in args.targets],roots=rows,imagery_or_header_reads=0,optimizer_updates=0,admitted_members=0,per_chunk_timeout_seconds=45,max_refinement_depth=3)
    def save():
        tmp=args.output_dir/'summary.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(args.output_dir/'summary.json')
    save()
    while pending:
        path,depth,max_depth=pending.popleft();key=hashlib.sha256(json.dumps([path,max_depth]).encode()).hexdigest()[:20];directory=args.output_dir/key;detail=directory/'scan.json'
        if directory.exists():
            if not detail.exists():raise ValueError('Uncommitted chunk must be inspected before replay: '+str(directory))
            row=json.loads(detail.read_text(encoding='utf-8'))
            if row['root']!=path or sha256(row['filename_list'])!=row['filename_list_sha256']:raise ValueError('Chunk identity mismatch')
        else:
            directory.mkdir();row=scan(path,directory,timeout_seconds=45,max_depth=max_depth);row.update(refinement_depth=depth,max_depth=max_depth);detail.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        row=dict(row,detail_report=str(detail.resolve()));row.pop('parent_directories',None);rows.append(row)
        if row['status']=='PARTIAL_SCAN_TIMEOUT' and max_depth is None and depth<3:
            children,skipped=children_for_refinement(path);rows.extend(skipped)
            if children:
                row['refinement_children']=children;pending.append((path,depth,1));pending.extend((child,depth+1,None) for child in children)
        report.update(pending_chunks=len(pending),counts=dict(Counter(x['status'] for x in rows)));save();print(json.dumps(dict(path=path,status=row['status'],files=row['files'],pending=len(pending)),ensure_ascii=True),flush=True)
    report['status']='FILENAME_REFINEMENT_FINISHED_NOT_ADMISSION';save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--targets',nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True);main(p.parse_args())
