"""Refine only timed-out filename subtrees, retaining all partial/error evidence."""
import argparse
from collections import deque,Counter
import hashlib,json,math,os,time
from pathlib import Path
from .discover_rasters import scan
from .data import sha256


def children_for_refinement(root,*,deadline=None):
    work=[];skipped=[]
    with os.scandir(root) as entries:
        for entry in entries:
            if deadline is not None and time.perf_counter()>=deadline:raise TimeoutError('Directory enumeration reached diagnostic time limit')
            path=Path(entry.path)
            try:
                if entry.is_symlink():skipped.append(dict(root=str(path),status='LINK_NOT_FOLLOWED'))
                elif entry.is_dir(follow_symlinks=False):
                    if path.is_junction():skipped.append(dict(root=str(path),status='LINK_NOT_FOLLOWED'))
                    else:work.append(str(path))
            except OSError as exc:skipped.append(dict(root=str(path),status='CHILD_METADATA_ERROR',error=str(exc)))
    return sorted(work),sorted(skipped,key=lambda row:row['root'])


def initial_work(args):
    parent_path=getattr(args,'pending_from',None)
    if parent_path is None:
        targets=[str(Path(p).resolve()) for p in args.targets]
        return targets,[(p,0,None) for p in targets],None
    parent_path=Path(parent_path);blob=parent_path.read_bytes();parent=json.loads(blob)
    if parent.get('status')!='FILENAME_REFINEMENT_STOPPED_LIMIT_NOT_ADMISSION':raise ValueError('Committed terminal pending summary required')
    work=parent.get('pending_work')
    if not isinstance(work,list) or not work or len(work)!=parent.get('pending_chunks'):raise ValueError('Complete nonempty pending work list required')
    if args.output_dir.exists() or args.output_dir.resolve()==parent_path.parent.resolve():raise ValueError('Resume must use a new output directory; preserve parent evidence')
    targets=parent['targets'];roots=[Path(p).resolve() for p in targets];pending=[]
    for item in work:
        path=Path(item['root']).resolve();depth=item['depth'];limit=item['max_depth']
        if type(depth) is not int or not 0<=depth<=3 or limit not in (None,1):raise ValueError('Invalid pending refinement depth')
        if not any(path==r or path.is_relative_to(r) for r in roots):raise ValueError('Pending path outside original audit roots')
        pending.append((str(path),depth,limit))
    if len(set(pending))!=len(pending):raise ValueError('Duplicate pending work records')
    if hashlib.sha256(parent_path.read_bytes()).digest()!=hashlib.sha256(blob).digest():raise ValueError('Parent summary changed during read')
    return targets,pending,dict(path=str(parent_path.resolve()),sha256=hashlib.sha256(blob).hexdigest(),pending_chunks=len(pending))


def main(args):
    started=time.perf_counter();limit=getattr(args,'max_seconds',None)
    if limit is not None and (not math.isfinite(limit) or limit<=0):raise ValueError('Positive finite diagnostic time limit required')
    deadline=None if limit is None else started+limit
    targets,work,parent=initial_work(args)
    args.output_dir.mkdir(parents=True,exist_ok=True);pending=deque(work);rows=[]
    report=dict(status='REFINING_NAMES_ONLY',targets=targets,parent_pending_summary=parent,source_sha256=sha256(Path(__file__)),roots=rows,imagery_or_header_reads=0,optimizer_updates=0,admitted_members=0,per_chunk_timeout_seconds=45,max_refinement_depth=3,max_seconds=limit)
    def save():
        tmp=args.output_dir/'summary.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(args.output_dir/'summary.json')
    save()
    stopped=False
    while pending:
        if deadline is not None and time.perf_counter()>=deadline:stopped=True;break
        path,depth,max_depth=pending.popleft();key=hashlib.sha256(json.dumps([path,max_depth]).encode()).hexdigest()[:20];directory=args.output_dir/key;detail=directory/'scan.json'
        if directory.exists():
            if not detail.exists():raise ValueError('Uncommitted chunk must be inspected before replay: '+str(directory))
            row=json.loads(detail.read_text(encoding='utf-8'))
            if row['root']!=path or sha256(row['filename_list'])!=row['filename_list_sha256']:raise ValueError('Chunk identity mismatch')
        else:
            directory.mkdir();row=scan(path,directory,timeout_seconds=45 if deadline is None else min(45,max(.01,deadline-time.perf_counter())),max_depth=max_depth);row.update(refinement_depth=depth,max_depth=max_depth);detail.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        row=dict(row,detail_report=str(detail.resolve()));row.pop('parent_directories',None);rows.append(row)
        if row['status']=='PARTIAL_SCAN_TIMEOUT' and max_depth is None and depth<3:
            try:children,skipped=children_for_refinement(path,deadline=deadline)
            except TimeoutError:
                pending.appendleft((path,depth,max_depth));report['directory_enumeration_timed_out_root']=path;stopped=True;break
            rows.extend(skipped)
            if children:
                row['refinement_children']=children;pending.append((path,depth,1));pending.extend((child,depth+1,None) for child in children)
        report.update(pending_chunks=len(pending),counts=dict(Counter(x['status'] for x in rows)));save();print(json.dumps(dict(path=path,status=row['status'],files=row['files'],pending=len(pending)),ensure_ascii=True),flush=True)
    report['status']='FILENAME_REFINEMENT_STOPPED_LIMIT_NOT_ADMISSION' if stopped else 'FILENAME_REFINEMENT_FINISHED_NOT_ADMISSION'
    report.update(pending_chunks=len(pending),pending_work=[dict(root=path,depth=depth,max_depth=max_depth) for path,depth,max_depth in pending],seconds=time.perf_counter()-started,counts=dict(Counter(x['status'] for x in rows)));save()


if __name__=='__main__':
    p=argparse.ArgumentParser();selection=p.add_mutually_exclusive_group(required=True);selection.add_argument('--targets',nargs='+');selection.add_argument('--pending-from',type=Path);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--max-seconds',type=float);main(p.parse_args())
