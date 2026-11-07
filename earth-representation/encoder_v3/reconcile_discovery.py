"""Reconcile hashed filename lists without opening any candidate imagery."""
import argparse
from collections import Counter
import hashlib,json,ntpath
from pathlib import Path
import re
from .data import sha256


OPTICAL_HINT=re.compile(r'sentinel.?2|s2[abc]_|msil2[ac]|landsat|lc0[89]|enmap|hyperspec|aviris|prisma|desis|gaofen|worldview|pleiades|uav',re.I)


def reconcile(summaries,known_source_files=None):
    known=json.loads(Path(known_source_files).read_text(encoding='utf-8')) if known_source_files is not None else {}
    known_exact={};known_folded={};known_matched=set();known_case_matched=set()
    for name in known:
        normalized=ntpath.normpath(name);drive,tail=ntpath.splitdrive(normalized);normalized=drive.upper()+tail
        known_exact.setdefault(hashlib.sha256(normalized.encode()).digest(),[]).append(name)
        known_folded.setdefault(hashlib.sha256(normalized.casefold().encode()).digest(),[]).append(name)
    seen=set();folded={};counts=Counter();extensions=Counter();drives=Counter();stages=[];parents=Counter();upper=Counter();case_aliases=0
    for summary_path in summaries:
        summary=json.loads(Path(summary_path).read_text(encoding='utf-8'))
        if summary['status']=='REFINING_NAMES_ONLY':raise ValueError('Refinement is still running; no final reconciliation')
        added=0;source_rows=[]
        for row in summary['roots']:
            if not row.get('filename_list'):continue
            path=Path(row['filename_list']);before=path.stat();digest=hashlib.sha256();raw_count=0;local_upper=0;pending=b''
            with path.open('rb') as stream:
                while block:=stream.read(1024*1024):
                    digest.update(block);parts=(pending+block).split(b'\0');pending=parts.pop()
                    for raw in parts:
                        if not raw:continue
                        name=raw.decode('utf-8',errors='strict');normalized=ntpath.normpath(name);drive,tail=ntpath.splitdrive(normalized);normalized=drive.upper()+tail
                        raw_count+=1;suffix=ntpath.splitext(normalized)[1];local_upper+=int(suffix!=suffix.lower());key=hashlib.sha256(normalized.encode()).digest()
                        if key in known_exact:known_matched.update(known_exact[key])
                        else:known_case_matched.update(known_folded.get(hashlib.sha256(normalized.casefold().encode()).digest(),[]))
                        if key in seen:continue
                        seen.add(key);fold=hashlib.sha256(normalized.casefold().encode()).digest()
                        if fold in folded and folded[fold]!=key:case_aliases+=1
                        folded.setdefault(fold,key);added+=1;extensions[suffix.lower()]+=1;drives[drive.upper()]+=1
                        hint=bool(OPTICAL_HINT.search(normalized));kind='optical_name_hint_not_admitted' if hint else 'unclassified_candidate';counts[kind]+=1
                        if hint:parents[ntpath.dirname(normalized)]+=1
            after=path.stat()
            if pending:raise ValueError('Incomplete NUL filename record: '+str(path))
            if digest.hexdigest()!=row['filename_list_sha256'] or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Filename snapshot changed: '+str(path))
            upper[str(summary_path)]+=local_upper;source_rows.append(dict(root=row['root'],status=row['status'],raw_names=raw_count,uppercase_extension_names=local_upper,filename_list=str(path),sha256=digest.hexdigest(),error_bytes=row.get('error_bytes',0)))
        stages.append(dict(summary=str(summary_path),summary_sha256=sha256(summary_path),source_status=summary['status'],added_exact_normalized_paths=added,lists=source_rows))
    return dict(status='RECONCILED_FILENAME_CANDIDATES_NOT_ADMISSION',unique_exact_normalized_paths=len(seen),casefold_alias_collisions_retained=case_aliases,known_source_name_join=dict(snapshot_sha256=None if known_source_files is None else sha256(known_source_files),total_paths=len(known),exact_matches=len(known_matched),additional_case_variant_matches=len(known_case_matched-known_matched),not_found_paths=sorted(set(known)-known_matched-known_case_matched),scope='Filename presence only, not a reread or new validation of source bytes'),counts=dict(counts),extensions=dict(extensions),drives=dict(drives),stages=stages,optical_name_hint_parent_directories=dict(parents),optimizer_updates=0,imagery_or_header_reads=0,admitted_members=0,limits=['Filename hints do not prove optical modality, physical reflectance, true SRF, QA or native grid','Exact normalized paths deduplicated; case variants retained pending actual file identity','Historical names may no longer exist; live existence not checked for every path','Junction aliases, identical physical copies and archives are not resolved by lexical path deduplication','Access errors, partial chunks, unfollowed links and missing formats remain explicit scope gaps'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summaries',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--known-source-files',type=Path);args=p.parse_args();report=reconcile(args.summaries,args.known_source_files);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:report[k] for k in ('status','unique_exact_normalized_paths','casefold_alias_collisions_retained','counts')},indent=2))
