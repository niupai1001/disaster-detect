"""Full cache/physical-contract reread. This is not V3 training admission."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import time
import torch
from .data import NativeOpticalDataset


def write(path, report):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)


def run(args):
    counts=Counter();shapes=Counter();grids=Counter();failures=[]; groups=defaultdict(set);products=defaultdict(set);start=time.perf_counter()
    report=dict(status='RUNNING',scope='existing hash-bound native cache snapshot, not all-machine admission',optimizer_updates=0,read_members=0,failed=[],roles={},pending_v3=['all-machine source discovery/association','V3 pair registration and temporal-change review','independent calibration/QA source spot checks','training/validation/test and protected-region approval'])
    write(args.output,report)
    torch.set_num_threads(2)
    for role in ('TRAIN','VAL'):
        data=NativeOpticalDataset(args.release,args.contracts,role=role)
        report['source_identities']=data.identities
        for index,row in enumerate(data.rows):
            try:
                item=data[index]
                for band in item.bands:
                    grids[f'{band.affine[0]}x{-band.affine[4]}']+=1
                    shapes['x'.join(map(str,band.reflectance.shape))]+=1
                    if not bool(torch.isfinite(band.wavelengths_nm).all() & torch.isfinite(band.response).all()): raise ValueError('Nonfinite SRF')
                    if not bool(((band.wavelengths_nm[1:]-band.wavelengths_nm[:-1])>0).all() & (band.response>=0).all()) or not bool(torch.trapezoid(band.response,band.wavelengths_nm)>0): raise ValueError('Invalid SRF integration')
                groups[row['release_isolation_group']].add(role);products[row['product']].add(role)
                counts[f'{role}/{row["instrument"]}']+=1
            except (ValueError,OSError,KeyError) as exc:
                failures.append(dict(id=row['id'],role=role,error=str(exc)))
            report.update(read_members=report['read_members']+1,failed=failures,roles=dict(counts),gsd_band_counts=dict(grids),native_shapes=dict(shapes),seconds=time.perf_counter()-start)
            if (index+1)%100==0: write(args.output,report)
    collisions=[key for key,roles in groups.items() if len(roles)>1]
    report.update(status='CACHE_REREAD_PASS_NOT_V3_ADMISSION' if not failures and not collisions else 'CACHE_REREAD_FAILED',isolation_group_collisions=collisions,products_shared_across_roles=sum(len(roles)>1 for roles in products.values()),seconds=time.perf_counter()-start)
    write(args.output,report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--contracts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
