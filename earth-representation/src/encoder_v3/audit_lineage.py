"""Reproduce historical cache construction from local parents, never admit data."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from .audit_native import write
from .data import NativeOpticalDataset,sha256


def aerosol_mask(qa):
    """Historical mask reproduced for lineage; not a new V3 QA-policy decision."""
    qa=qa.astype(np.uint16)
    high=((qa>>6)&3)==3
    return ((qa&1)==0)&~(high&(((qa&2)==0)|((qa&32)!=0)))


def quality_mask(name,value,valid,sensor):
    name=Path(name).name.lower()
    if sensor=='S2' and name in ('scl.tif','scl.tiff'):return valid&~np.isin(value,(0,1,3,8,9,10,11,255))
    if sensor in ('L8','L9'):
        if name=='qa_pixel.tif':return valid&((value.astype(np.uint16)&63)==0)
        if name=='qa_radsat.tif':return valid&(value==0)
        if name=='qa_aerosol.tif':return valid&aerosol_mask(value)
    raise ValueError('Unknown historical QA source: '+name)


def audit_member(row,canonical,release):
    if row['role'] not in ('TRAIN','VAL'):raise ValueError('Protected role: no imagery access')
    if row.get('source_role') not in ('TRAIN','VAL') or (row['role']=='TRAIN' and row['source_role']!='TRAIN'):
        raise ValueError('Protected source cannot become TRAIN')
    identities={}
    def identity(path,expected=None):
        path=Path(path);before=path.stat();actual=sha256(path);after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during hashing')
        if expected is not None and actual!=expected:raise ValueError('Parent/source SHA256 mismatch: '+str(path))
        identities[str(path.resolve())]=dict(sha256=actual,size=before.st_size,mtime_ns=before.st_mtime_ns)
    identity(row['cache'],row['cache_sha256'])
    qa=[];raw_qa=False;provenance=row['provenance'];parent=None
    if provenance=='native_lattice_cache_channels_only':
        identity(row['source_cache'],row['source_cache_sha256'])
        with np.load(row['source_cache'],allow_pickle=False) as source:
            parent=dict(x=source['x'].copy(),valid=source['pixel_valid'].copy(),metadata=source['metadata'].copy())
        if parent['x'].dtype!=np.float32 or parent['valid'].dtype!=np.bool_:raise ValueError('Parent cache dtype mismatch')
        if row['sensor'] in ('L8','L9'):
            path=Path(release)/'raw_QA'/(row['id']+'.tif');identity(path)
            with rasterio.open(path) as src:qa=[(path,aerosol_mask(src.read(1))&(src.dataset_mask()>0),src.transform,src.crs)]
    elif provenance in ('existing_native_TIF_with_QA','new_native_Landsat_existing_local_anchor'):
        if provenance=='existing_native_TIF_with_QA':
            if canonical is None:raise ValueError('Missing canonical source/QA association')
            if canonical.get('effective_partition') not in ('TRAIN','VAL') or (row['role']=='TRAIN' and canonical['effective_partition']!='TRAIN'):raise ValueError('Protected canonical imagery')
            paths=[Path(p) for p in canonical['quality_paths']]
        else:
            folder=Path(row['bands'][0]['source']).parent
            paths=[folder/(name+'.tif') for name in ('qa_pixel','qa_radsat','qa_aerosol')]
        required={'scl.tif'} if row['sensor']=='S2' else {'qa_pixel.tif','qa_radsat.tif','qa_aerosol.tif'}
        if not required.issubset({p.name.lower() for p in paths}):raise ValueError('Missing full historical QA set')
        for path in paths:
            identity(path)
            with rasterio.open(path) as src:
                qa.append((path,quality_mask(path,src.read(1),src.dataset_mask()>0,row['sensor']),src.transform,src.crs))
        raw_qa=True
    else:raise ValueError('Unknown provenance; no inferred parent')
    bands=[]
    with np.load(row['cache'],allow_pickle=False) as cache:
        for descriptor in row['bands']:
            key=descriptor['band_key'];shape=tuple(descriptor['shape']);affine=rasterio.Affine(*descriptor['transform']);crs=rasterio.crs.CRS.from_user_input(descriptor['crs'])
            if parent is not None:
                channel=descriptor['source_cache_channel'];values=parent['x'][channel].copy();valid=parent['valid'][channel].copy()
                if shape!=values.shape or abs(float(parent['metadata'][channel,2])-descriptor['native_pixel_support_m'][0])>1e-5:raise ValueError('Parent channel is not on its declared native support')
                if not descriptor.get('cached_already_scaled'):raise ValueError('Parent scaling contract missing')
            else:
                path=Path(descriptor['source']);expected=None
                if canonical is not None and provenance=='existing_native_TIF_with_QA':
                    source=canonical['bands'][key]
                    if Path(source['path'])!=path:raise ValueError('Canonical source path differs')
                    expected=source.get('expected_sha256')
                identity(path,expected)
                with rasterio.open(path) as src:
                    if src.transform!=affine or src.crs!=crs or src.shape!=shape:raise ValueError('Reflectance native grid changed')
                    dn=src.read(1);valid=(src.dataset_mask()>0)&np.isfinite(dn)&(dn!=0)
                offset=descriptor.get('prepared_declared_offset',descriptor['offset'])
                values=dn.astype(np.float32)*descriptor['scale']+offset
            for path,mask,transform,qa_crs in qa:
                same_grid=transform==affine and qa_crs==crs and mask.shape==shape
                if parent is not None or provenance=='new_native_Landsat_existing_local_anchor':
                    if not same_grid:raise ValueError('Native Landsat QA grid mismatch')
                    valid &= mask
                elif same_grid:valid &= mask
                else:
                    # Historical categorical QA alignment only; reflectance is never warped.
                    aligned=np.zeros(shape,np.uint8)
                    reproject(mask.astype(np.uint8),aligned,src_transform=transform,src_crs=qa_crs,dst_transform=affine,dst_crs=crs,resampling=Resampling.nearest)
                    valid &= aligned.astype(bool)
            correction=float(row.get('calibration_revision',{}).get('additive_correction',0.))
            expected=np.where(valid,values+np.float32(correction),0).astype(np.float32)
            actual=cache[key];actual_mask=cache[key+'__valid']
            if actual.shape!=shape or not np.array_equal(actual_mask,valid):raise ValueError('Reconstructed QA mask differs: '+key)
            difference=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64))))
            if not np.array_equal(actual,expected):raise ValueError(f'Reconstructed reflectance differs: {key}, max_abs={difference}')
            bands.append(dict(band=key,valid_pixels=int(valid.sum()),max_absolute_difference=difference))
    for path,value in identities.items():
        stat=Path(path).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(value['size'],value['mtime_ns']):raise ValueError('Source changed while reading pixels')
    return dict(id=row['id'],role=row['role'],sensor=row['sensor'],provenance=provenance,status='EXACT_PARENT_REPRODUCTION',raw_QA_recomputed=raw_qa,original_parent_QA_independently_recomputed=raw_qa,bands=bands,source_files=identities)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('release','contracts','canonical','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,required=True);args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('Bounded read-only audit workers required')
    started=time.perf_counter();dataset=NativeOpticalDataset(args.release,args.contracts,role='TRAIN')
    with (args.release/'members.jsonl').open(encoding='utf-8-sig') as stream:rows=[json.loads(line) for line in stream if line.strip()]
    rows=[row for row in rows if row['role'] in ('TRAIN','VAL')]
    canonical_hash=sha256(args.canonical)
    with args.canonical.open(encoding='utf-8-sig') as stream:canonical={row['id']:row for row in map(json.loads,stream)}
    source_file_manifest=args.output.with_name(args.output.stem+'_source_files.json')
    report=dict(status='RUNNING',scope='all TRAIN/VAL members in existing expanded native release; not all-machine inventory',source_identities=dataset.identities,canonical_sha256=canonical_hash,members_total=len(rows),members_checked=0,failures=[],counts={},raw_QA_recomputed_members=0,parent_QA_only_members=0,source_file_manifest=str(source_file_manifest.resolve()),source_file_count=0,optimizer_updates=0,limitations=['Exact reproduction checks historical processing, not independent scientific calibration or V3 QA approval','Cached-parent SCL/QA_PIXEL/RADSAT masks are traced, not independently rederived from original QA imagery','SRF applicability, all-machine association, pair registration/change risk and V3 role/statistics freezing remain separate'])
    write(args.output,report);counts=Counter();source_files={}
    def one(row):
        try:return audit_member(row,canonical.get(row['id']),args.release)
        except (ValueError,OSError,KeyError,IndexError) as exc:return dict(id=row['id'],role=row['role'],status='FAILED',error=type(exc).__name__+': '+str(exc))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(one,rows):
            report['members_checked']+=1
            if result['status']=='FAILED':report['failures'].append(result)
            else:
                counts[f"{result['role']}/{result['sensor']}/{result['provenance']}"]+=1
                report['raw_QA_recomputed_members' if result['raw_QA_recomputed'] else 'parent_QA_only_members']+=1
                for path,value in result['source_files'].items():
                    if path in source_files and value!=source_files[path]:raise ValueError('Shared source identity changed during audit')
                    source_files[path]=value
            if report['members_checked']%100==0:
                report.update(counts=dict(counts),source_file_count=len(source_files),seconds=time.perf_counter()-started);write(args.output,report)
                print('LINEAGE',report['members_checked'],'/',len(rows),'failed',len(report['failures']),flush=True)
    if sha256(args.canonical)!=canonical_hash or sha256(args.release/'members.jsonl')!=dataset.identities['members']:raise ValueError('Input inventory changed during audit')
    write(source_file_manifest,source_files)
    report.update(status='LINEAGE_REPRODUCTION_PASS_NOT_V3_ADMISSION' if not report['failures'] else 'LINEAGE_REPRODUCTION_HAS_FAILURES',source_file_manifest_sha256=sha256(source_file_manifest),source_file_count=len(source_files),counts=dict(counts),seconds=time.perf_counter()-started)
    write(args.output,report)
    print(json.dumps({k:report[k] for k in ('status','members_checked','raw_QA_recomputed_members','parent_QA_only_members','seconds')},indent=2))
