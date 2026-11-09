"""Read-only EnMAP admission. Missing calibration/QA/lineage never becomes a training member."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from scipy.io import netcdf_file
from encoder_v3.data import sha256
from encoder_v3.latent_response.checkpoint import digest

ROOT=Path(__file__).resolve().parents[2]


def spectral_metadata(path):
    root=ET.parse(path).getroot(); records=[]
    for element in root.iter():
        fields={child.tag.split('}')[-1]:(child.text or '').strip() for child in element}
        if 'GainOfBand' in fields and 'wavelengthCenterOfBand' in fields:
            records.append(dict(channel_id='B'+str(int(element.attrib['number'])).zfill(3),
                ordinal=int(element.attrib['number'])-1,center_nm=float(fields['wavelengthCenterOfBand']),
                fwhm_nm=float(fields['FWHMOfBand']),gain=float(fields['GainOfBand']),offset=float(fields['OffsetOfBand'])))
    records.sort(key=lambda r:r['ordinal'])
    if not records or [r['ordinal'] for r in records]!=list(range(len(records))):
        raise ValueError('Unique contiguous product band identities required')
    if any(not all(np.isfinite(r[k]) for k in ('center_nm','fwhm_nm','gain','offset')) or r['gain']<=0 or r['fwhm_nm']<=0 for r in records):
        raise ValueError('Invalid spectral/calibration metadata')
    levels=[(e.text or '').strip() for e in root.iter() if e.tag.split('}')[-1]=='processingLevel']
    if levels!=['L2A']:
        raise ValueError('Explicit L2A metadata required')
    return records


def native_blockers(metadata_exists,qa_exists,role_frozen,train_statistics):
    problems=[]
    if not metadata_exists:problems.append('MISSING_PRODUCT_XML_CALIBRATION_AND_CHANNEL_IDENTITY')
    if not qa_exists:problems.append('MISSING_PIXEL_QA_CLOUD_SHADOW_CIRRUS_HAZE_SNOW_TESTFLAGS')
    if not role_frozen:problems.append('MISSING_FROZEN_TRAIN_VAL_LINEAGE_ASSIGNMENT')
    if not train_statistics:problems.append('MISSING_QA_FILTERED_TRAIN_ONLY_MEAN_STD')
    return problems


def overlap(a,b):return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])


def audit(root,output):
    output.mkdir(parents=True,exist_ok=True)
    release=ROOT/'observation_model/outputs/encoder_v2/data_expansion'
    existing=[json.loads(line) for line in (release/'members.jsonl').read_text().splitlines() if line.strip()]
    images=sorted(root.rglob('*SPECTRAL_IMAGE*.TIF'))
    result=dict(status='BLOCKED_NOT_ADMITTED',admitted_members=0,raw_images=[],derived_windows={},
                frozen_release_unchanged=True,real_optimizer_steps=0,
                model_srf_requirement='LR does not ingest measured SRF or Gaussian substitutes; measured SRF absence alone is not an admission blocker',
                source_release_sha256=sha256(release/'members.jsonl'))
    for image in images:
        prefix=image.name.split('-SPECTRAL_IMAGE')[0]
        metadata=list(root.rglob(prefix+'-METADATA.XML'))
        metadata=metadata[0] if metadata else None
        companions={kind:list(root.rglob(prefix+'*'+kind+'*.TIF')) for kind in ('PIXELMASK','QUALITY_CLOUD','QUALITY_CLOUDSHADOW','QUALITY_CIRRUS','QUALITY_HAZE','QUALITY_SNOW','QUALITY_TESTFLAGS')}
        row=dict(scene=prefix,path=str(image.resolve()),sha256=sha256(image),metadata_path=None if metadata is None else str(metadata.resolve()),
                 qa_files={k:[str(p.resolve()) for p in paths] for k,paths in companions.items()},role='UNASSIGNED',admitted=False)
        row['blockers']=native_blockers(metadata is not None,all(companions.values()),False,False)
        records=None
        if metadata:
            row['metadata_sha256']=sha256(metadata)
            try:
                records=spectral_metadata(metadata)
                row['catalog']=records
                row['radiometry']='reflectance = DN * GainOfBand + OffsetOfBand; nodata masked before transform'
            except Exception as exc:row['blockers'].append('INVALID_METADATA: '+str(exc))
        with rasterio.open(image) as ds:
            row.update(shape=[ds.count,ds.height,ds.width],crs=str(ds.crs),affine=list(ds.transform)[:6],nodata=ds.nodata,
                       embedded_scale=list(sorted(set(ds.scales))),embedded_units=sorted({str(u) for u in ds.units}))
            if ds.crs is None or not ds.crs.is_projected or abs(ds.transform.b)>0 or abs(ds.transform.d)>0 or ds.transform.a!=30 or ds.transform.e!=-30:
                row['blockers'].append('UNSUPPORTED_NATIVE_GRID')
            if records and len(records)!=ds.count:row['blockers'].append('BAND_COUNT_METADATA_MISMATCH')
            nodata_counts=np.zeros(ds.count,dtype=np.int64); finite_counts=np.zeros_like(nodata_counts)
            low=np.full(ds.count,np.inf);high=np.full(ds.count,-np.inf)
            for _,window in ds.block_windows(1):
                data=ds.read(window=window)
                valid=data!=ds.nodata if ds.nodata is not None else np.ones(data.shape,dtype=bool)
                nodata_counts+=(~valid).sum((1,2));finite_counts+=(valid&np.isfinite(data)).sum((1,2))
                low=np.minimum(low,np.where(valid,data,np.inf).min((1,2)))
                high=np.maximum(high,np.where(valid,data,-np.inf).max((1,2)))
            row['pixel_scan']=dict(all_bands_all_blocks=True,valid_not_nodata_per_band=finite_counts.tolist(),
                nodata_per_band=nodata_counts.tolist(),dn_min=[float(v) if np.isfinite(v) else None for v in low],dn_max=[float(v) if np.isfinite(v) else None for v in high],
                note='NoData and numeric checks only; these are not QA-filtered valid-pixel counts or TRAIN statistics')
            if records:
                row['physical_range_from_xml']=[[float(lo*r['gain']+r['offset']),float(hi*r['gain']+r['offset'])] if np.isfinite(lo) and np.isfinite(hi) else None for lo,hi,r in zip(low,high,records)]
            if np.any(finite_counts==0):
                row['blockers'].append('EMPTY_PRODUCT_BANDS_REQUIRE_EXPLICIT_BAD_BAND_POLICY')
                row['entirely_nodata_band_ordinals']=np.flatnonzero(finite_counts==0).tolist()
            bbox=transform_bounds(ds.crs,'EPSG:4326',*ds.bounds,densify_pts=21)
            hits=[r for r in existing if r.get('bbox') and overlap(bbox,r['bbox'])]
            row['bbox_wgs84']=list(bbox)
            row['existing_release_bbox_overlaps']=dict(Counter(r['role'] for r in hits))
            row['overlap_note']='Conservative bounding boxes; does not establish exact valid-support overlap or independent scene count'
        result['raw_images'].append(row)
    files=sorted((root/'paired_windows').glob('*.nc'))
    signatures=[];summary=Counter();sample=[]
    for path in files:
        stat=path.stat(); signatures.append([path.name,stat.st_size,stat.st_mtime_ns])
        try:
            with netcdf_file(str(path),'r',mmap=True) as nc:
                attrs=nc._attributes
                native=float(attrs.get('native_gsd_m',float('nan')))
                stored=float(attrs.get('resampled_to_m',float('nan')))
                if native==30 and stored==10:summary['declared_resampled_30m_to_10m']+=1
                if not all(k in attrs for k in ('crs','transform','datetime')):summary['missing_explicit_crs_transform_time']+=1
                if len(sample)<3:
                    sample.append(dict(file=path.name,shape=list(nc.variables['spectral'].shape),attrs={k:(v.decode(errors='replace') if isinstance(v,bytes) else v.item() if isinstance(v,np.generic) else str(v)) for k,v in attrs.items()}))
        except Exception as exc:summary['unreadable_headers']+=1
    result['derived_windows']=dict(files=len(files),header_checks=dict(summary),examples=sample,
        listing_signature=digest(signatures),admitted=False,
        blockers=['RESAMPLED_PRODUCTS_NOT_NATIVE_LR_INPUT','NO_TRACEABLE_PER_PIXEL_QA','INCOMPLETE_GEOMETRY_CALIBRATION_LINEAGE_CONTRACT'],
        payloads_rehashed=False)
    result['next_required']=['Download original quality-mask companions for each scene and matching XML for the 2026 scene',
        'Audit original QA semantics and bad-band handling; freeze scene/overlap groups and roles against existing TRAIN/VAL',
        'Compute EnMAP TRAIN-only mean/std from QA-filtered native observations; create a new independent release, registry/catalog and E1 identity',
        'Only then include admitted EnMAP observations in training recipe; never reinterpret 10m windows as native 30m measurements']
    (output/'admission.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT/'data/optical/enmap')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/encoder_v3/latent_response/plan_execution/enmap')
    args=parser.parse_args(); result=audit(args.root,args.output)
    print(result['status'],'native scenes',len(result['raw_images']),'derived',result['derived_windows']['files'])
