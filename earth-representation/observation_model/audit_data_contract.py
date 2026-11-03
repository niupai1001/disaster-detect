"""Freeze local inventories, deduplicate native observations and screen V2 input metadata.

No imagery download, model import, normalization fit or TEST pixel read.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.strtree import STRtree
import yaml

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
BASE=ROOT/'observation_model/outputs/encoder_v2/pre_G1_data_contract'
OUT=BASE/'v4'
S2=('B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12')
LS=('SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7')
S2_ALIAS=dict(zip(('blue','green','red','rededge1','rededge2','rededge3','nir','nir08','swir16','swir22'),S2))
LS_ALIAS=dict(zip(('coastal','blue','green','red','nir08','swir16','swir22'),LS))


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def key(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def save(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(path): return [json.loads(t) for t in Path(path).read_text(encoding='utf-8-sig').splitlines() if t.strip()]
def dump(name,rows): (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
def url(value):
    s=urlsplit(value);return urlunsplit((s.scheme.lower(),s.netloc.lower(),s.path,'',''))


def identity(product,date,sensor):
    instrument=next((v for v in ['S2A','S2B','S2C','LC08','LC09','LC8','LC9'] if product.startswith(v)),None)
    if not instrument:return None,None
    instrument={'LC8':'LC08','LC9':'LC09'}.get(instrument,instrument)
    tile=re.search(r'(?:^|_)T?(\d{1,2}[A-Z]{3})(?:_|$)',product) if sensor=='S2' else re.search(r'_L2\w{2}_(\d{6})_',product)
    day=re.search(r'(20\d{6})',product)
    day=day.group(1) if day else (date or '')[:10].replace('-','')
    tile_key=tile.group(1).zfill(5) if tile and sensor=='S2' else tile.group(1) if tile else None
    return instrument,(instrument+':'+day+':'+tile_key if day and tile_key else None)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():raise FileExistsError('Preserve frozen contract result; use a new version for a new inventory.')
    save('authorization.json',dict(stage='pre_G1_local_data_contract_and_SRF_acquisition',
        optimizer_updates_allowed=0,model_implementation_allowed=False,imagery_download_allowed=False,
        SRF_download_allowed=True,G1_started=False,automatic_continuation=False,
        test_pixel_reads_allowed=False,old_policy_modified=False,
        instruction='用户授权跨库存去重、核对进入契约并下载SRF；完成后由用户决定是否进入G1。'))
    # Reuse the first frozen snapshots, not a concurrently changing shared release.
    paths={name:BASE/(name+'.jsonl') for name in ['native_release','shared_release','legacy_locations','legacy_companions']}
    inventory={}
    for name,path in paths.items():
        data=path.read_bytes();(OUT/(name+'.jsonl')).write_bytes(data)
        inventory[name]=dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    save('inventory_snapshots.json',inventory)
    native=lines(OUT/'native_release.jsonl');shared=lines(OUT/'shared_release.jsonl')
    shared=[r for r in shared if r['sensor'] in ['s2','landsat']]
    print('Frozen native/shared rows',len(native),len(shared),flush=True)
    # Product metadata may exist in local STAC catalogue caches; no remote lookup.
    products={}
    catalog=WORK/'landslide-seg/outputs/use_data_integration/completion/catalogue'
    for p in sorted(catalog.glob('*.json')):
        try:
            obj=read(p)
            for item in obj.get('features',[]):products[item['id']]=item.get('properties',{})
        except (ValueError,OSError):continue
    href_paths=defaultdict(list);inherited_source_hashes={}
    for r in shared:
        for c in r.get('calibration',{}).values():
            src=c.get('source',{});p=src.get('path',c.get('source_uri'));href=src.get('asset_metadata',{}).get('href')
            if p and href:href_paths[url(href)].append(p)
            if p and c.get('sha256',src.get('sha256')):inherited_source_hashes[p]=c.get('sha256',src.get('sha256'))
    prior={v['path']:v for v in read(ROOT/'observation_model/outputs/encoder_v2/G0_audit/data_file_hashes.json')['files']}
    records=[]
    for r in native:
        sensor=r['sensor'];bands={b['name'].split(':')[-1]:dict(path=b['source'],scale=b['scale'],offset=b['offset'],expected_gsd=b['native_gsd_m'],expected_sha256=prior.get(b['source'],{}).get('sha256') or inherited_source_hashes.get(b['source'])) for b in r['bands']}
        records.append(dict(inventory='native_release',source_id=r['observation_id'],sensor=sensor,
            product=r['product_id'],date=r['date'],declared_role=r['split'].upper(),group=r['group'],
            bbox=r['geographic_bbox'],bands=bands,quality_paths=[str(Path(next(iter(bands.values()))['path']).parent/q) for q in (['SCL.tif'] if sensor=='S2' else ['qa_pixel.tif','qa_radsat.tif','qa_aerosol.tif'])],
            window=None,metadata_origin='accepted_native_release',properties={}))
    for r in shared:
        sensor='S2' if r['sensor']=='s2' else 'L8';expected=S2 if sensor=='S2' else LS
        alias=S2_ALIAS if sensor=='S2' else LS_ALIAS
        product=r.get('source_item',r.get('item_id',''));props=r.get('properties') or products.get(product,{})
        bands={};cal=r.get('calibration',{});physical=r.get('physical_transforms',[])
        for i,original in enumerate(r['band_ids']):
            b=alias.get(original,original)
            if b not in expected:continue
            source=cal.get(original,cal.get(b,{}));item=source.get('source',{});p=source.get('source_uri',item.get('path'))
            scale=source.get('scale');offset=source.get('offset')
            if not p and r.get('source_folder'):
                p=str(Path(r['source_folder'])/(b+'_AOI.tif'))
                side=Path(p).with_suffix('.json')
                if side.exists():
                    metadata=read(side);scale=metadata.get('scale');offset=metadata.get('offset');product=product or metadata.get('item','')
            if not p and isinstance(physical,list) and i<len(physical):
                item=physical[i];href=item.get('href');available=href_paths.get(url(href),[]) if href else []
                p=available[0] if available else None;scale=item.get('scale');offset=item.get('offset')
            if p:bands[b]=dict(path=p,scale=scale,offset=offset,expected_gsd=10 if sensor=='S2' and b in ['B02','B03','B04','B08'] else 20 if sensor=='S2' else 30,
                expected_sha256=source.get('sha256',item.get('sha256',prior.get(p,{}).get('sha256'))))
        if not product and bands:
            candidate=Path(next(iter(bands.values()))['path']).parent.name
            if candidate.startswith(('S2A','S2B','S2C','LC08','LC09')):product=candidate
        if bands:
            product_file=Path(next(iter(bands.values()))['path']).parent/'product.json'
            if product_file.exists():props=read(product_file).get('properties',props)
        qs=[q['path'] for q in r.get('quality_sources',{}).values() if q.get('path')]
        if bands and not qs:
            parent=Path(next(iter(bands.values()))['path']).parent
            qs=[str(parent/q) for q in (['SCL_AOI.tif'] if r.get('source_folder') else ['SCL.tif'] if sensor=='S2' else ['qa_pixel.tif','qa_radsat.tif','qa_aerosol.tif'])]
        if sensor=='L8' and bands:
            aerosol=str(Path(next(iter(bands.values()))['path']).parent/'qa_aerosol.tif')
            if aerosol not in qs:qs.append(aerosol)
        records.append(dict(inventory='shared_release',source_id=r['id'],sensor=sensor,product=product,
            date=r.get('date',r.get('datetime')),declared_role={'PRETRAIN_VAL':'VAL'}.get(r['role'],r['role']),
            group=r.get('group',r.get('site_id')),bbox=r['bbox'],bands=bands,quality_paths=qs,
            window=None if r.get('origin')=='four_view_pilot' else dict(crs=r['crs'],transform=r['transform'],shape=r.get('shape',r.get('cache_audit',{}).get('shape',[0,128,128]))[-2:]),
            metadata_origin=r.get('origin','legacy'),properties=props,cache_uri=r.get('cache_uri'),units=r.get('units')))
    # Exact native source+support aliases collapse even across renamed product metadata.
    unique=[];aliases=[];seen={}
    for r in records:
        instrument,acq=identity(r['product'],r['date'],r['sensor']);r.update(instrument=instrument,acquisition_key=acq)
        if instrument=='LC09':r['sensor']='L9'
        source=[(b,v.get('expected_sha256') or str(Path(v['path'])).casefold(),v['scale'],v['offset']) for b,v in sorted(r['bands'].items())]
        fingerprint=key(dict(sensor=r['sensor'],sources=source,window=r['window'])) if len(source)==len(S2 if r['sensor']=='S2' else LS) else key(dict(sensor=r['sensor'],product=r['product'],date=r['date'],bbox=[round(x,8) for x in r['bbox']],unknown_provenance=r['metadata_origin']))
        rid='obs_'+fingerprint[:24]
        if fingerprint in seen:
            u=unique[seen[fingerprint]];u['aliases'].append(dict(inventory=r['inventory'],source_id=r['source_id'],role=r['declared_role']))
            u['declared_roles'].append(r['declared_role']);aliases.append(dict(inventory=r['inventory'],source_id=r['source_id'],canonical_id=u['id']))
        else:
            r.update(id=rid,declared_roles=[r['declared_role']],aliases=[dict(inventory=r['inventory'],source_id=r['source_id'],role=r['declared_role'])]);seen[fingerprint]=len(unique);unique.append(r)
    print('Unique observations',len(unique),'duplicate aliases',len(aliases),flush=True)
    parent=list(range(len(unique)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def merge(a,b):
        a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
    seen_acq={};seen_group={}
    for i,r in enumerate(unique):
        if r['acquisition_key']:
            if r['acquisition_key'] in seen_acq:merge(i,seen_acq[r['acquisition_key']])
            else:seen_acq[r['acquisition_key']]=i
        if r['group']:
            g=(r['inventory'],r['group'])
            if g in seen_group:merge(i,seen_group[g])
            else:seen_group[g]=i
    geoms=[box(*r['bbox']) for r in unique];tree=STRtree(geoms)
    regions=yaml.safe_load((ROOT/'configs/regions.yaml').read_text())['regions']
    protected=[box(*regions[n]['bbox']) for n in ['hiroshima','thrissur','italy','newzealand']]
    for i,g in enumerate(geoms):
        for j in tree.query(g,predicate='intersects'):
            j=int(j)
            if j>i and g.intersection(geoms[j]).area>1e-12:merge(i,j)
        if any(g.intersection(p).area>1e-12 for p in protected):unique[i]['declared_roles'].append('GUARD_PROTECTED_GEOGRAPHY')
    component=defaultdict(set)
    for i,r in enumerate(unique):component[find(i)].update(r['declared_roles'])
    role_priority=lambda s: 'TEST_ISOLATION' if 'TEST' in s else 'GUARD' if any(x.startswith('GUARD') for x in s) else 'VAL' if 'VAL' in s else 'TRAIN'
    for i,r in enumerate(unique):
        r['isolation_group']='group_'+str(find(i));r['effective_partition']=role_priority(component[find(i)])
        r['component_roles']=sorted(component[find(i)]);r['role_conflict']=len(component[find(i)])>1
    print('Isolation partitions',dict(Counter(r['effective_partition'] for r in unique)),flush=True)
    headers={}
    def header(p):
        if p in headers:return headers[p]
        path=Path(p)
        if not path.is_file():headers[p]=dict(exists=False);return headers[p]
        try:
            with rasterio.open(path) as f:
                t=f.transform
                h=dict(exists=True,crs=str(f.crs),transform=list(t)[:6],shape=[f.height,f.width],dtype=f.dtypes[0],
                    nodata=f.nodata,scale=f.scales[0],offset=f.offsets[0],bounds=list(f.bounds),
                    native_x_m=math.hypot(t.a,t.d),native_y_m=math.hypot(t.b,t.e),projected=bool(f.crs and f.crs.is_projected),
                    bbox_wgs84=list(transform_bounds(f.crs,'EPSG:4326',*f.bounds)))
            headers[p]=h
        except Exception as e:headers[p]=dict(exists=True,error=str(e))
        return headers[p]
    for i,r in enumerate(unique):
        issues=[];expected=S2 if r['sensor']=='S2' else LS
        if r['effective_partition']=='TEST_ISOLATION':r.update(data_status='TEST_METADATA_ONLY',issues=['TEST_pixels_and_headers_not_opened']);continue
        if not r['instrument'] or not r['acquisition_key']:issues.append('original_instrument_or_acquisition_unresolved')
        if r.get('units','surface_reflectance')!='surface_reflectance':issues.append('not_L2_surface_reflectance')
        if r['instrument'] in ['LC09']:issues.append('L9_interface_only_not_current_primary_domain')
        if len(r['bands'])!=len(expected):issues.append('original_native_band_paths_unresolved')
        descriptors=[]
        for b,v in r['bands'].items():
            h=header(v['path'])
            if not h.get('exists'):issues.append('missing_local_native_band:'+b);continue
            if 'error' in h:issues.append('unreadable_native_band:'+b);continue
            if not h['projected'] or max(abs(h[k]-v['expected_gsd']) for k in ['native_x_m','native_y_m'])>1e-5:issues.append('native_grid_mismatch:'+b)
            if v['scale'] is None or v['offset'] is None:issues.append('explicit_scale_offset_missing:'+b)
            if r['window']:
                w=r['window'];a,_,c,_,e,f=w['transform'];height,width=w['shape'];bounds=[c,f+e*height,c+a*width,f]
                needed=transform_bounds(w['crs'],h['crs'],*bounds)
                tolerance=max(h['native_x_m'],h['native_y_m'])
                if needed[0]<h['bounds'][0]-tolerance or needed[1]<h['bounds'][1]-tolerance or needed[2]>h['bounds'][2]+tolerance or needed[3]>h['bounds'][3]+tolerance:issues.append('source_window_not_covered:'+b)
            descriptors.append(dict(band_key=b,source=v['path'],native_grid=h,calibration=dict(scale=v['scale'],offset=v['offset'],unit='surface_reflectance'),
                hash_evidence=prior.get(v['path'],{}).get('sha256') or v.get('expected_sha256'),
                hash_evidence_scope='G0_actual_file_hash' if v['path'] in prior else 'inherited_acquisition_manifest_hash_not_rehashed',
                spectral_response_ref='downloaded_resources_pending_G1_instrument_mapping'))
        for p in r['quality_paths']:
            h=header(p)
            if not h.get('exists'):issues.append('missing_local_QA:'+Path(p).name)
            elif 'error' in h:issues.append('unreadable_QA:'+Path(p).name)
        if not r['quality_paths']:issues.append('QA_paths_unresolved')
        r['native_band_contract']=descriptors;r['issues']=sorted(set(issues))
        r['data_status']='NATIVE_METADATA_READY_PENDING_G1' if not issues else 'LOCAL_CONTRACT_GAPS'
        if i%1000==0:print('Header contract checked',i,flush=True)
    # Legacy files without primary native provenance stay in an explicit pending index.
    legacy=[];known={r['source']:r for r in lines(OUT/'legacy_locations.jsonl')}
    guards=('hiroshima','thrissur','italy','newzealand')
    for p in sorted((WORK/'landslide-seg/data/sen12').glob('*.nc')):
        r=known.get(str(p),{});legacy.append(dict(source=str(p),sample_id=p.stem,bytes=p.stat().st_size,
            role_constraint=r.get('role','protected' if any(p.stem.startswith(g+'_') for g in guards) else 'unresolved'),
            known_dates=r.get('dates'),status='DERIVED_CACHE_NATIVE_PROVENANCE_PENDING',
            reason='Original instrument/product and per-band native support not established; not discarded as bad imagery.'))
    dump('canonical_observations.jsonl',unique);dump('duplicate_aliases.jsonl',aliases);dump('legacy_pending.jsonl',legacy)
    # Derived copies are indexed as artifacts/lineage, not counted as fresh native observations.
    companions={(r['sample_id'],r['sensor']):r for r in lines(OUT/'legacy_companions.jsonl')}
    artifacts=[];lineages=defaultdict(list)
    for p in sorted((ROOT/'data').rglob('*.nc')):
        relative=p.relative_to(ROOT/'data');folder=relative.parts[0];sensor='landsat' if 'landsat' in str(relative).lower() else 'hls' if 'hls' in str(relative).lower() else 'unknown'
        if sensor=='unknown' and 'companions' in folder:
            sensor='landsat' if p.parent.name.lower() in ['l8','landsat'] else 'hls' if p.parent.name.lower()=='hls' else 'unknown'
        c=companions.get((p.stem,sensor),{});lineage=key([sensor,c.get('source_product'),p.stem])
        record=dict(path=str(p),container='derived_nc',sensor=sensor,sample_id=p.stem,processing_directory=str(relative.parent),bytes=p.stat().st_size,
                    source_product=c.get('source_product'),role_constraint=c.get('role','unresolved'),lineage_key=lineage,
                    status='DERIVED_DIAGNOSTIC_ONLY_PENDING_SOURCE_AUDIT',
                    boundary='Same lineage is not proof of equal pixels; old/fixed/selected-time versions are not extra native observations.')
        artifacts.append(record);lineages[lineage].append(str(p))
    dump('derived_artifacts.jsonl',artifacts)
    dump('derived_lineage_groups.jsonl',[dict(lineage_key=k,files=v,versions=len(v)) for k,v in lineages.items()])
    dump('native_metadata_candidates.jsonl',[r for r in unique if r['data_status']=='NATIVE_METADATA_READY_PENDING_G1'])
    dump('train_native_metadata_candidates.jsonl',[r for r in unique if r['data_status']=='NATIVE_METADATA_READY_PENDING_G1' and r['effective_partition']=='TRAIN'])
    save('native_headers.json',headers)
    summary=dict(stage='pre_G1',optimizer_updates=0,model_forward_backward=0,imagery_downloads=0,G1_started=False,
        raw_inventory_rows=len(records),canonical_observations=len(unique),duplicate_aliases=len(aliases),
        declared_input_counts=dict(Counter(r['inventory']+'/'+r['declared_role']+'/'+r['sensor'] for r in records)),
        statuses=dict(Counter(r['data_status'] for r in unique)),partitions=dict(Counter(r['effective_partition'] for r in unique)),
        status_by_partition=dict(Counter(r['effective_partition']+'/'+r['data_status'] for r in unique)),
        candidate_by_sensor=dict(Counter(r['effective_partition']+'/'+r['sensor'] for r in unique if r['data_status']=='NATIVE_METADATA_READY_PENDING_G1')),
        reasons=dict(Counter(issue for r in unique for issue in r['issues'])),legacy_pending_files=len(legacy),
        native_header_files=len(headers),test_pixel_or_header_reads=0,
        derived_artifact_files=len(artifacts),derived_lineage_groups=len(lineages),derived_multi_version_groups=sum(len(v)>1 for v in lineages.values()),
        ready_semantics='Metadata/native-source candidates only; not physically accepted or trainable until G1 SRF applicability/radiometry/QA tests and explicit later authorization.',
        frozen_old_roles_changed=False,normalization_statistics_fitted=False,
        isolation='Connected components join acquisition, inherited groups and positive-area geographic overlap across dates/sensors. TEST > GUARD > VAL > TRAIN; source roles remain recorded; no held-out-to-train promotion.')
    assert len(records)==len(unique)+len(aliases)
    assert all(not (r['effective_partition']=='TRAIN' and any(x!='TRAIN' for x in r['component_roles'])) for r in unique)
    assert identity('S2A_MSIL2A_20180216T093051_R136_T33PTP_20201013T215030',None,'S2')[1]==identity('S2A_33PTP_20180216_0_L2A',None,'S2')[1]
    assert identity('S2B_2LLK_20241030_0_L2A',None,'S2')==identity('S2B_02LLK_20241030_0_L2A',None,'S2')
    save('summary.json',summary)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='hashes.json'}
    save('hashes.json',files)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
