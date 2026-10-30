"""Materialize local sources without changing frozen data or starting model training."""
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from landslide_seg.data.use_admission import sha256
from landslide_seg.models.spectral_contract import S2_WAVELENGTH_UM
from prepare_use_external import exclusion
from download_use_four_view_pilot import save

OUT=ROOT/'outputs/use_data_integration'
PILOT=ROOT/'data/use_multisensor/four_view_pilot_v1'
S2_BANDS=('B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12')
S2_WAVES=dict(S2_WAVELENGTH_UM,B01=.443,B09=.945)
LS_BANDS=('coastal','blue','green','red','nir08','swir16','swir22')
LS_WAVES=(.443,.482,.561,.655,.865,1.609,2.201)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def append(path, value):
    with Path(path).open('a',encoding='utf8') as handle:
        handle.write(json.dumps(value,ensure_ascii=False)+'\n')


def identity(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def publish(rows, name='local_release'):
    directory=OUT/name
    directory.mkdir(parents=True,exist_ok=True)
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate observation identity')
    unique=[];content={};aliases={}
    for source in rows:
        row=dict(source)
        key=(row['sensor'],row.get('sha256'),row['crs'],tuple(row['transform']),row.get('date'))
        if key in content and row.get('sha256'):
            retained=content[key]
            if retained['role']!=row['role']:
                raise ValueError('Identical observation crosses data roles')
            aliases[row['id']]=retained['id']
            retained.setdefault('source_alias_ids',[]).append(row['id'])
            continue
        content[key]=row;unique.append(row)
    rows=unique
    for row in rows:
        anchor=row.get('request',{}).get('anchor_id')
        if anchor in aliases:
            row['request']=dict(row['request'],anchor_id=aliases[anchor],original_anchor_id=anchor)
    path=directory/'manifest.jsonl'
    tmp=path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    tmp.replace(path)
    counts=dict(Counter(r['role']+'/'+r['sensor'] for r in rows))
    by_id={r['id']:r for r in rows}
    bundles=defaultdict(dict)
    for row in rows:
        anchor=by_id.get(row.get('request',{}).get('anchor_id'),row)
        group=anchor['site_id'] if anchor.get('origin')=='four_view_pilot' else anchor['id']
        bundles[(anchor['role'],group)][row['sensor']]=row
    pairs=[]
    for (role,group),views in bundles.items():
        if role=='GUARD' or not {'s2','landsat'}<=set(views):continue
        left,right=views['s2'],views['landsat']
        common=None
        if left['crs']==right['crs'] and np.allclose(left['transform'],right['transform'],atol=1e-6,rtol=0):
            with np.load(left['cache_uri']) as a,np.load(right['cache_uri']) as b:
                if a['pixel_valid'].shape[1:]==b['pixel_valid'].shape[1:]:
                    common=float((a['pixel_valid'].all(0)&b['pixel_valid'].all(0)).mean())
        pairs.append(dict(id=group,role=role,s2_id=left['id'],landsat_id=right['id'],
            joint_valid_ratio=common,qualified=common is not None and common>=.8,
            semantics='Same physical region; dates may differ; not pixelwise reflectance truth',
            dates=[left.get('date'),right.get('date')],views={k:v['id'] for k,v in views.items()}))
    pair_path=directory/'pairs.jsonl'
    pair_temp=pair_path.with_suffix('.tmp')
    pair_temp.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pairs),encoding='utf8')
    pair_temp.replace(pair_path)
    save(directory/'release.json',dict(ok=True,manifest_sha256=sha256(path),counts=counts,
        pairs_sha256=sha256(pair_path),qualified_pairs=dict(Counter(p['role'] for p in pairs if p['qualified'])),
        duplicate_aliases=aliases,
        code_sha256=sha256(Path(__file__)), units='Physical domains explicit per observation; no new fitted statistics',
        protocol_sha256=sha256(ROOT/'manifests/v4r1/protocol.json')))
    return counts


def materialize(identifier, x, valid, metadata, known, row):
    x=np.asarray(x,dtype='float32');valid=np.asarray(valid,dtype=bool)
    if x.shape!=valid.shape or not np.isfinite(x[valid]).all() or valid.all(0).mean()<.8:
        raise ValueError('Less than 80% finite common clear coverage')
    path=OUT/'cache'/(identifier+'.npz')
    path.parent.mkdir(parents=True,exist_ok=True)
    values=dict(x=np.where(valid,x,0).astype('float32'),pixel_valid=valid,
                metadata=np.asarray(metadata,dtype='float32'),known=np.asarray(known,dtype=bool))
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(values) or any(not np.array_equal(old[k],v) for k,v in values.items()):
                raise ValueError('Existing derived cache differs')
    else:
        temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**values);temporary.replace(path)
    return dict(row,id=identifier,cache_uri=str(path),sha256=sha256(path),
                valid_ratio=float(valid.all(0).mean()),shape=list(x.shape))


def read_grid(path, crs, affine, width, height, nearest=False):
    with rasterio.open(path) as source:
        with WarpedVRT(source,crs=crs,transform=affine,width=width,height=height,
                       resampling=Resampling.nearest if nearest else Resampling.bilinear) as vrt:
            return vrt.read(1,masked=True)


def local_pilot(site, role):
    bounds=site['bbox_projected'];crs=site['crs']
    # Native sources remain untouched; this explicit derived 128-grid covers the same 1.44km ROI.
    affine=from_bounds(*bounds,128,128)
    common=dict(role=role,origin='four_view_pilot',site_id=site['site_id'],group=site['anchor_root'],
        crs=crs,transform=list(affine)[:6],bbox=list(shape(site['footprint_wgs84']).bounds),
        grid_gsd_m=(bounds[2]-bounds[0])/128,registration='source_georeferencing_only',
        source_site_sha256=sha256(PILOT/'sites'/site['site_id']/'site.json'))
    rows=[];errors=[]
    for sensor,kind in (('s2','s2_l2a'),('landsat','landsat_l2'),('s1','s1'),('hr','hr')):
        try:
            assets=site['files'][kind]
            bands=S2_BANDS if sensor=='s2' else LS_BANDS if sensor=='landsat' else ('vv','vh') if sensor=='s1' else ('ms',)
            x=[];valid=[];metadata=[];known=[];provenance={}
            if sensor=='s2':
                scl=read_grid(PILOT/assets['SCL']['file'],crs,affine,128,128,True)
                quality=np.isin(scl.data,(4,5,6,7))&~np.ma.getmaskarray(scl)
            elif sensor=='landsat':
                qa=read_grid(PILOT/assets['qa_pixel']['file'],crs,affine,128,128,True)
                sat=read_grid(PILOT/assets['qa_radsat']['file'],crs,affine,128,128,True)
                quality=(qa.data.astype('uint16')&63)==0
                quality &= (sat.data==0)&~np.ma.getmaskarray(qa)&~np.ma.getmaskarray(sat)
            else:
                quality=np.ones((128,128),dtype=bool)
            for i,band in enumerate(bands):
                record=assets[band];path=PILOT/record['file']
                if sha256(path)!=record['sha256']:
                    raise ValueError('Source hash mismatch: '+str(path))
                props=record.get('asset_metadata',{})
                if sensor=='hr':
                    with rasterio.open(path) as source:
                        for channel in range(1,source.count+1):
                            with WarpedVRT(source,crs=crs,transform=affine,width=128,height=128,resampling=Resampling.bilinear) as vrt:
                                data=vrt.read(channel,masked=True)
                            x.append(data.data);valid.append(~np.ma.getmaskarray(data)&np.isfinite(data.data))
                    metadata=np.zeros((len(x),3));known=np.zeros((len(x),3),bool)
                    break
                data=read_grid(path,crs,affine,128,128)
                info=props.get('raster:bands',[{}])[0]
                if sensor=='s2':
                    baseline=re.search(r'_N(\d{4})_',props.get('href',''))
                    if 'scale' in info and 'offset' in info:
                        scale,offset=info['scale'],info['offset'];evidence='asset raster:bands'
                    elif baseline:
                        scale,offset=.0001,(-.1 if int(baseline.group(1))>=400 else 0.)
                        evidence='ESA L2A processing baseline in original product URI'
                    else:
                        raise ValueError('Missing verifiable S2 processing baseline/offset')
                    wave=S2_WAVES[band];gsd=10 if band in ('B02','B03','B04','B08') else 60 if band in ('B01','B09') else 20
                elif sensor=='landsat':
                    if 'scale' not in info or 'offset' not in info:
                        raise ValueError('Missing Landsat calibration')
                    scale,offset=info['scale'],info['offset'];wave=LS_WAVES[i];gsd=30;evidence='asset raster:bands'
                else:
                    scale,offset=1.,0.;wave=0.;gsd=10;evidence='source RTC gamma-naught description'
                rho=data.data.astype('float32')*scale+offset
                ok=~np.ma.getmaskarray(data)&np.isfinite(rho)&quality
                x.append(rho);valid.append(ok);metadata.append([wave,0,gsd]);known.append([sensor!='s1',False,True])
                provenance[band]=dict(source_uri=str(path),sha256=record['sha256'],scale=scale,offset=offset,evidence=evidence)
            units='surface_reflectance' if sensor in ('s2','landsat') else 'gamma0_linear' if sensor=='s1' else 'photographic_DN_not_reflectance'
            band_ids=list(bands) if sensor!='hr' else ['SPOT_MS_'+str(i+1) for i in range(len(x))]
            row=dict(common,sensor=sensor,date=site['datetime'][sensor],band_ids=band_ids,units=units,calibration=provenance)
            rows.append(materialize('pilot_'+site['site_id']+'_'+sensor,x,valid,metadata,known,row))
        except (KeyError,ValueError,OSError) as exc:
            errors.append(dict(site_id=site['site_id'],sensor=sensor,error=str(exc)))
    return rows,errors


def additional_f():
    audit=read(ROOT/'outputs/use/fg_audit/summary.json')
    acceptance=read(ROOT/'outputs/use/external/f_local_acceptance.json')
    old=read(ROOT/'outputs/use/external/f_local_manifest.json')
    roles={name:role for role,key in [('TRAIN','train_ids'),('PRETRAIN_VAL','validation_ids'),('GUARD','guard_ids')] for name in acceptance[key]}
    previous={tuple(r['source_window'][:2]):r for r in old}
    used={(r['source_item'],tuple(r['source_window'])) for r in old}
    observations=lines(ROOT/'outputs/use/fg_audit/f_qualified_observations.jsonl')
    by_folder=defaultdict(list)
    for row in observations:
        if row['scene_cloud_percent']<20 and (row['item'],tuple(row['window'])) not in used and tuple(row['window'][:2]) in previous:
            by_folder[row['folder']].append(row)
    rows=[]
    for folder,items in by_folder.items():
        base=Path(folder);bands=tuple(S2_WAVELENGTH_UM)
        for band in (*bands,'SCL'):
            path=base/(band+'_AOI.tif')
            if sha256(path)!=audit['F']['source_hashes'][next(i for i,h in enumerate(audit['F']['source_hashes']) if h['path']==str(path))]['sha256']:
                raise ValueError('Changed F source '+str(path))
        with ExitStack() as stack:
            blue=stack.enter_context(rasterio.open(base/'B02_AOI.tif'))
            readers=[];physical=[]
            for band in (*bands,'SCL'):
                src=stack.enter_context(rasterio.open(base/(band+'_AOI.tif')))
                readers.append(stack.enter_context(WarpedVRT(src,crs=blue.crs,transform=blue.transform,width=blue.width,height=blue.height,
                    resampling=Resampling.nearest if band=='SCL' else Resampling.bilinear)))
                physical.append((src.scales[0],src.offsets[0]))
            for item in items:
                window=Window(*item['window']);scl=readers[-1].read(1,window=window,masked=True)
                quality=np.isin(scl.data,(4,5,6,7))&~np.ma.getmaskarray(scl)
                x=[];valid=[]
                for reader,(scale,offset) in zip(readers[:-1],physical[:-1]):
                    data=reader.read(1,window=window,masked=True);rho=data.data.astype('float32')*scale+offset
                    x.append(rho);valid.append(quality&~np.ma.getmaskarray(data)&np.isfinite(rho))
                inherited=previous[tuple(item['window'][:2])]
                meta=[[S2_WAVELENGTH_UM[b],0,10 if b in ('B02','B03','B04','B08') else 20] for b in bands]
                known=np.ones((10,3),bool);known[:,1]=False
                row=dict(role=roles[inherited['file']],sensor='s2',origin='f_additional_date',
                    site_id='F_'+item['key'],group=inherited['group'],band_ids=list(bands),date=item['date'],
                    crs=str(blue.crs),transform=list(blue.window_transform(window))[:6],bbox=item['bbox_wgs84'],
                    units='surface_reflectance',source_item=item['item'],source_folder=folder,source_window=item['window'])
                rows.append(materialize('F_'+identity(item['item']+item['key']),x,valid,meta,known,row))
        print('F additional',len(rows),flush=True)
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/'outputs/use_data_admission'
    release=read(baseline/'release.json')
    for path,expected in release['source_sha256'].items():
        if sha256(ROOT/path)!=expected:raise ValueError('Frozen source changed '+path)
    rows=lines(baseline/'admitted.jsonl')
    for row in rows:
        row.update(id='old_'+identity(row['file']),cache_uri=str(ROOT/'outputs/use/external'/row['file']),
                   units='surface_reflectance',date=row.get('date',row.get('datetime')))
    extra=additional_f();rows.extend(extra)
    publish(rows)
    save(OUT/'local_status.json',dict(phase='pilot',f_added=dict(Counter(r['role'] for r in extra))))
    sites=read(PILOT/'manifest.json')
    protected=STRtree(exclusion(events={'hiroshima','thrissur','italy','newzealand'}))
    old_geos=[box(*r['bbox']) for r in rows if r['role']!='TRAIN']
    old_tree=STRtree(old_geos)
    groups=defaultdict(list)
    for site in sites:groups[site['anchor_root']].append(shape(site['footprint_wgs84']))
    # Root identity and actual intersecting footprints form indivisible connected groups.
    keys=sorted(groups);geos=[unary_union(groups[k]) for k in keys];tree=STRtree(geos);parent=list(range(len(keys)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,g in enumerate(geos):
        for j in tree.query(g,predicate='intersects'):parent[find(int(j))]=find(i)
    group_roles={k:('PRETRAIN_VAL' if int(identity(keys[find(i)]),16)%20==0 else 'TRAIN') for i,k in enumerate(keys)}
    new_val=unary_union([g for k,g in zip(keys,geos) if group_roles[k]=='PRETRAIN_VAL'])
    errors=[];added=[]
    for i,site in enumerate(sites):
        geo=shape(site['footprint_wgs84']);role=group_roles[site['anchor_root']]
        if len(protected.query(geo,predicate='intersects')):role='GUARD'
        # Geographic buffers are metric in the site's local CRS; guard on any old held-out contact.
        project=Transformer.from_crs(4326,site['crs'],always_xy=True).transform
        reverse=Transformer.from_crs(site['crs'],4326,always_xy=True).transform
        buffered=transform(reverse,transform(project,geo).buffer(1280))
        if role=='TRAIN' and (len(old_tree.query(buffered,predicate='intersects')) or buffered.intersects(new_val)):
            role='GUARD'
        result,failures=local_pilot(site,role);added.extend(result);errors.extend(failures)
        if (i+1)%20==0:print('pilot',i+1,'observations',len(added),'failures',len(errors),flush=True)
    rows.extend(added)
    counts=publish(rows)
    save(OUT/'local_status.json',dict(phase='local_complete',counts=counts,f_added=dict(Counter(r['role'] for r in extra)),
        pilot_added=dict(Counter(r['role']+'/'+r['sensor'] for r in added)),pilot_failures=errors))
    print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
