"""Materialize the audited F-drive windows for USE; no labels, network or training."""
import hashlib
import json
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import box
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from landslide_seg.models.spectral_contract import S2_WAVELENGTH_UM

OUT = ROOT/'outputs/use/external'
AUDIT = ROOT/'outputs/use/fg_audit'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def geographic_split(rows):
    groups = sorted({r['group'] for r in rows},key=lambda g:hashlib.sha256(('f_split1234:'+g).encode()).hexdigest())
    held = set(groups[:max(1,round(len(groups)*.05))])
    validation = [r for r in rows if r['group'] in held]
    # Dense tiles share boundaries: keep one full tile of separation around validation.
    buffered = STRtree([box(*r['bbox_projected']).buffer(1280) for r in validation])
    train, guard = [], []
    for row in rows:
        if row['group'] in held:
            continue
        (guard if len(buffered.query(box(*row['bbox_projected']),predicate='intersects')) else train).append(row)
    if not train or not validation:
        raise ValueError('Empty geographic split')
    return train, validation, guard


def load_f_members(root=OUT):
    manifest = root/'f_local_manifest.json'
    acceptance = read(root/'f_local_acceptance.json')
    if not acceptance.get('ok') or sha(manifest)!=acceptance['manifest_sha256']:
        raise ValueError('F manifest/acceptance changed')
    if sha(ROOT/'manifests/v4r1/protocol.json')!=acceptance['protocol_sha256']:
        raise ValueError('F source protocol changed')
    rows = read(manifest)
    train, validation = [], []
    for key, destination in [('train_ids',train),('validation_ids',validation)]:
        ids = set(acceptance[key])
        destination.extend(r for r in rows if r['file'] in ids)
        if len(destination)!=len(ids) or not ids:
            raise ValueError('Invalid F members')
    expected = geographic_split(rows)
    for members, key in zip(expected,('train_ids','validation_ids','guard_ids')):
        if {r['file'] for r in members}!=set(acceptance[key]):
            raise ValueError('F geographic split differs from its registered seed/group/buffer rule')
    return train, validation, acceptance


def main():
    source = AUDIT/'f_strict_qualified_locations.jsonl'
    audit = read(AUDIT/'summary.json')
    if sha(source)!=audit['artifact_sha256'][source.name]:
        raise ValueError('Strict audited source selection changed')
    selected = [json.loads(l) for l in source.read_text().splitlines()]
    assert len(selected)==2208 and len({r['key'] for r in selected})==2208
    by_folder = defaultdict(list)
    for row in selected:
        assert row['scene_cloud_percent']<20 and row['valid_ratio']>=.8
        by_folder[row['folder']].append(row)
    folder = OUT/'f_local_v1'
    folder.mkdir(parents=True,exist_ok=True)
    source_hashes = {r['path']:r['sha256'] for r in audit['F']['source_hashes']}
    rows = []
    for name, members in sorted(by_folder.items()):
        base = Path(name)
        bands = tuple(members[0]['bands'])
        assert bands==tuple(S2_WAVELENGTH_UM)
        metadata = np.asarray([[S2_WAVELENGTH_UM[b],0,10 if b in ('B02','B03','B04','B08') else 20] for b in bands],dtype='float32')
        known = np.ones((10,3),dtype=bool);known[:,1]=False
        with ExitStack() as stack:
            blue = stack.enter_context(rasterio.open(base/'B02_AOI.tif'))
            vrts = []
            transforms = []
            for b in (*bands,'SCL'):
                path = base/(b+'_AOI.tif')
                if sha(path)!=source_hashes[str(path)]:
                    raise ValueError('Audited source changed: '+str(path))
                src = stack.enter_context(rasterio.open(path))
                vrts.append(stack.enter_context(WarpedVRT(src,crs=blue.crs,transform=blue.transform,
                    width=blue.width,height=blue.height,resampling=Resampling.nearest if b=='SCL' else Resampling.bilinear)))
                transforms.append((src.scales[0],src.offsets[0]))
            stripes = defaultdict(list)
            for row in members:
                stripes[row['window'][1]].append(row)
            for y, chunk in sorted(stripes.items()):
                win = Window(0,y,blue.width,128)
                quality = np.isin(vrts[-1].read(1,window=win),(4,5,6,7))
                images, valids = [], []
                for vrt,(scale,offset) in zip(vrts[:-1],transforms[:-1]):
                    data = vrt.read(1,window=win,masked=True)
                    rho = data.data.astype('float32')*scale+offset
                    valid = ~np.ma.getmaskarray(data)&np.isfinite(rho)&quality
                    images.append(np.where(valid,rho,0));valids.append(valid)
                images, valids = np.asarray(images),np.asarray(valids)
                for old in chunk:
                    x = old['window'][0]
                    data = images[:,:,x:x+128].copy();valid = valids[:,:,x:x+128].copy()
                    assert data.shape==(10,128,128) and valid.all(0).mean()>=.8
                    assert abs(float(valid.all(0).mean())-old['valid_ratio'])<1e-12
                    target = folder/('f_'+old['key'].replace(':','_')+'.npz')
                    arrays = dict(x=data,pixel_valid=valid,metadata=metadata,known=known)
                    if target.exists():
                        with np.load(target) as previous:
                            if not all(np.array_equal(previous[k],v) for k,v in arrays.items()):
                                raise ValueError('Existing F cache changed')
                    else:
                        temp = target.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(target)
                    affine = window_transform(Window(*old['window']),blue.transform)
                    xmin,ymin,xmax,ymax = old['bbox_projected']
                    row = dict(file='f_local_v1/'+target.name,sha256=sha(target),sensor='s2',origin='f_local',
                        band_ids=list(bands),date=old['date'],source_item=old['item'],source_folder=name,
                        source_window=old['window'],crs=str(blue.crs),transform=list(affine)[:6],
                        bbox=old['bbox_wgs84'],bbox_projected=old['bbox_projected'],
                        physical_scale=.0001,physical_offset=-.1,units='surface_reflectance',
                        valid_ratio=float(valid.all(0).mean()),
                        group=f'{blue.crs}:{int(np.floor(xmin/5120))}:{int(np.floor(ymax/5120))}')
                    rows.append(row)
                if len(rows)%200<42:print('F caches',len(rows),flush=True)
    rows.sort(key=lambda r:r['file'])
    train, validation, guard = geographic_split(rows)
    manifest = OUT/'f_local_manifest.json'
    if manifest.exists() and read(manifest)!=rows:
        raise ValueError('Frozen F manifest differs; use a new version')
    manifest.write_text(json.dumps(rows,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    result = dict(ok=True,locations=len(rows),train_ids=[r['file'] for r in train],
        validation_ids=[r['file'] for r in validation],guard_ids=[r['file'] for r in guard],
        manifest_sha256=sha(manifest),selection_sha256=sha(source),protocol_sha256=sha(ROOT/'manifests/v4r1/protocol.json'),
        code_sha256=sha(Path(__file__)),source_hashes={str(base/(b+'_AOI.tif')):source_hashes[str(base/(b+'_AOI.tif'))] for base in map(Path,by_folder) for b in (*S2_WAVELENGTH_UM,'SCL')},
        split='seed1234; 5% of 5120m groups held; one 1280m tile buffer around validation, touching tiles also guarded',
        target='masked reflectance reconstruction only; no landslide labels',stats='reuse frozen TRAIN core25; no F validation fitting')
    (OUT/'f_local_acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    load_f_members()
    print(json.dumps(dict(locations=len(rows),train=len(train),validation=len(validation),guard=len(guard))),flush=True)


if __name__=='__main__':
    main()
