"""Read-only F/G optical admission audit. Never downloads or changes frozen datasets."""
import hashlib
import argparse
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, bounds
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely import transform as transform_geometry
from shapely.strtree import STRtree

from prepare_use_external import exclusion

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/use/fg_audit'
FROOT = Path('F:/data/wildfire_20230420_sentinel2')
GROOT = Path('G:/damagedata/rs')
BANDS = ('B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12')
REQUIRED_CODES = {1,2,3,4,5,6,7,8,11,12}


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rejected(geometry, tree):
    return len(tree.query(geometry, predicate='intersects')) > 0


def retain_observation(selected, row):
    key = row['key']
    if key not in selected or (-row['valid_ratio'],row['date']) < (-selected[key]['valid_ratio'],selected[key]['date']):
        selected[key] = row


def audit_f(forbidden, existing):
    catalog = {r['id']: r for r in read(FROOT/'catalog_candidates.geojson')['features']}
    manifest = read(FROOT/'download_manifest.json')
    sources, selected, scenes, grid = [], {}, [], None
    acquisition_filtered_locations = {}
    observation_rows = []
    for item, folder in sorted(manifest['selected_items'].items()):
        base = FROOT/'imagery'/folder
        metadata = {b: read(base/(b+'_AOI.json')) for b in BANDS}
        metadata['SCL'] = read(base/'SCL_AOI.json')
        # Hash every optical/quality source used; a historical manifest alone is insufficient.
        for band, info in metadata.items():
            path = base/(band+'_AOI.tif')
            digest = sha(path)
            assert digest == info['sha256'], f'Source changed: {path}'
            sources.append(dict(path=str(path), sha256=digest))
        with ExitStack() as stack:
            blue = stack.enter_context(rasterio.open(base/'B02_AOI.tif'))
            assert blue.crs and blue.res == (10,10)
            identity = (str(blue.crs), tuple(blue.transform), blue.width, blue.height)
            if grid is None:
                grid = identity
            assert grid == identity, 'Dates require aligned grids before location deduplication'
            optical = []
            for band in BANDS:
                source = stack.enter_context(rasterio.open(base/(band+'_AOI.tif')))
                info = metadata[band]
                assert source.count == 1 and str(source.crs) == info['crs']
                assert source.scales[0] == info['scale'] and source.offsets[0] == info['offset']
                assert np.isfinite([info['scale'],info['offset']]).all() and info['scale'] > 0
                assert info['item'] == item and info['source_url']
                optical.append(stack.enter_context(WarpedVRT(source, crs=blue.crs,
                    transform=blue.transform, width=blue.width, height=blue.height,
                    resampling=Resampling.bilinear)))
            scl = stack.enter_context(rasterio.open(base/'SCL_AOI.tif'))
            quality = stack.enter_context(WarpedVRT(scl, crs=blue.crs, transform=blue.transform,
                width=blue.width, height=blue.height, resampling=Resampling.nearest))
            counters = Counter()
            project = Transformer.from_crs(blue.crs, 4326, always_xy=True).transform
            cloud = float(catalog[item]['properties']['eo:cloud_cover'])
            for y in range(0, blue.height-127, 128):
                # Stripes bound RAM while avoiding re-reading the same COG blocks per window.
                stripe = Window(0,y,blue.width,128)
                valid = np.isin(quality.read(1,window=stripe), (4,5,6,7))
                for band, vrt in zip(BANDS,optical):
                    data = vrt.read(1,window=stripe,masked=True)
                    rho = data.data.astype(np.float32)*metadata[band]['scale']+metadata[band]['offset']
                    valid &= ~np.ma.getmaskarray(data) & np.isfinite(rho)
                for x in range(0,blue.width-127,128):
                    counters['full_windows'] += 1
                    fraction = float(valid[:,x:x+128].mean())
                    if fraction < .8:
                        counters['quality_below_80pct'] += 1
                        continue
                    bb = bounds(Window(x,y,128,128),blue.transform)
                    geo = transform_geometry(box(*bb),project,interleaved=False)
                    if rejected(geo,forbidden):
                        counters['project_event_buffer'] += 1
                        continue
                    if rejected(geo,existing):
                        counters['existing_sample_overlap'] += 1
                        continue
                    counters['accepted_observation_windows'] += 1
                    if cloud < 20:
                        counters['also_scene_cloud_below_20'] += 1
                    key = f'{y}:{x}'
                    row = dict(key=key,item=item,folder=str(base),window=[x,y,128,128],
                        valid_ratio=fraction,scene_cloud_percent=cloud,crs=str(blue.crs),
                        bbox_projected=list(bb),bbox_wgs84=list(geo.bounds),
                        date=catalog[item]['properties']['datetime'],
                        bands=list(BANDS),physical_scale=.0001,physical_offset=-.1)
                    observation_rows.append(row)
                    if cloud < 20:
                        retain_observation(acquisition_filtered_locations,row)
                    # Keep one clearest date per location; equal-quality ties use earliest date.
                    retain_observation(selected,row)
                if y % (128*10) == 0:
                    print('F stripe',folder,y,'accepted',counters['accepted_observation_windows'],flush=True)
            counters['edge_pixels_discarded'] = blue.width*blue.height-(blue.width//128)*(blue.height//128)*128*128
            scenes.append(dict(item=item,folder=folder,scene_cloud_percent=cloud,counts=dict(counters)))
            print('F scene complete',folder,dict(counters),flush=True)
    rows = sorted(selected.values(),key=lambda r:r['key'])
    (OUT/'f_qualified_locations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (OUT/'f_qualified_observations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in observation_rows),encoding='utf-8')
    strict_rows = sorted(acquisition_filtered_locations.values(),key=lambda r:r['key'])
    (OUT/'f_strict_qualified_locations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in strict_rows),encoding='utf-8')
    return dict(scenes=scenes,unique_qualified_locations=len(rows),source_hashes=sources,
        unique_locations_also_scene_cloud_below_20=len(acquisition_filtered_locations),
        all_dates_same_grid=True,bands=list(BANDS),location_selection='highest common valid ratio; earliest date for ties',
        admission='one 128x128 window at 10m; common finite/not-nodata optical and SCL4/5/6/7 ratio >=0.8; outside all project event envelopes+10km and existing accepted windows',
        scene_cloud_boundary='Scene cloud <20% is reported separately as the old acquisition prefilter, not substituted for pixel quality; scenes above it remain flagged.')


def audit_g():
    groups = defaultdict(dict)
    ancillary, files = Counter(), 0
    bundles, quality_files = [], []
    for folder in sorted(GROOT.iterdir()):
        if not folder.is_dir() or folder.name.startswith('.'):
            continue
        for f in folder.iterdir():
            if any(token in f.name.upper() for token in ('SCL','QA_PIXEL','QA60','MSK_CLD','CLOUDMASK')):
                quality_files.append(str(f))
            match = re.search(r'_IM(\d{8})_F(\d+)\.tif$',f.name,re.I)
            if match:
                groups[(folder.name,match[1])][int(match[2])] = f
                files += 1
            else:
                ancillary[f.suffix.lower()] += 1
                if f.suffix.lower() == '.tif':
                    bundles.append(f)
    # Review a blue-band header for EVERY full ten-band date, not one arbitrary example.
    complete = [(k,v) for k,v in groups.items() if REQUIRED_CODES <= v.keys()]
    def header(entry):
        key, bands = entry
        p = bands[1]
        try:
            with rasterio.open(p) as ds:
                return dict(site=key[0],date=key[1],source=str(p),crs=str(ds.crs),shape=[ds.height,ds.width],
                    description=ds.descriptions[0],dtype=ds.dtypes[0],scale=ds.scales[0],offset=ds.offsets[0],
                    tags=ds.tags(),band_tags=ds.tags(1),nodata=ds.nodata)
        except Exception as exc:
            return dict(site=key[0],date=key[1],source=str(p),error=str(exc))
    with ThreadPoolExecutor(max_workers=8) as executor:
        headers = list(executor.map(header,complete))
    def bundle_header(p):
        with rasterio.open(p) as ds:
            return dict(file=str(p),count=ds.count,dtypes=ds.dtypes,descriptions=ds.descriptions)
    with ThreadPoolExecutor(max_workers=8) as executor:
        bundle_headers = list(executor.map(bundle_header,bundles))
    assert not quality_files, 'Quality files discovered: revisit G admission rather than rejecting blindly'
    assert all(r['count']==3 and set(r['dtypes'])=={'uint8'} for r in bundle_headers), 'Unexpected extra raster: inspect its bands before deciding G admission'
    (OUT/'g_rgb_headers.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in bundle_headers),encoding='utf-8')
    (OUT/'g_complete_headers.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in headers),encoding='utf-8')
    (OUT/'g_groups.jsonl').write_text(''.join(json.dumps(dict(site=k[0],date=k[1],codes=sorted(v)),ensure_ascii=False)+'\n' for k,v in sorted(groups.items())),encoding='utf-8')
    return dict(root=str(GROOT),files=files,sites=len({k[0] for k in groups}),site_date_combinations=len(groups),
        complete_ten_band_dates=len(complete),complete_ten_band_sites=len({k[0] for k,_ in complete}),
        ancillary_extensions=dict(ancillary),header_failures=sum('error' in r for r in headers),
        quality_filename_candidates=quality_files,rgb_browse_files=len(bundle_headers),
        blue_header_signatures=dict(Counter(str((r.get('dtype'),r.get('scale'),r.get('offset'),r.get('description'))) for r in headers)),
        usable_under_current_admission=0,
        blocked_reasons=['No per-observation SCL/QA files in the audited source folders; S1 VV/VH and thermal bands cannot replace optical cloud/quality masks.',
                         'Filename-band map is supported by the downloader source, but historical uint16 TIFFs lack per-product reflectance/processing provenance. Current exporter code is not proof of old-file scale or product level.'],
        boundary='Zero means not admitted with available evidence, not that all optical pixels are bad. No label mask used as input quality. No G pixel admission claimed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g-only',action='store_true',help='Update G metadata audit while retaining the prior F pixel audit')
    parser.add_argument('--f-only',action='store_true',help='Update F window audit while retaining the G metadata audit')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.g_only:
        result = read(OUT/'summary.json')
        result['G'] = audit_g()
        result['g_supplement_script_sha256'] = sha(Path(__file__))
        (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('G supplement complete',result['G']['complete_ten_band_dates'],result['G']['rgb_browse_files'],flush=True)
        return
    forbidden = STRtree(exclusion())
    remote = read(ROOT/'outputs/use/external/current_manifest.json')
    local = read(ROOT/'outputs/use/external/local_unlabeled_manifest.json')
    geos = [shape(r['geometry']) if r.get('geometry') else box(*r['bbox']) for r in remote+local]
    existing = STRtree(geos)
    if args.f_only:
        result = read(OUT/'summary.json')
        result['F'] = audit_f(forbidden,existing)
        result['f_supplement_script_sha256'] = sha(Path(__file__))
        (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('F supplement complete',result['F']['unique_qualified_locations'],result['F']['unique_locations_also_scene_cloud_below_20'],flush=True)
        return
    result = dict(as_of='2026-10-24',script_sha256=sha(Path(__file__)),read_only_sources=True,
        source_scope=[str(FROOT/'imagery'),str(GROOT)],new_training_or_download=False,
        protocol_sha256=sha(ROOT/'manifests/v4r1/protocol.json'))
    result['G'] = audit_g()
    print('G inventory complete',result['G']['complete_ten_band_dates'],flush=True)
    result['F'] = audit_f(forbidden,existing)
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert result['F']['unique_qualified_locations'] <= sum(s['counts'].get('accepted_observation_windows',0) for s in result['F']['scenes'])
    print('DONE unique F locations',result['F']['unique_qualified_locations'],flush=True)


if __name__ == '__main__':
    main()
