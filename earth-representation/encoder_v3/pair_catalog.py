"""Complete metadata candidate join; no pair is certified from bounds or dates."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shapely
from shapely.affinity import translate
from rasterio.warp import transform_bounds
from .data import sha256
from .geometry_views import northing_shift


def native_window(row):
    crs=row['bands'][0]['crs'];regions=[]
    for band in row['bands']:
        h,w=band['shape'];a,b,left,d,e,top=band['transform']
        if b or d or a<=0 or e>=0:raise ValueError('Unsupported native affine')
        delta=northing_shift(band['crs'],crs)
        regions.append(shapely.box(left,top+e*h+delta,left+a*w,top+delta))
    region=shapely.union_all(regions)
    bbox=transform_bounds(crs,'EPSG:4326',*region.bounds,densify_pts=21)
    if not all(math.isfinite(value) for value in bbox) or bbox[0]>bbox[2]:raise ValueError('Invalid/antimeridian bounds require a separate geometry operator')
    acquired=datetime.fromisoformat(row['date'].replace('Z','+00:00'));explicit=acquired.tzinfo is not None
    return dict(row=row,crs=crs,region=region,bbox=shapely.box(*bbox),time=acquired.replace(tzinfo=timezone.utc) if not explicit else acquired.astimezone(timezone.utc),timezone_explicit=explicit)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--members',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    before=sha256(args.members)
    with args.members.open(encoding='utf-8-sig') as stream:rows=[json.loads(line) for line in stream if line.strip()]
    optical=[r for r in rows if r['role'] in ('TRAIN','VAL') and r['sensor'] in ('S2','L8','L9') and r['units']=='surface_reflectance']
    sources=[];targets=[];unsupported=[]
    for row in optical:
        try:item=native_window(row)
        except (ValueError,KeyError) as exc:unsupported.append(dict(id=row['id'],error=str(exc)));continue
        (sources if row['sensor']=='S2' else targets).append(item)
    tree=shapely.STRtree([r['bbox'] for r in targets]);candidates=[];counts=Counter();gaps=Counter()
    for source in sources:
        for index in tree.query(source['bbox'],predicate='intersects'):
            target=targets[int(index)];a=source['row'];b=target['row']
            if a['role']!=b['role']:counts['cross_role_bounds_excluded']+=1;continue
            gap=abs((source['time']-target['time']).total_seconds());area=None;shift=None
            try:
                shift=northing_shift(target['crs'],source['crs']);area=source['region'].intersection(translate(target['region'],yoff=shift)).area
                if area<=0:counts['bounds_false_positive_excluded']+=1;continue
                geometry='same_CRS' if target['crs']==source['crs'] else 'UTM_false_northing_only'
            except ValueError:geometry='common_projection_operator_required'
            bin_name='within_1h' if gap<=3600 else '1h_to_1d' if gap<=86400 else '1d_to_2d' if gap<=172800 else '2d_to_7d' if gap<=604800 else 'over_7d'
            gaps[bin_name]+=1;counts[a['role']+'/'+geometry]+=1
            candidates.append(dict(S2_id=a['id'],Landsat_id=b['id'],Landsat_instrument=b['instrument'],role=a['role'],time_gap_seconds=gap,timezones_explicit=source['timezone_explicit'] and target['timezone_explicit'],S2_crs=source['crs'],Landsat_crs=target['crs'],geometry_status=geometry,Landsat_northing_shift_m=shift,native_window_overlap_m2=area,v3_admitted=False,pending=['valid_QA_overlap','processing_version','registration','temporal_threshold','surface_change_risk','pair_role_and_SRF_review']))
    if sha256(args.members)!=before:raise ValueError('Inventory changed during candidate join')
    candidates.sort(key=lambda r:(r['time_gap_seconds'],r['S2_id'],r['Landsat_id']))
    report=dict(status='PAIR_METADATA_CANDIDATES_NOT_ADMITTED',members_sha256=before,optical_members=len(optical),S2_members=len(sources),Landsat_members=len(targets),counts=dict(counts),descriptive_time_bins=dict(gaps),candidate_pairs=len(candidates),admitted_pairs=0,unsupported_geometry=unsupported,pairs=candidates,imagery_or_header_reads=0,optimizer_updates=0,limitations=['All metadata candidates retained; no nearest-only teacher selection','Date bins describe availability and are not acceptance thresholds','Window intersection is not QA-valid support, image registration or unchanged-surface evidence','Cross-zone projections are flagged rather than warped; same-zone false northing is an exact coordinate view','No training masks, members, tasks or data roles changed'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('status','optical_members','candidate_pairs','admitted_pairs','counts','descriptive_time_bins')},indent=2))
