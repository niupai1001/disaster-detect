"""Read-only identity and cross-role native-footprint audit; no split changes."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import math
import shapely
from shapely.affinity import translate
from .pair_catalog import native_window
from .geometry_views import northing_shift
from .data import sha256


def shared_role_groups(rows,key):
    groups=defaultdict(list)
    for row in rows:
        value=row.get(key)
        if value:groups[str(value)].append(row)
    return [dict(value=value,roles=sorted({x['role'] for x in items}),ids=sorted(x['id'] for x in items)) for value,items in groups.items() if len({x['role'] for x in items})>1]


def audit(rows,protected):
    identities={key:shared_role_groups(rows,key) for key in ('cache_sha256','source_cache_sha256','acquisition_key','product','release_isolation_group')}
    items=[];unsupported=[]
    for row in rows:
        try:items.append(native_window(row))
        except (ValueError,KeyError) as e:unsupported.append(dict(id=row.get('id'),error=str(e)))
    train=[x for x in items if x['row']['role']=='TRAIN'];val=[x for x in items if x['row']['role']=='VAL'];tree=shapely.STRtree([x['bbox'] for x in val]);overlaps=[];touch_only=0
    for a in train:
        for j in tree.query(a['bbox'],predicate='intersects'):
            b=val[int(j)];area=None
            try:
                shift=northing_shift(b['crs'],a['crs']);area=a['region'].intersection(translate(b['region'],yoff=shift)).area
                if area<=0:touch_only+=1;continue
                geometry='EXACT_NATIVE_FOOTPRINT_OVERLAP'
            except ValueError:geometry='CROSS_PROJECTION_CANDIDATE_NOT_VERIFIED'
            overlaps.append(dict(train_id=a['row']['id'],val_id=b['row']['id'],train_sensor=a['row']['sensor'],val_sensor=b['row']['sensor'],time_gap_seconds=abs((a['time']-b['time']).total_seconds()),geometry=geometry,overlap_m2=area,same_acquisition_key=bool(a['row'].get('acquisition_key') and a['row'].get('acquisition_key')==b['row'].get('acquisition_key'))))
    regions=[];guards=[];unmeasured=[]
    for row in protected:
        bbox=row.get('bbox')
        if not isinstance(bbox,list) or len(bbox)!=4 or not all(math.isfinite(v) for v in bbox) or not (-180<=bbox[0]<bbox[2]<=180 and -90<=bbox[1]<bbox[3]<=90):
            unmeasured.append(row['id']);continue
        regions.append(shapely.box(*bbox));guards.append(row)
    tree=shapely.STRtree(regions);protected_candidates=[]
    for a in train:
        for j in tree.query(a['bbox'],predicate='intersects'):
            j=int(j)
            if a['bbox'].intersection(regions[j]).area<=0:continue
            protected_candidates.append(dict(train_id=a['row']['id'],protected_id=guards[j]['id'],protected_role=guards[j]['effective_partition'],status='DECLARED_BOUNDS_CANDIDATE_NOT_NATIVE_OR_QA_PROOF'))
    return dict(shared_cross_role_identities=identities,train_val_overlaps=overlaps,bounds_touch_or_false_positive_excluded=touch_only,protected_bounds_candidates=protected_candidates,unsupported_released_geometry=unsupported,protected_bbox_unmeasured=unmeasured,counts=dict(train=len(train),val=len(val),cross_role_exact_overlap=sum(x['geometry']=='EXACT_NATIVE_FOOTPRINT_OVERLAP' for x in overlaps),cross_role_projection_pending=sum(x['geometry']!='EXACT_NATIVE_FOOTPRINT_OVERLAP' for x in overlaps),train_members_in_exact_overlap=len({x['train_id'] for x in overlaps if x['geometry']=='EXACT_NATIVE_FOOTPRINT_OVERLAP'}),protected_declared_bounds_candidates=len(protected_candidates)),duplicate_manifest_ids=[key for key,n in Counter(row['id'] for row in rows).items() if n>1])


def main(args):
    before=sha256(args.members);canonical_hash=sha256(args.canonical)
    rows=[row for row in map(json.loads,args.members.open(encoding='utf-8-sig')) if row.get('role') in ('TRAIN','VAL') and row.get('units')=='surface_reflectance']
    protected=[row for row in map(json.loads,args.canonical.open(encoding='utf-8-sig')) if row.get('effective_partition') in ('GUARD','TEST_ISOLATION')]
    report=dict(status='CROSS_ROLE_METADATA_AUDIT_NOT_V3_SPLIT_ACCEPTANCE',members_sha256=before,canonical_sha256=canonical_hash,optimizer_updates=0,imagery_or_header_reads=0,split_or_member_changes=0,**audit(rows,protected),limits=['Shared scene/acquisition identity alone is not proof of overlapping pixels','Native footprint overlap is independent of QA, visibility and full Encoder receptive fields','Protected checks use declared geographic bounds only; protected imagery and headers never opened','Different projection candidates remain unresolved rather than being treated as safe','No observation removed, repartitioned or newly admitted'])
    if sha256(args.members)!=before or sha256(args.canonical)!=canonical_hash:raise ValueError('Inventory changed during audit')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(counts=report['counts'],identity_groups={k:len(v) for k,v in report['shared_cross_role_identities'].items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--members',type=Path,required=True);p.add_argument('--canonical',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
