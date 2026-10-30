"""Reaudit frozen USE caches, publish an input release and offline counterpart requests."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.ops import transform
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from landslide_seg.data.use_admission import sha256
from audit_use_independent_data import members, audit_cache, EXTERNAL, read


def footprint(row):
    # Use actual grid corners rather than the enclosing WGS84 bounding rectangle.
    a, b, c, d, e, f = row['transform']
    _, h, w = row['cache_audit']['shape']
    polygon = Polygon([(a*x+b*y+c, d*x+e*y+f) for x,y in ((0,0),(w,0),(w,h),(0,h))])
    return transform(Transformer.from_crs(row['crs'], 4326, always_xy=True).transform, polygon)


def counterpart_requests(rows):
    geometries = [footprint(r) for r in rows]
    tree = STRtree(geometries)
    pairs, requests = [], []
    cross_role = []
    for i, (row, geo) in enumerate(zip(rows, geometries)):
        if row['role'] == 'GUARD':
            continue
        matches = []
        project = Transformer.from_crs(4326, row['crs'], always_xy=True).transform
        anchor = transform(project, geo)
        for j in tree.query(geo, predicate='intersects'):
            other = rows[int(j)]
            if j == i or other['sensor'] == row['sensor']:
                continue
            candidate = transform(project, geometries[int(j)])
            coverage = anchor.intersection(candidate).area / anchor.area
            if coverage < .8:
                continue
            if row['role'] != other['role']:
                cross_role.append([row['file'], other['file']])
                continue
            matches.append((coverage, other['file']))
        if matches:
            coverage, name = max(matches)
            pairs.append(dict(anchor=row['file'], counterpart=name, role=row['role'],
                anchor_coverage=coverage, status='SPATIAL_ONLY_NOT_PIXEL_ALIGNMENT_CERTIFIED'))
        else:
            date = row.get('datetime', row.get('date'))
            requests.append(dict(request_id='pair_' + sha256_text(row['file'])[:20],
                anchor_file=row['file'], anchor_sha256=row['sha256'], role=row['role'],
                target_sensor='landsat' if row['sensor']=='s2' else 's2',
                bbox_wgs84=list(geo.bounds), geometry_wgs84=geo.__geo_interface__, anchor_date=date,
                target_coverage_min=.8, target_clear_fraction_min=.8,
                role_rule='Inherit anchor role; overlap with any different-role/guard/protected area blocks admission',
                date_rule='Single observation; nearest clear acquisition preferred; record date gap; no time series required',
                grid_rule='Clip to common physical footprint; preserve 10m S2 / 30m Landsat native GSD',
                status='PLANNED_NOT_DOWNLOADED'))
    return pairs, requests, cross_role


def sha256_text(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/use_data_admission')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, sources = members()
    failures = []
    for i, row in enumerate(rows):
        try:
            row['cache_audit'] = audit_cache(EXTERNAL, row)
        except (OSError, ValueError, KeyError) as exc:
            failures.append(dict(file=row['file'], error=str(exc)))
        if (i+1)%500 == 0:
            print(f'audit {i+1}/{len(rows)} failures={len(failures)}', flush=True)
    if failures:
        (args.output/'failures.json').write_text(json.dumps(failures, indent=2), encoding='utf8')
        raise ValueError('No release: failed frozen cache audit')
    pairs, requests, cross_role = counterpart_requests(rows)
    if cross_role:
        raise ValueError('Cross-role spatial counterpart conflicts: '+str(cross_role[:5]))
    audit_path = ROOT/'outputs/use_next/local_inventory_audit/audit.json'
    pilot_path = ROOT/'data/use_multisensor/four_view_pilot_v1/manifest.json'
    inventory = read(audit_path)
    sources[str(audit_path.relative_to(ROOT))] = sha256(audit_path)
    sources[str(pilot_path.relative_to(ROOT))] = sha256(pilot_path)
    sources['outputs/data_inventory_20261027/README.md'] = sha256(ROOT/'outputs/data_inventory_20261027/README.md')
    candidates = [
        dict(source='F_other_dates_strict', count=inventory['F']['strict_additional_by_role']['TRAIN'], unit='TRAIN observations', status='PENDING', gates=['recheck_source_hash_and_SCL','materialize_cache','inherit_frozen_location_role','deduplicate']),
        dict(source='F_other_dates_relaxed', count=3595, unit='TRAIN observations including strict 590', status='PENDING_POLICY_CHANGE', gates=['explicit_scene_filter_change','same_gates_as_strict']),
        dict(source='E_S2', count=inventory['E']['complete_main10_stacks'], unit='multiband observations', status='PENDING', gates=['product_scale_offset','cloud_quality','native_band_identity','full_content_dedup','geometry_and_split']),
        dict(source='firepaper', count=inventory['firepaper']['tif_count'], unit='multiband files', status='PENDING', gates=['acquisition_or_composite_provenance','cloud_quality','scale_offset','dedup','geometry_and_split']),
        dict(source='G_old_source', status='OFFLINE_NOT_ADMITTED'),
        dict(source='nine_channel_landslide', status='EXCLUDED_UNKNOWN_BAND_IDENTITY'),
        dict(source='HSS_UAV_global_products_static_satellite_SLC', status='OUTSIDE_CURRENT_OPTICAL_INPUT_CONTRACT'),
        dict(source='TerraMesh', status='NOT_DOWNLOADED_NOT_ADMITTED'),
    ]
    for site in read(pilot_path):
        candidates.append(dict(source='four_view_pilot', site_id=site['site_id'], anchor_root=site['anchor_root'],
            status='PENDING_NOT_TRAINING', role=site['role'], gates=['L2A_product_calibration','pixel_quality','location_role_isolation','paired_common_footprint','exclude_L1C_quality_assets','QA_PIXEL_QA_RADSAT_required']))
    def write_json(name, value):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    def write_lines(name, values):
        with (args.output/name).open('w', encoding='utf8') as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False)+'\n')
    write_lines('admitted.jsonl', rows)
    write_lines('spatial_pairs.jsonl', pairs)
    write_lines('completion_requests.jsonl', requests)
    write_lines('candidate_gates.jsonl', candidates)
    config = read(ROOT/'outputs/use/pretraining/resolved_config.json')
    write_json('normalization.json', config['stats'])
    summary = dict(roles=dict(Counter(r['role'] for r in rows)),
        role_sensor=dict(Counter(r['role']+'/'+r['sensor'] for r in rows)),
        spatial_pair_anchors=len(pairs), missing_counterparts=dict(Counter(r['role']+'/'+r['target_sensor'] for r in requests)),
        newly_admitted_observations=0, cross_role_counterpart_conflicts=len(cross_role),
        limitations=['Full frozen cache hashes/arrays verified; source products and historical cloud algorithms not reaudited',
                    'Candidate counts reuse frozen inventory evidence, not new full-drive pixel scans',
                    'Spatial pairing does not certify date compatibility or subpixel registration',
                    'No downloads, training, original cache changes or new statistics fitting'])
    artifacts = {p.name:sha256(p) for p in args.output.iterdir() if p.is_file()}
    sources['scripts/build_use_data_admission.py'] = sha256(Path(__file__))
    sources['src/landslide_seg/data/use_admission.py'] = sha256(ROOT/'src/landslide_seg/data/use_admission.py')
    write_json('release.json', dict(ok=True, schema_version=1, project_root=str(ROOT),
        created_utc=datetime.now().astimezone().isoformat(), source_sha256=sources, artifacts=artifacts, summary=summary))
    print(json.dumps(summary,ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
