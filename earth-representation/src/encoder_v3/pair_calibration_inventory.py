"""Inventory calibration candidates using local metadata/headers; no pixel decoding."""
import hashlib,json,time
from pathlib import Path
import rasterio


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def candidate_match(row,member):
    # Exact product references or exact sensor/time are candidates, never proof.
    return member['product'] in json.dumps(row) or (row.get('sensor')==member['sensor'].lower() and (row.get('date') or row.get('datetime'))==member['date'])


def hash_values(value):
    if isinstance(value,dict):return set().union(set(),*(hash_values(v) for v in value.values()))
    if isinstance(value,list):return set().union(set(),*(hash_values(v) for v in value))
    if isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value):return {value}
    return set()


def main():
    started=time.perf_counter();root=Path(__file__).resolve().parent;r=root/'reports';output=r/'pair_calibration_inventory.json'
    if output.exists():raise ValueError('Preserve committed inventory')
    study=json.loads((r/'pilot_study_candidate.json').read_text(encoding='utf-8'));proposal=json.loads((r/'B2_pair_geometry_proposal.json').read_text(encoding='utf-8'));lineage=json.loads((r/'pair_full_QA_lineage.json').read_text(encoding='utf-8'))
    native=Path(study['release'])/'members.jsonl';active=Path('N:/Users/yzh/Disater/landslide-seg/outputs/use_data_integration/active_release/manifest.jsonl')
    paths=[native,active,r/'pilot_study_candidate.json',r/'B2_pair_geometry_proposal.json',r/'pair_full_QA_lineage.json'];identities={str(p):digest(p) for p in paths}
    if identities[str(native)]!=study['source_identities']['members']:raise ValueError('Native membership changed')
    rows=[json.loads(x) for x in native.read_text(encoding='utf-8-sig').splitlines() if x.strip()];other=[json.loads(x) for x in active.read_text(encoding='utf-8-sig').splitlines() if x.strip()];by_id={x['id']:x for x in rows};qa={x['id']:x for x in lineage['rows']};members=[]
    for case in proposal['candidate']['cases']:
        if not case['initial_proposal_case']:continue
        for key in ('S2_id','Landsat_id'):
            a=by_id[case[key]];sources=qa[a['id']]['source_files'];source_hashes={v['sha256'] for k,v in sources.items() if Path(k).suffix.lower() in ('.tif','.tiff')};matches=[];headers=[]
            assert a['role']=='TRAIN'
            for b in other:
                if candidate_match(b,a):matches.append(dict(id=b['id'],role=b['role'],sensor=b['sensor'],crs=b.get('crs'),transform=b.get('transform'),shape=b.get('shape',b.get('cache_audit',{}).get('shape')),shared_raw_raster_hashes=sorted(hash_values(b)&source_hashes),independence_established=False))
            for path,record in sources.items():
                if Path(path).suffix.lower() not in ('.tif','.tiff'):continue
                if digest(path)!=record['sha256']:raise ValueError('Raw raster changed')
                with rasterio.open(path) as src:headers.append(dict(path=path,sha256=record['sha256'],shape=list(src.shape),crs=src.crs.to_string(),affine=list(src.transform)[:6],bounds=list(src.bounds)))
                if digest(path)!=record['sha256']:raise ValueError('Raster changed during header read')
            members.append(dict(id=a['id'],product=a['product'],role=a['role'],native_same_product_members=[dict(id=b['id'],role=b['role'],grid=b['bands'][0]) for b in rows if b['product']==a['product'] and b['instrument']==a['instrument']],active_release_candidates=matches,raw_headers=headers))
    if any(digest(p)!=identities[str(p)] for p in paths):raise ValueError('Inventory source changed')
    assert len(members)==4 and all(len(x['native_same_product_members'])==1 for x in members)
    report=dict(status='LOCAL_METADATA_INVENTORY_NOT_CALIBRATION_OR_ADMISSION',members=members,native_metadata_rows=len(rows),active_metadata_rows=len(other),source_reports=identities,source_sha256=digest(Path(__file__)),independent_additional_windows_established=0,decoded_image_pixels=0,offsets_estimated=0,training_targets_changed=False,optimizer_updates=0,pairs_admitted=0,seconds=time.perf_counter()-started,limits=['Exact product references and exact sensor/timestamp are metadata candidates only; references may identify the same original crop','Shared raw-file hashes demonstrate duplicate input lineage, not an independent calibration sample','Full satellite asset dimensions in remote STAC metadata do not establish locally available pixels beyond downloaded crops','Metadata outside TRAIN read only; no non-TRAIN pixels decoded','No claim that all local archives or unindexed sources were exhausted; global filename audit remains incomplete','Calibration/validation isolation and specific offsets still require scientific review; existing spatial holdout diagnostics do not freeze either'])
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],members=len(members),active_candidates=sum(len(x['active_release_candidates']) for x in members),shared_lineage_candidates=sum(bool(b['shared_raw_raster_hashes']) for x in members for b in x['active_release_candidates']),seconds=report['seconds'])))


if __name__=='__main__':main()
