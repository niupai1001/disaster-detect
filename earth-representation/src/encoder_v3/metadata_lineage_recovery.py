"""Recover local cached STAC evidence, never fabricate product sidecars or admit data."""
import json,math,subprocess,time
from pathlib import Path
import rasterio
from .data import sha256


def check_processing_identity(item,member,canonical):
    properties=item['properties'];revision=member['calibration_revision']
    if item['id']!=member['product'] or properties!=canonical['properties']:raise ValueError('Cached STAC item differs from frozen canonical properties')
    if properties['datetime']!=member['date'] or properties['s2:processing_baseline']!=revision['source_baseline'] or type(properties['earthsearch:boa_offset_applied']) is not bool or properties['earthsearch:boa_offset_applied']!=revision['source_flag']:raise ValueError('Processing or acquisition identity mismatch')
    keys=('datetime','platform','s2:processing_baseline','s2:product_uri','s2:generation_time','earthsearch:boa_offset_applied','earthsearch:s3_path','processing:software','created','updated')
    return {k:properties.get(k) for k in keys}


def native_crop_check(affine,shape,asset):
    source=asset['proj:transform'];source_shape=asset['proj:shape']
    if len(affine)!=6 or len(source)<6 or affine[1]!=0 or affine[3]!=0 or source[1]!=0 or source[3]!=0 or affine[0]<=0 or affine[4]>=0 or affine[0]!=source[0] or affine[4]!=source[4]:raise ValueError('Native pixel scale/axis mismatch')
    col=(affine[2]-source[2])/source[0];row=(affine[5]-source[5])/source[4]
    if any(not math.isfinite(v) or abs(v-round(v))>1e-8 for v in (col,row)):raise ValueError('Crop is not on the source native lattice')
    col,row=int(round(col)),int(round(row))
    if row<0 or col<0 or row+shape[0]>source_shape[0] or col+shape[1]>source_shape[1]:raise ValueError('Crop extends outside original asset')
    return dict(source_pixel_offset_rc=[row,col],native_shape=list(shape),native_gsd_xy=[affine[0],-affine[4]],resampled=False)


def main():
    started=time.perf_counter();reports=Path(__file__).resolve().parent/'reports';study=json.loads((reports/'pilot_study_candidate.json').read_text(encoding='utf-8'));release=Path(study['release']);members_path=release/'members.jsonl'
    if sha256(members_path)!=study['source_identities']['members']:raise ValueError('Frozen member identity changed')
    ids=json.loads((reports/'pair_admission_evidence.json').read_text(encoding='utf-8'))['missing_local_product_metadata_members'];members={x['id']:x for x in map(json.loads,members_path.open(encoding='utf-8-sig')) if x['id'] in ids}
    canonical_path=release.parent/'pre_G1_data_contract/v4/canonical_observations.jsonl';canonical_hash=sha256(canonical_path)
    if canonical_hash!=json.loads((reports/'native_lineage_audit.json').read_text(encoding='utf-8'))['canonical_sha256']:raise ValueError('Canonical identity changed')
    canonical={x['id']:x for x in map(json.loads,canonical_path.open(encoding='utf-8-sig')) if x['id'] in ids};base=Path('N:/Users/yzh/Disater/landslide-seg/outputs/use_data_integration/completion');command=['rg','-l','-F','-g','*.json']
    for row in members.values():command+=['-e',row['product']]
    command.append(str(base/'catalogue'));scan=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW);found={row['product']:[] for row in members.values()};source_files={}
    for name in scan.stdout.splitlines():
        path=Path(name);digest=sha256(path);data=json.loads(path.read_text(encoding='utf-8-sig'));source_files[str(path)]=digest
        for item in data.get('features',[]):
            if item['id'] in found:found[item['id']].append((str(path),item))
    rows=[]
    for identifier in ids:
        member=members[identifier];original=canonical[identifier]
        if member['role']!='TRAIN' or member['source_role']!='TRAIN' or original['effective_partition']!='TRAIN':raise ValueError('TRAIN source lineage only')
        candidates=found[member['product']]
        if not candidates:raise ValueError('Original cached STAC item not found')
        for _,item in candidates:check_processing_identity(item,member,original)
        item=candidates[0][1];completion=base/'results'/(original['source_id'].removeprefix('remote_')+'.json');source_files[str(completion)]=sha256(completion);record=json.loads(completion.read_text(encoding='utf-8-sig'))
        if record['source_item']!=member['product'] or record['date']!=member['date'] or record['role']!='TRAIN':raise ValueError('Original completion record mismatch')
        matched=[]
        for band in member['bands']:
            key=band['band_key'];source=record['calibration'][key]['source'];path=Path(band['source']);asset=source['asset_metadata'];assets=[value for value in item['assets'].values() if value.get('href')==asset['href']]
            if str(path)!=source['path'] or not assets or any(value!=asset for value in assets):raise ValueError('Asset metadata/path mismatch')
            digest=sha256(path)
            if digest!=source['sha256'] or digest!=original['bands'][key]['expected_sha256']:raise ValueError('Reflectance byte identity changed')
            source_files[str(path)]=digest
            with rasterio.open(path) as dataset:
                if dataset.crs.to_epsg()!=item['properties']['proj:epsg']:raise ValueError('Native CRS differs from STAC')
                crop=native_crop_check(tuple(dataset.transform)[:6],dataset.shape,asset)
            matched.append(dict(band=key,path=str(path),sha256=digest,asset_href=asset['href'],scale=record['calibration'][key]['scale'],declared_offset=record['calibration'][key]['offset'],**crop))
        qa=record['quality_sources']['scl'];qa_path=Path(qa['path']);digest=sha256(qa_path)
        if digest!=qa['sha256'] or item['assets']['scl']!=qa['asset_metadata']:raise ValueError('QA byte/asset identity mismatch')
        source_files[str(qa_path)]=digest
        with rasterio.open(qa_path) as dataset:qa_grid=native_crop_check(tuple(dataset.transform)[:6],dataset.shape,qa['asset_metadata'])
        rows.append(dict(id=identifier,product=member['product'],cached_STAC_files=[name for name,_ in candidates],properties_exactly_equal_frozen_canonical=True,processing=check_processing_identity(item,member,original),completion_record=str(completion),bands=matched,SCL=dict(path=str(qa_path),sha256=digest,**qa_grid),local_product_json_created=False))
    for path,digest in source_files.items():
        if sha256(path)!=digest:raise ValueError('Source changed during metadata/asset verification')
    if sha256(canonical_path)!=canonical_hash:raise ValueError('Canonical changed during verification')
    report=dict(status='CACHED_ORIGINAL_METADATA_RECOVERED_NOT_ADMISSION',members=len(rows),native_bands=sum(len(x['bands']) for x in rows),native_QA=len(rows),source_identities=study['source_identities'],canonical_sha256=canonical_hash,rows=rows,source_files=source_files,optimizer_updates=0,admitted_pairs=0,seconds=time.perf_counter()-started,source_code_sha256=sha256(Path(__file__)),limitations=['Original product-folder JSON remains absent; original cached STAC response is separately traced','Cached metadata authenticity is local lineage evidence, not a new provider certification','No SRF, QA, temporal or registration policy adopted; no files moved, recreated or downloaded'])
    (reports/'pair_metadata_recovery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:report[k] for k in ('status','members','native_bands','native_QA','seconds')},indent=2))


if __name__=='__main__':main()
