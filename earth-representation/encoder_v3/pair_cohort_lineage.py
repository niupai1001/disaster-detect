"""Read-only pair-cohort QA lineage evidence; no source/pair admission."""
import argparse
import json
from pathlib import Path
import time
import warnings
import numpy as np
import rasterio
from .audit_lineage import audit_member
from .data import sha256


def main(index=None,output=None):
    reports=Path(__file__).resolve().parent/'reports';study=json.loads((reports/'pilot_study_candidate.json').read_text(encoding='utf-8'))
    index_path=reports/'pair_registration_diagnostics/index.json' if index is None else Path(index)
    output_path=reports/'pair_cohort_QA_lineage.json' if output is None else Path(output)
    index_hash=sha256(index_path);cohort=json.loads(index_path.read_text(encoding='utf-8'))
    if cohort['status'].startswith('RUNNING'):raise ValueError('Cohort measurement must finish before lineage/header report')
    pairs=cohort['rows']
    canonical_path=Path(study['release']).parent/'pre_G1_data_contract/v4/canonical_observations.jsonl'
    expected=json.loads((reports/'native_lineage_audit.json').read_text(encoding='utf-8'))['canonical_sha256']
    if sha256(canonical_path)!=expected:raise ValueError('Canonical inventory changed')
    identities={x[k] for x in pairs for k in ('source_id','target_id')}
    canonical={row['id']:row for row in map(json.loads,canonical_path.open(encoding='utf-8-sig')) if row['id'] in identities}
    members={row['id']:row for row in map(json.loads,(Path(study['release'])/'members.jsonl').open(encoding='utf-8-sig')) if row['id'] in identities}
    rows=[];started=time.perf_counter()
    for identifier in sorted(identities):
        row=members[identifier]
        if row['role']!='TRAIN':raise ValueError('Only TRAIN cohort may be read')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');result=audit_member(row,canonical.get(identifier),Path(study['release']))
        headers=[];metadata_files=[]
        for name,identity in result['source_files'].items():
            p=Path(name)
            if p.suffix.lower() not in ('.tif','.tiff'):continue
            if p.name.lower() not in ('scl.tif','qa_pixel.tif','qa_radsat.tif','qa_aerosol.tif') and p.parent.name!='raw_QA':continue
            with rasterio.open(p) as dataset:
                values,counts=np.unique(dataset.read(1),return_counts=True)
                headers.append(dict(path=name,sha256=identity['sha256'],shape=list(dataset.shape),crs=str(dataset.crs),affine=list(dataset.transform)[:6],dtype=dataset.dtypes[0],nodata=dataset.nodata,raw_value_histogram={str(int(v)):int(n) for v,n in zip(values,counts)}))
        source_paths=[Path(b['source']) for b in row['bands'] if b.get('source')]
        for folder in {p.parent for p in source_paths}:
            metadata_files.extend(str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.json','.xml','.txt'))
        processing=[]
        for path in sorted(set(metadata_files)):
            if Path(path).suffix.lower()!='.json':continue
            digest=sha256(path);item=json.loads(Path(path).read_text(encoding='utf-8-sig'));properties=item.get('properties',{})
            processing.append(dict(path=path,sha256=digest,item_id=item.get('id'),acquired=properties.get('datetime'),acquisition_timestamp_exact_match=properties.get('datetime')==row['date'],platform=properties.get('platform'),description=properties.get('description'),collection_number=properties.get('landsat:collection_number'),collection_category=properties.get('landsat:collection_category'),correction=properties.get('landsat:correction'),created=properties.get('created'),note='Created is catalog metadata; not inferred to be acquisition or processing time'))
            if sha256(path)!=digest:raise ValueError('Original metadata changed during audit')
        rows.append(dict(id=identifier,role=row['role'],product=row['product'],timestamp=row['date'],release_isolation_group=row['release_isolation_group'],processing_domain=row.get('processing_domain'),calibration_revision=row.get('calibration_revision'),raw_QA_independently_recomputed=result['raw_QA_recomputed'],bands=result['bands'],warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],qa_headers=headers,metadata_files=sorted(set(metadata_files)),processing_metadata=processing,source_files=result['source_files']))
    if sha256(index_path)!=index_hash or sha256(canonical_path)!=expected:raise ValueError('Cohort inventory changed during lineage/header audit')
    report=dict(input_index=str(index_path),input_index_sha256=index_hash,source_sha256=sha256(Path(__file__)),status='PAIR_COHORT_LINEAGE_RECHECK_NOT_ADMISSION',pairs=len(pairs),members=len(rows),raw_QA_members=sum(row['raw_QA_independently_recomputed'] for row in rows),parent_QA_only_members=sum(not row['raw_QA_independently_recomputed'] for row in rows),canonical_sha256=expected,optimizer_updates=0,admitted_pairs=0,seconds=time.perf_counter()-started,rows=rows,limitations=['Historical QA rules are reproduced, not newly adopted as V3 scientific policy','Parent-only original QA_PIXEL/RADSAT still unavailable in this checked cohort','Original processing metadata read only where locally available; missing fields not inferred from short product IDs','No registration threshold, temporal threshold or surface-change classification adopted'])
    output_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:report[k] for k in ('status','members','raw_QA_members','parent_QA_only_members','seconds')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args();main(args.index,args.output)
