"""Rebind fixed R3 members to verified native reflectance in a separate release."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import requests
import xml.etree.ElementTree as ET
import rasterio
from rasterio.windows import Window

from train_pd_here_geotoken_dualvit_r3 import ROOT
from observation_model.metered_http import HTTPRangeFile
from pd_here_geotoken_dualvit_r3.data import geometry, write_json

BASE = ROOT / 'outputs/pd_here_geotoken_dualvit_r3'
RECEIPTS = ROOT / 'outputs/encoder_v3/data_download_plan'
CALIBRATION = ROOT / 'outputs/observation_model/encoder_v2/data_expansion/s2_calibrated_v2'
READ_TOLERANCE = 2e-7


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VerifiedRanges:
    def get(self, url, start, end):
        delimiter = '&' if '?' in url else '?'
        with requests.get(f'{url}{delimiter}r3range={start}-{end}',
                          headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},
                          timeout=(12,25),stream=True) as response:
            response.raise_for_status()
            span, _, size = response.headers.get('Content-Range','').partition('/')
            if response.status_code != 206 or span != f'bytes {start}-{end}':
                raise ValueError('Source returned a different HTTP byte range')
            data = response.raw.read(end-start+1)
            if len(data) != end-start+1:
                raise OSError('Incomplete HTTP byte range')
            return data,int(size)


ESA_PRODUCTS = {
    'S2B_58EDQ_20220202_0_L2A': (
        '58/E/DQ/2022/2/2','S2B_MSIL2A_20220202T223339_N0400_R029_T58EDQ_20220202T235249'),
    'S2B_44DNP_20221011_0_L2A': (
        '44/D/NP/2022/10/11','S2B_MSIL2A_20221011T033619_N0400_R032_T44DNP_20221011T054615'),
}


def esa_reference(row, cached):
    tile, expected = ESA_PRODUCTS[row['product']]
    endpoint = 'https://sentinel-s2-l2a.s3.eu-central-1.amazonaws.com/'
    url = f'{endpoint}tiles/{tile}/0/'
    response = requests.get(url+'productInfo.json',timeout=15)
    response.raise_for_status()
    product = response.json()
    if product['name'] != expected:
        raise ValueError('Original ESA product differs from the source STAC product URI')
    response = requests.get(endpoint+product['path']+'/metadata.xml',timeout=20)
    response.raise_for_status()
    xml = ET.fromstring(response.content)
    offsets = [float(e.text) for e in xml.iter() if e.tag.split('}')[-1]=='BOA_ADD_OFFSET']
    scales = [float(e.text) for e in xml.iter() if e.tag.split('}')[-1]=='BOA_QUANTIFICATION_VALUE']
    if len(offsets) != 13 or len(set(offsets)) != 1 or scales != [10000.]:
        raise ValueError('Original product calibration metadata is ambiguous')
    band = next(b for b in row['bands'] if b['band_key']=='B04')
    dn = native_pixels(url+'R10m/B04.jp2',band)
    valid = cached['B04__valid'] & ~np.ma.getmaskarray(dn) & (dn.data != 0)
    reference = (dn.data.astype('float32')+offsets[0])/scales[0]
    if valid.sum()<32:
        raise ValueError('Insufficient original ESA B04 overlap')
    difference = reference[valid]-cached['B04'][valid]
    delta = min((0.,.1),key=lambda candidate: abs(float(np.median(difference))-candidate))
    if not np.allclose(difference,delta,rtol=0,atol=READ_TOLERANCE):
        raise ValueError('Original ESA B04 pixels disagree beyond an additive correction')
    return dict(product=product['name'],source=url+'R10m/B04.jp2',
                compared=int(valid.sum()),product_offset=offsets[0]/scales[0],
                additive_correction=delta,max_absolute_error=float(np.abs(difference-delta).max()))



def native_pixels(path, band):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT='15',
                      GDAL_HTTP_MAX_RETRY='2', GDAL_HTTP_RETRY_DELAY='1'):
        remote = str(path).startswith('https://')
        opener = (lambda name, mode='r': HTTPRangeFile(path, VerifiedRanges())) if remote else None
        with rasterio.open('r3_asset'+Path(path).suffix if remote else path, opener=opener) as source:
            affine = band['transform']
            col, row = ~source.transform * (affine[2], affine[5])
            h, w = band['shape']
            if (str(source.crs) != band['crs'] or source.transform.a != affine[0]
                    or source.transform.e != affine[4] or source.transform.b or source.transform.d
                    or abs(col-round(col)) > 1e-6 or abs(row-round(row)) > 1e-6
                    or col < 0 or row < 0 or col+w > source.width or row+h > source.height):
                raise ValueError('Cache bounds are not an integer native crop of source')
            return source.read(1, window=Window(round(col), round(row), w, h), masked=True)


def audit_member(row, revision, sample):
    corrections, checks, errors = {}, {}, {}
    provenance = row['provenance']
    cache = Path(row['cache'])
    if digest(cache) != row['cache_sha256']:
        raise ValueError('Frozen cache digest differs')
    with np.load(cache, allow_pickle=False) as data:
        if provenance == 's2_calibration_resolved':
            checks['esa_reference'] = esa_reference(row,data)
        for band in row['bands']:
            key = band['band_key']
            try:
                offset = band['offset']
                if provenance == 's2_core10_completed':
                    if revision is None:
                        raise ValueError('Independent calibration lineage unavailable')
                    offset += revision['additive_correction']
                    receipt = RECEIPTS/'batch_a_s2_completion/raw'/row['id']/f'{key}.tif'
                elif provenance == 's2_calibration_resolved':
                    receipt = RECEIPTS/'calibration_resolution/raw'/row['id']/f'{key}.tif'
                    offset += checks['esa_reference']['additive_correction']
                else:
                    receipt = Path(band['source'])
                valid = data[key+'__valid']
                if not valid.any() or not np.isfinite(data[key][valid]).all():
                    raise ValueError('Invalid frozen QA or reflectance')
                path = receipt if receipt.is_file() else band.get('source_asset') if sample else None
                if path is None:
                    # Four cached 10m bands are verified against the original asset at group level.
                    if provenance != 's2_core10_completed' or band['native_pixel_support_m'] != [10,10]:
                        raise ValueError('Native raster missing')
                    path = None
                corrected = data[key] + np.float32(offset-band['offset'])
                if path is not None:
                    dn = native_pixels(path, band)
                    if (valid & (np.ma.getmaskarray(dn) | (dn.data == 0))).any():
                        raise ValueError('QA marks source nodata valid')
                    reference = dn.data.astype('float32')*band['scale']+offset
                    if not np.allclose(corrected[valid], reference[valid], rtol=0, atol=READ_TOLERANCE):
                        raise ValueError('Native DN and expected calibrated cache disagree')
                    checks[key] = dict(source=str(path), compared=int(valid.sum()),
                                       max_absolute_error=float(np.abs(corrected-reference)[valid].max()))
                corrections[key] = offset
            except (KeyError, ValueError, OSError) as error:
                errors[key] = str(error)
    return dict(offsets=corrections, pixel_checks=checks, errors=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=BASE/'texture_study')
    parser.add_argument('--retry', action='store_true', help='Recheck blocked members from an existing audit')
    args = parser.parse_args()
    rows = {role:[json.loads(l) for l in (BASE/f'{role}.jsonl').read_text().splitlines()]
            for role in ('train','val')}
    groups = defaultdict(list)
    revisions = {}
    for row in rows['train']+rows['val']:
        sidecar = CALIBRATION/'cache'/f"{row['id']}.json"
        revision = read(sidecar)['calibration_revision'] if sidecar.is_file() else None
        revisions[row['id']] = revision
        baseline = (revision or {}).get('source_baseline', row.get('calibration_baseline'))
        if baseline is None and '_N' in row['product']:
            baseline = row['product'].split('_N')[1].split('_')[0]
        offsets = tuple(sorted({(band['scale'], band['offset']) for band in row['bands']}))
        groups[(row['provenance'], baseline, offsets, (revision or {}).get('source_flag'),
                (revision or {}).get('additive_correction'))].append(row)
    examples = read(BASE/'subset700_10epochs/predictions_val/examples.json')['examples']
    sample_ids = {r['id'] for cases in examples.values() for r in cases}
    for members in groups.values():
        sample_ids.update(x['id'] for x in sorted(members,key=lambda r:r['id'])[:3])
    audit = {}
    members = rows['train']+rows['val']
    previous = args.out/'source_audit.json'
    manifests = {role:digest(BASE/f'{role}.jsonl') for role in rows}
    if args.retry:
        prior = read(previous)
        if list(prior['members']) != [row['id'] for row in members]:
            raise ValueError('Previous audit member identity changed')
        if prior.get('source_manifests',manifests) != manifests:
            raise ValueError('Previous audit source manifests changed')
        audit = prior['members']
        members = [row for row in members if row['id'] in prior['blockers']]
    def inspect(row):
        identifier = row['id']
        try:
            return identifier,audit_member(row,revisions[identifier],identifier in sample_ids)
        except (ValueError,OSError) as error:
            return identifier,dict(offsets={},pixel_checks={},errors={'member':str(error)})
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(identifier,result) in enumerate(pool.map(inspect,members),1):
            audit[identifier] = result
            if i%100==0:
                print(f'Audited {i}/1000 members',flush=True)
    blockers = {identifier:r['errors'] for identifier,r in audit.items() if r['errors']}
    report = dict(status='BLOCKED' if blockers else 'PASS', groups=[
        dict(provenance=k[0],baseline=k[1],calibration=k[2],provider_flag=k[3],delta=k[4],
             count=len(v),sample_ids=[r['id'] for r in v if r['id'] in sample_ids])
        for k,v in groups.items()], blockers=blockers, members=audit,source_manifests=manifests)
    write_json(args.out/'source_audit.json',report)
    if blockers:
        write_json(args.out/'dataset_identity.json',dict(status='BLOCKED',blocker_count=len(blockers)))
        print(f'Source gate BLOCKED on {len(blockers)} members. See source_audit.json.',flush=True)
        return
    # Publish a separate cache and manifest only after the comparison gate is complete.
    (args.out/'cache').mkdir(exist_ok=True)
    for role, members in rows.items():
        sealed=[]
        for row in members:
            offsets=audit[row['id']]['offsets']
            output=dict(row,bands=[dict(b,offset=offsets[b['band_key']]) for b in row['bands']])
            if any(offsets[b['band_key']] != b['offset'] for b in row['bands']):
                target=args.out/'cache'/f"{row['id']}.npz"
                with np.load(row['cache'],allow_pickle=False) as source:
                    arrays={k:source[k] for k in source.files}
                for b in row['bands']:
                    key=b['band_key']
                    arrays[key]=np.where(arrays[key+'__valid'],
                        arrays[key]+np.float32(offsets[key]-b['offset']),0)
                temp=target.with_suffix('.tmp.npz')
                np.savez_compressed(temp,**arrays)
                temp.replace(target)
                output.update(cache=str(target.resolve()),cache_sha256=digest(target))
            assert geometry(output)==geometry(row)
            sealed.append(output)
        (args.out/f'{role}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sealed),encoding='utf8')
    write_json(args.out/'dataset_identity.json',dict(status='PASS', train_sha256=digest(args.out/'train.jsonl'),
        val_sha256=digest(args.out/'val.jsonl'), source_audit_sha256=digest(args.out/'source_audit.json'),
        window_plan_sha256=digest(BASE/'window_plan.json')))


if __name__ == '__main__':
    main()
