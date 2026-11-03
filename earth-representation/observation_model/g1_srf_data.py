"""G1: local-only SRF ingestion and physical input acceptance; no model imports."""
from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import tarfile
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from .expansion_data import ROOT, OUT, read, save, lines, digest
from .native_data import aerosol_valid

DEST = ROOT/'observation_model/outputs/encoder_v2/G1'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def xlsx_sheets(path):
    # Read cached numeric cells with stdlib, retaining the official wavelength grid.
    with zipfile.ZipFile(path) as archive:
        strings = [''.join(n.itertext()) for n in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        relationships = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets = {r.attrib['Id']:r.attrib['Target'] for r in relationships}
        for sheet in workbook.find('m:sheets', NS):
            if not sheet.attrib['name'].startswith('Spectral Responses ('): continue
            rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target = targets[rid]
            target = target.lstrip('/') if target.startswith('/') else 'xl/'+target
            rows = []
            for row in ET.fromstring(archive.read(target)).findall('m:sheetData/m:row', NS):
                values = {}
                for cell in row:
                    raw = cell.findtext('m:v', '', NS)
                    values[re.sub(r'\d', '', cell.attrib['r'])] = strings[int(raw)] if cell.attrib.get('t')=='s' else raw
                rows.append(values)
            yield sheet.attrib['name'], rows


def normalize_curve(wavelength, response):
    wavelength, response = np.asarray(wavelength, float), np.asarray(response, float)
    order = np.argsort(wavelength); wavelength, response = wavelength[order], response[order]
    assert np.isfinite(wavelength).all() and np.isfinite(response).all()
    assert (np.diff(wavelength)>0).all() and (response>=0).all()
    area = float(np.trapezoid(response, wavelength)); assert area>0
    normalized = response/area
    center = float(np.trapezoid(wavelength*normalized, wavelength))
    half = response.max()/2
    indices = np.flatnonzero(response>=half); first,last = indices[0],indices[-1]
    left = wavelength[first] if first==0 else float(np.interp(half,response[first-1:first+1],wavelength[first-1:first+1]))
    right = wavelength[last] if last==len(response)-1 else float(np.interp(half,response[last:last+2][::-1],wavelength[last:last+2][::-1]))
    assert abs(float(np.trapezoid(normalized,wavelength))-1)<1e-10 and right>left
    return wavelength, normalized, center, right-left, area


def build_srf():
    source = ROOT/'data/physics/srf'
    acquisition = read(source/'acquisition_manifest.json'); curves = []
    def record(resource,instrument,band,wavelength,response,member):
        w,r,c,f,a = normalize_curve(wavelength,response)
        key = instrument+'_'+band+'_v'+resource['version'].split()[0]
        path = DEST/'srf'/f'{key}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,spectral_wavelength_grid_nm=w,normalized_response=r)
        curves.append(dict(id=key,instrument=instrument,band_key=band,version=resource['version'],
            source_url=resource['url'],file_hash=resource['sha256'],source_member=member,
            path=str(path),sha256=digest(path),source_type=resource['source_type'],
            wavelength_unit='nm',response_unit='nm^-1',integral=1.,raw_integral_nm=a,
            center_nm=c,fwhm_nm=f,points=len(w),spatial_resampling=False,spectral_resampling=False,
            validity_notes='Instrument-specific band average. Historical characterization; does not reprocess image reflectance.'))
    for resource in acquisition['resources']:
        path = source/resource['file']; assert digest(path)==resource['sha256']
        if path.suffix=='.xlsx':
            for title,rows in xlsx_sheets(path):
                instrument = re.search(r'\((S2[ABC])\)',title).group(1)
                for column,header in rows[0].items():
                    match = re.fullmatch(instrument+r'_SR_AV_B(\d+A?)',header)
                    if not match: continue
                    name = match.group(1); band = 'B'+(name.zfill(2) if name.isdigit() else name)
                    values = [(float(row['A']),float(row[column])) for row in rows[1:] if row.get('A') and row.get(column)]
                    w,r = zip(*values); record(resource,instrument,band,w,r,title+':'+header)
        else:
            instrument = 'LC08' if 'landsat8' in path.name else 'LC09'
            with tarfile.open(path) as archive:
                for member in archive.getmembers():
                    if not member.isfile(): continue
                    channel = int(re.search(r'ch(\d+)\.txt$',member.name).group(1))
                    text = archive.extractfile(member).read().decode('utf-8')
                    data = np.loadtxt(io.StringIO('\n'.join(text.splitlines()[4:])))
                    assert len(data)==int(text.splitlines()[2]) and (data[:,0]>0).all()
                    # R is a dimensionless response evaluated at transformed coordinates,
                    # not a probability density in wavenumber. Normalize after conversion.
                    record(resource,instrument,'SR_B'+str(channel),1e7/data[:,0],data[:,1],member.name)
    assert len(curves)==96 and len({c['id'] for c in curves})==96
    save(DEST/'srf_manifest.json',dict(status='PASS',curves=curves,gaussian_approximations=0,
        selection_policy='S2 V4 before PB05.13; V5 at PB05.13+. Missing PB: acquisition before 2026-08-20 uses V4, otherwise pending. L8 and L9 use separate instrument archives.',
        normalization='Integral over wavelength_nm is one; no spectral resampling; no Jacobian applied to dimensionless response values.'))
    return curves


def known_checks():
    dn = np.array([1,10000,30000,50000],dtype=np.uint16)
    actual = dn.astype('float32')*.0000275-.2
    assert np.allclose(actual,[-.1999725,.075,.625,1.175],atol=2e-7)
    qa = np.array([0,1,2,32,192,194,226],dtype='uint16')
    assert np.array_equal(aerosol_valid(qa),[True,False,True,True,False,True,False])
    t=Affine(30,0,602415,0,-30,6972435)
    xy=t*(.5,.5); assert xy==(602430.,6972420.) and tuple(~t*xy)==(.5,.5)
    w,r,center,width,_=normalize_curve([3,1,2],[0,0,1])
    assert np.array_equal(w,[1,2,3]) and center==2 and width==1
    return dict(known_DN_float32=True,QA_aerosol_bits=True,pixel_center_orientation=True,
        reversed_spectral_grid_normalization=True)


def main():
    started=time.perf_counter(); DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'data_contract_tests.json').exists(): raise FileExistsError('Preserve previous G1 acceptance')
    save(DEST/'approval_state.json',dict(stage='G1',authorized_by='User: existing-local quality check, then G1',
        optimizer_updates=0,old_training_revival=False,G2_authorized=False,G4_authorized=False,
        downloads='Existing queue retained; G1 itself is local only',timestamp=datetime.now(timezone.utc).isoformat()))
    checks=known_checks(); curves=build_srf(); lookup={(c['instrument'],c['band_key'],c['version']):c for c in curves}
    release=read(OUT/'release.json'); rows=lines(OUT/'members.jsonl'); assert digest(OUT/'members.jsonl')==release['manifest_sha256']
    tasks={t['id']:t for t in lines(OUT/'tasks.jsonl')}; contracts=[]; failures=[]; coverage=Counter(); ranges={}
    train_groups={r['release_isolation_group'] for r in rows if r['role']=='TRAIN'}
    val_groups={r['release_isolation_group'] for r in rows if r['role']=='VAL'}
    assert not train_groups&val_groups
    for i,row in enumerate(rows):
        if row['role'] not in ['TRAIN','VAL']:continue
        try:
            assert digest(Path(row['cache']))==row['cache_sha256'],'prepared cache hash changed'
            origin=tasks.get(row['id'],{}); props=origin.get('original') or origin.get('record',{})
            props=props.get('properties',{})
            baseline=props.get('s2:processing_baseline')
            instrument=row['instrument']
            version=('5.0' if float(baseline)>=5.13 else '4.0') if baseline else ('4.0' if row['date'][:10]<'2026-08-20' else None)
            mappings=[];pixel_count=0
            with np.load(row['cache'],allow_pickle=False) as arrays:
                for b in row['bands']:
                    key=b['band_key'];x,q=arrays[key],arrays[key+'__valid']
                    assert x.dtype==np.float32 and q.dtype==bool and x.shape==q.shape==tuple(b['shape'])
                    assert float(q.mean())>=.5 and np.isfinite(x[q]).all()
                    t=Affine(*b['transform']);crs=CRS.from_user_input(b['crs'])
                    assert crs.is_projected and crs.linear_units=='metre' and t.a>0 and t.e<0 and t.b==t.d==0
                    assert np.allclose([t.a,-t.e],b['native_pixel_support_m'])
                    valid=x[q];pixel_count+=len(valid)
                    rk=instrument+':'+key;lo,hi=float(valid.min()),float(valid.max())
                    old=ranges.setdefault(rk,[lo,hi]);old[0]=min(old[0],lo);old[1]=max(old[1],hi)
                    if instrument.startswith('LC'):
                        candidates=[c for c in curves if c['instrument']==instrument and c['band_key']==key]
                        assert len(candidates)==1;curve=candidates[0]
                    else:curve=lookup[(instrument,key,version)]
                    mappings.append(dict(band_key=key,spectral_response_ref=curve['id'],center_nm=curve['center_nm'],fwhm_nm=curve['fwhm_nm'],
                        grid=dict(crs=b['crs'],affine_transform=b['transform'],shape=b['shape'],native_pixel_support_m=b['native_pixel_support_m'])))
                    coverage[f'{row["role"]}/{instrument}/{key}/{row["date"][:4]}']+=1
            contracts.append(dict(id=row['id'],role=row['role'],product_id=row['product'],instrument_id=instrument,
                timestamp=row['date'],processing_baseline=baseline,baseline_source='source_product_metadata' if baseline else 'not_recorded; explicit acquisition-date SRF policy',
                processing_domain=row['processing_domain'],units='surface_reflectance',values_already_scaled=True,
                cache=row['cache'],cache_sha256=row['cache_sha256'],bands=mappings,valid_pixels=pixel_count,
                release_isolation_group=row['release_isolation_group'],model_category_ids_permitted=False))
        except Exception as exc:failures.append(dict(id=row['id'],error=str(exc)))
        if i%1000==0:print('G1_LOCAL_CHECK',i,len(rows),flush=True)
    path=DEST/'observation_contracts.jsonl';path.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in contracts),encoding='utf-8')
    save(DEST/'coverage.json',dict(by_role_instrument_band_year=dict(coverage),reflectance_ranges=ranges,
        members=dict(Counter(c['role']+'/'+c['instrument_id'] for c in contracts)),
        date_range=[min(c['timestamp'] for c in contracts),max(c['timestamp'] for c in contracts)]))
    save(DEST/'pending.json',dict(failed_members=failures,calibration_pending=read(OUT/'calibration_pending.json'),
        download_tasks=release['pending_tasks'],queue_preserved=True))
    evidence=read(OUT/'s2_calibrated_v2/summary.json')
    assert all(c['rmse_after']<1e-6 for c in evidence['controls'])
    checks.update(all_materialized_members_checked=len(contracts)+len(failures),accepted=len(contracts),failed=len(failures),
        TRAIN_VAL_isolation=True,all_valid_values_finite=not failures,native_grid_and_masks=not failures,
        instrument_specific_SRF_coverage=not failures,S2_known_controls=len(evidence['controls']),
        S2_ambiguous_resolved=evidence['counts']['resolved'],data_manifest_sha256=release['manifest_sha256'],
        observation_contracts_sha256=digest(path),srf_manifest_sha256=digest(DEST/'srf_manifest.json'),
        optimizer_updates=0,download_bytes=0,wall_seconds=time.perf_counter()-started,
        status='PASS' if not failures else 'FAIL_STOP_G1')
    save(DEST/'data_contract_tests.json',checks)
    save(DEST/'source_manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'observation_model/expanded_dataset.py',ROOT/'observation_model/finalize_expanded_data.py']})
    text=['# G1：现有数据质量与SRF契约','',
        f'验收状态：{checks["status"]}。逐份检查{len(contracts)+len(failures)}份冻结观测，合格{len(contracts)}份，失败{len(failures)}份；本阶段0次优化更新、0新增网络读取。下载队列保留，不作为本轮前置。','',
        '| 角色 | S2A | S2B | S2C | Landsat8 | Landsat9 |','|---|---:|---:|---:|---:|---:|']
    counts=Counter(c['role']+'/'+c['instrument_id'] for c in contracts)
    for role in ['TRAIN','VAL']:text.append('| '+('训练' if role=='TRAIN' else '验证')+' | '+' | '.join(str(counts[role+'/'+s]) for s in ['S2A','S2B','S2C','LC08','LC09'])+' |')
    text+=['','![冻结数据覆盖](../data_expansion/data_coverage.png)','',
        '左图按传感器显示数据角色，右图显示真实波段数。观测是带实际采集日期的空间窗口，不等于独立整景或严格跨卫星配对。输入保留10/20/30米真实支持，不把插值通道补成原生值。','',
        '已核对每份缓存哈希、有效像元有限性、有效比例、波段维度、米制CRS与像元方向/大小；定标和QA位使用已知数值检查，S2偏移修复另有同次采集参考。采用训练集统计，不读取测试目标，不按标签或模型成绩选成员。实际原生支持及采集合并后的训练/验证组件互不相交。','',
        'SRF使用本地四份已下载资源，ESA V4/V5各39条，L8/L9各9条，合计96条。保留原谱采样，在波长nm上积分归一化；Landsat波数转换为波长后排序，响应值保持维度无关的响应值，再归一化，不错误乘密度Jacobian。记录中心和半高全宽，不用Gaussian冒充实测曲线。','',
        'S2历史影像默认V4，明确PB05.13及以上使用V5；缺baseline且早于2026-08-20的记录明确使用历史V4政策，不伪造baseline。不同仪器使用各自曲线。SRF版本映射是光谱描述选择，不表示重新处理了旧影像。归一化曲线不做额外重采样，G2若需统一谱基仍须另验积分/中心变化。','',
        '![独立定标控制](../data_expansion/s2_calibration_controls.png)','',
        '三个柱组分别是S2A/B/C各1024个同次采集红波段控制，纵轴为对数RMSE；应看修复前0.1系统偏差的消除。这不能恢复上游已截断的负值，也不用于解释不同历史数据上的模型失败。','',
        '下载未完成成员、10份未确认定标的S2仍在pending，不降低质量要求；部分S2只保留四条真实10米通道，不能因此声称SWIR覆盖与四波段样本数一样大。新增Landsat位置关联不等于已通过时间差≤2天的配对监督。HLS不是本次实际准入域，不声称其QA已验收。','',
        '产物：srf_manifest.json、srf/*.npz、observation_contracts.jsonl、data_contract_tests.json、coverage.json、pending.json与source_manifest.json。可复现命令：在项目根运行 `../landslide-seg/.venv-use-independent/Scripts/python.exe -X utf8 -m observation_model.g1_srf_data`。','',
        '本次授权执行到G1；未实现G2新模型、未进入G3课程或G4诊断训练。旧训练、自动续行、checkpoint和原始数据保留。共享阶段台账仍由其他聊天认领，本任务不越过文件锁改写。']
    (DEST/'compatibility.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False),flush=True)
    if failures:raise SystemExit(2)


if __name__=='__main__':main()
