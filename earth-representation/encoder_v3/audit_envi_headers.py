"""Metadata-only ENVI discovery; spectral centers are never converted into SRFs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time
from .data import sha256


def parse_header(data):
    encoding=None
    for candidate in ('utf-8-sig','gb18030','latin1'):
        try:text=data.decode(candidate);encoding=candidate;break
        except UnicodeDecodeError:continue
    if text.lstrip().splitlines()[0].strip().upper()!='ENVI':return dict(status='NOT_ENVI_HEADER',encoding=encoding)
    matches=list(re.finditer(r'(?m)^\s*([A-Za-z][^=\r\n]*?)\s*=',text))
    fields={m.group(1).strip().lower():text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip().strip('{}').strip() for i,m in enumerate(matches)}
    waves=[float(v) for v in fields.get('wavelength','').split(',') if v.strip()]
    bands=int(fields['bands']) if 'bands' in fields else None
    return dict(status='ENVI_METADATA_ONLY',encoding=encoding,bands=bands,samples=int(fields['samples']) if 'samples' in fields else None,lines=int(fields['lines']) if 'lines' in fields else None,sensor=fields.get('sensor type'),data_type=fields.get('data type'),interleave=fields.get('interleave'),reflectance_scale_factor=fields.get('reflectance scale factor'),plot_titles=fields.get('z plot titles'),wavelength_units=fields.get('wavelength units'),wavelength_count=len(waves),wavelength_count_matches_bands=bool(waves) and len(waves)==bands,wavelengths_finite=all(math.isfinite(v) for v in waves),wavelengths_increasing=bool(waves) and all(a<b for a,b in zip(waves,waves[1:])),wavelength_range=[min(waves),max(waves)] if waves and all(math.isfinite(v) for v in waves) else None,map_info=fields.get('map info'),coordinate_system_string=fields.get('coordinate system string'),data_ignore_value=fields.get('data ignore value'),has_bbl='bbl' in fields,has_fwhm='fwhm' in fields,header_keys=sorted(fields))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--filename-list',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    started=time.perf_counter();identity=sha256(args.filename_list)
    paths=sorted({Path(x.decode('utf-8')) for x in args.filename_list.read_bytes().split(b'\0') if x and x.lower().endswith(b'.hdr')})
    rows=[];counts=Counter()
    for path in paths:
        row=dict(path=str(path),v3_admitted=False)
        try:
            before=path.stat()
            if before.st_size>2*1024**2:row.update(status='UNREAD_OVERSIZED_HEADER',bytes=before.st_size)
            else:
                data=path.read_bytes();row.update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data));row.update(parse_header(data))
                if (path.stat().st_size,path.stat().st_mtime_ns)!=(before.st_size,before.st_mtime_ns):raise ValueError('Header changed during read')
        except (OSError,ValueError,IndexError) as exc:row.update(status='HEADER_READ_OR_PARSE_ERROR',error=str(exc))
        counts[row['status']]+=1;rows.append(row)
    if sha256(args.filename_list)!=identity:raise ValueError('Filename snapshot changed')
    report=dict(status='ENVI_HEADERS_REVIEWED_NOT_OPTICAL_ADMISSION',filename_list=str(args.filename_list.resolve()),filename_list_sha256=identity,headers=len(rows),counts=dict(counts),rows=rows,seconds=time.perf_counter()-started,imagery_reads=0,admitted_members=0,limitations=['Only headers present in this filename snapshot; snapshot scan gaps remain','Reflectance labels and scale factors require calibration verification','Wavelength centers/FWHM/bbl are not authentic sampled SRF or spatial QA','Georeferencing metadata is not native-grid or registration acceptance','No role, split, training membership or task changed'])
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('status','headers','counts','seconds')}))
