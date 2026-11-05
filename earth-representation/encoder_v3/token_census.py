"""QA-only full-visible token census; no window or member policy changes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .data import checked_file,sha256
from .geometry_views import northing_shift
from .pair_catalog import native_window


def token_count(mask,affine,patch,origin):
    a,b,left,d,e,top=affine
    if b or d or a<=0 or e>=0:raise ValueError('North-up native grid required')
    h,w=mask.shape
    x=np.floor((left+(np.arange(w)+.5)*a-origin[0])/patch).astype(np.int64)
    y=np.floor((top+(np.arange(h)+.5)*e-origin[1])/patch).astype(np.int64)
    xs=np.r_[0,np.flatnonzero(np.diff(x))+1];ys=np.r_[0,np.flatnonzero(np.diff(y))+1]
    occupied=np.logical_or.reduceat(np.logical_or.reduceat(mask,xs,axis=1),ys,axis=0)
    return int(occupied.sum())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--members',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    before=sha256(args.members);started=time.perf_counter()
    members=[json.loads(line) for line in args.members.read_text(encoding='utf-8-sig').splitlines() if line.strip()];rows=[]
    for row in members:
        if row['role'] not in ('TRAIN','VAL'):continue
        region=native_window(row);left,bottom,_,_=region['region'].bounds;origin=(left,bottom);counts={60:[],120:[]}
        checked_file(row['cache'],row['cache_sha256']);stat=Path(row['cache']).stat()
        with np.load(row['cache'],allow_pickle=False) as arrays:
            for band in row['bands']:
                mask=arrays[band['band_key']+'__valid']
                if mask.dtype!=np.bool_ or mask.shape!=tuple(band['shape']):raise ValueError('QA grid changed')
                affine=list(band['transform']);affine[5]+=northing_shift(band['crs'],region['crs'])
                for patch in counts:counts[patch].append(token_count(mask,affine,patch,origin))
        after=Path(row['cache']).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during QA census')
        rows.append(dict(id=row['id'],role=row['role'],sensor=row['sensor'],cache_sha256=row['cache_sha256'],origin=origin,crs=region['crs'],patches={str(patch):dict(per_band=values,full_visible_tokens=sum(values),max_B1_single_hidden_tokens=sum(values)-min(values),one_dense_FP32_4head_score_tensor_bytes=4*4*sum(values)**2) for patch,values in counts.items()}))
    if sha256(args.members)!=before:raise ValueError('Member inventory changed')
    summaries={}
    for patch in ('60','120'):
        values=np.array([r['patches'][patch]['full_visible_tokens'] for r in rows])
        summaries[patch]=dict(min=int(values.min()),median=float(np.median(values)),p95=float(np.quantile(values,.95)),max=int(values.max()),largest=sorted(rows,key=lambda r:r['patches'][patch]['full_visible_tokens'],reverse=True)[:5])
    report=dict(status='EXACT_QA_FULL_VISIBLE_TOKEN_COUNTS_NOT_RUNTIME_ACCEPTANCE',members_sha256=before,observations=len(rows),summaries=summaries,rows=rows,seconds=time.perf_counter()-started,reflectance_array_reads=0,optimizer_updates=0,limitations=['Origin is lower-left union of common-metric native windows; no formal origin/window policy chosen','Full visibility is conservative for A/B and relevant to a full-input teacher; actual task counts differ','Dense score storage is one hypothetical 4-head FP32 N-by-N tensor, not measured peak memory or a claim about a fused backend','No tokens, windows, bands, tasks or members removed; no GSD binning or resampling'])
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(dict(observations=len(rows),seconds=report['seconds'],summaries={k:{a:b for a,b in v.items() if a!='largest'} for k,v in summaries.items()})))
