"""Frozen core S2 groups and strictly aligned, unresampled native windows."""
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import torch
import torch.nn.functional as F
from encoder_v4.data import OpticalDataset
from s2_native_grid_r1.members import RELEASE, RELEASE_SHA, OUTPUT, sha
from .model import Layout


def write_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    temp.replace(path)


def freeze(root, out):
    source=root/OUTPUT/'train_1000.jsonl'
    raw=source.read_bytes()
    old=json.loads((source.parent/'members_identity.json').read_text())
    if sha(raw)!=old['members_sha256'] or sha((root/RELEASE/'release.json').read_bytes())!=RELEASE_SHA:
        raise ValueError('Frozen pool/release checksum mismatch')
    rows=[json.loads(line) for line in raw.splitlines()]
    if len({r['id'] for r in rows})!=len(rows) or any(r['sensor']!='S2' or r['dataset_role']!='TRAIN' for r in rows):
        raise ValueError('Expected distinct core S2 original TRAIN parents')
    # Join the existing groups, identical acquisitions, and overlapping geographic footprints.
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j): parent[find(j)]=find(i)
    seen={}
    for i,row in enumerate(rows):
        for key in ('isolation_group','canonical_acquisition'):
            value=row.get(key)
            if not value: raise ValueError(f'Missing {key}')
            identity=(key,value)
            if identity in seen:union(i,seen[identity])
            else:seen[identity]=i
        a=row['bbox']
        for j in range(i):
            b=rows[j]['bbox']
            if a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]:union(i,j)
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[find(i)].append(row)
    ordered=sorted(groups.values(),key=lambda g:min(r['id'] for r in g))
    random.Random(17).shuffle(ordered)
    # Closest prefix to 70%; groups remain indivisible.
    counts=[0]
    for g in ordered:counts.append(counts[-1]+len(g))
    cut=min(range(1,len(counts)-1),key=lambda i:abs(counts[i]-.7*len(rows)))
    train=[r for g in ordered[:cut] for r in g];val=[r for g in ordered[cut:] for r in g]
    identity=dict(source=str(source),pool_sha256=sha(raw),release_sha256=RELEASE_SHA,
                  group_rule='union existing isolation_group, canonical_acquisition, overlapping lon/lat bbox; all dates',
                  seed=17,pool_count=len(rows),train_count=len(train),development_val_count=len(val),
                  groups=len(groups),old_val_test_included=False,role='development VAL, not independent TEST')
    out.mkdir(parents=True,exist_ok=True)
    for name,members in [('train',train),('val',val)]:
        payload=''.join(json.dumps(r,sort_keys=True)+'\n' for r in members).encode()
        target=out/f'{name}.jsonl'
        if target.exists() and target.read_bytes()!=payload:raise ValueError('Existing manifest differs')
        target.write_bytes(payload);identity[name+'_sha256']=sha(payload)
    write_json(out/'dataset_identity.json',identity)
    return train,val,identity


def geometry(row):
    bands=row['bands'];first=bands[0]
    left=max(b['transform'][2] for b in bands);top=min(b['transform'][5] for b in bands)
    coarsest=max(bands,key=lambda b:b['transform'][0])['transform']
    left=coarsest[2]+math.ceil((left-coarsest[2])/coarsest[0]-1e-9)*coarsest[0]
    top=coarsest[5]-math.ceil((coarsest[5]-top)/-coarsest[4]-1e-9)*-coarsest[4]
    shapes=[]
    for b in bands:
        a,r,x,s,e,y=b['transform']
        if b['crs']!=first['crs'] or not b['crs'].startswith(('EPSG:326','EPSG:327')) or r or s or a<=0 or e>=0:
            raise ValueError('Expected common north-up UTM grids')
        vals=(120/a,120/-e,(left-x)/a,(y-top)/-e)
        if any(abs(v-round(v))>1e-7 for v in vals):raise ValueError('Native grids do not align to 120m patches')
        shapes.append((round(vals[1]),round(vals[0])))
    return Layout(tuple(b['band_key'] for b in bands),tuple(shapes),(4,4)),(left,top)


def patchify(x,shape):
    h,w=shape
    return x.reshape(4,h,4,w).permute(0,2,1,3).reshape(16,h,w)


def unpatchify(x):
    h,w=x.shape[-2:]
    return x.reshape(4,4,h,w).permute(0,2,1,3).reshape(4*h,4*w)


def windows(row):
    layout,(left,top)=geometry(row)
    right=min(b['transform'][2]+b['transform'][0]*b['shape'][1] for b in row['bands'])
    bottom=max(b['transform'][5]+b['transform'][4]*b['shape'][0] for b in row['bands'])
    # Non-overlapping 480m windows; final edges are invalid-padded.
    return layout,[(left+480*c,top-480*r) for r in range(max(1,math.ceil((top-bottom)/480)))
                   for c in range(max(1,math.ceil((right-left)/480)))]


def load_window(dataset,row,origin,layout):
    observation=dataset.load(row,'cpu');values=[];qa=[]
    for band,shape in zip(observation.bands,layout.native_shapes,strict=True):
        a,_,x,_,e,y=band.grid.affine
        c=round((origin[0]-x)/a);r=round((y-origin[1])/-e);h,w=shape
        v=band.values[r:r+4*h,c:c+4*w];q=band.valid[r:r+4*h,c:c+4*w] & torch.isfinite(v)
        padding=(0,4*w-v.shape[1],0,4*h-v.shape[0])
        values.append(patchify(F.pad(torch.where(q,v,0.),padding),shape))
        qa.append(patchify(F.pad(q,padding,value=False),shape))
    return tuple(values),tuple(qa)


def masks(qa, ids, window_ids, exposure, seed=17):
    support=torch.stack([q.flatten(2).any(-1) for q in qa],-1)
    valid=support.any(-1);batch,p,n=support.shape
    a=torch.zeros_like(valid);b=torch.ones(batch,n,dtype=torch.bool,device=valid.device)
    for i,identity in enumerate(ids):
        legal=valid[i].nonzero().flatten().tolist()
        if len(legal)<2:raise ValueError('A needs two legal geographic patches')
        rng=random.Random(f'{seed}:{identity}:{window_ids[i]}:A:{exposure}')
        a[i,rng.sample(legal,max(1,len(legal)//4))]=True
        offset=int(hashlib.sha256(f'{seed}:{identity}:B1'.encode()).hexdigest()[:8],16)%n
        hidden=(offset+exposure)%n;b[i,hidden]=False
        if not support[i,:,hidden].any() or not (support[i,:,b[i]].any(-1)|~valid[i]).all():
            raise ValueError('B1 has empty hidden target or lost visible patch context')
    return (a,torch.ones_like(b)),(valid,b)


def batch(samples,device):
    values=tuple(torch.stack([s[0][j] for s in samples]).to(device) for j in range(len(samples[0][0])))
    qa=tuple(torch.stack([s[1][j] for s in samples]).to(device) for j in range(len(values)))
    return values,qa
