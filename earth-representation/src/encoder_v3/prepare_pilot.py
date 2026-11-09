"""Write a reviewable candidate study; never changes configuration or starts updates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from .data import NativeOpticalDataset
from .tasks import choose_patch,hidden_band_indices,task_rng


def cohort(dataset):
    chosen=[]
    selections=[('S2',4,2),('S2',10,2),('L8',6,2),('L8',7,2)] if dataset.role=='TRAIN' else [('S2',4,2),('S2',10,2),('L8',7,4)]
    for sensor,bands,count in selections:
        found=[r for r in dataset.rows if r['sensor']==sensor and len(r['bands'])==bands and max(max(b['shape']) for b in r['bands'])<=160]
        if len(found)<count:raise ValueError(f'Insufficient intact engineering observations: {dataset.role}/{sensor}/{bands}')
        chosen.extend(found[:count])
    return chosen


def bounds(row):
    boxes=[]
    for b in row['bands']:
        h,w=b['shape'];a,rotation,left,shear,e,top=b['transform']
        if rotation or shear or a<=0 or e>=0:raise ValueError('Candidate requires audited north-up native grids')
        boxes.append((left,top+e*h,left+a*w,top))
    return min(v[0] for v in boxes),min(v[1] for v in boxes),max(v[2] for v in boxes),max(v[3] for v in boxes)


def rectangle(row,patch,*,seed,update):
    left,bottom,right,top=bounds(row);nx=math.ceil((right-left)/patch);ny=math.ceil((top-bottom)/patch)
    rng=task_rng(seed,update,row['id'],'A_rectangle_candidate')
    ratio=math.exp(rng.uniform(math.log(.5),math.log(2.)))
    width=max(1,min(nx,round(math.sqrt(.5*nx*ny*ratio))))
    height=max(1,min(ny,round(math.sqrt(.5*nx*ny/ratio))))
    x=rng.randint(0,nx-width);y=rng.randint(0,ny-height)
    return [left+x*patch,bottom+y*patch,left+(x+width)*patch,bottom+(y+height)*patch]


def member(row):
    return {key:row[key] for key in ('id','role','instrument','sensor','date','product','cache_sha256','release_isolation_group','bands')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--contracts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    train=NativeOpticalDataset(args.release,args.contracts,role='TRAIN');validation=NativeOpticalDataset(args.release,args.contracts,role='VAL')
    train_rows=cohort(train);val_rows=cohort(validation)
    if {r['release_isolation_group'] for r in train_rows}&{r['release_isolation_group'] for r in val_rows}:raise ValueError('Candidate groups overlap across roles')
    schedule=[]
    for update in range(200):
        patch=choose_patch(0,update,'logical_batch_candidate')
        tasks=[dict(observation=r['id'],A_rectangle=rectangle(r,patch,seed=0,update=update),B1_hidden=[r['bands'][i]['band_key'] for i in sorted(hidden_band_indices(len(r['bands']),seed=0,update=update,observation_id=r['id']))]) for r in train_rows]
        schedule.append(dict(logical_update=update,patch_m=patch,members=[r['id'] for r in train_rows],tasks=tasks))
    panels=[]
    for patch in (60.,120.):
        for row in val_rows:panels.append(dict(observation=row['id'],patch_m=patch,A_rectangle=rectangle(row,patch,seed=1,update=0),B1_single=[row['bands'][0]['band_key']],B1_multi=[row['bands'][0]['band_key'],row['bands'][1]['band_key']]))
    study=dict(status='CANDIDATE_REQUIRES_USER_REVIEW',review_status='candidate',scope='400-update optimizer preflight only; not formal membership',source_identities=train.identities,release=str(args.release.resolve()),contracts=str(args.contracts.resolve()),seed=0,effective_batch=8,optimizers=['AdamW','CAdamW'],per_candidate_limit=200,total_limit=400,hypernetwork=dict(input=34,hidden=64,output=320,activation='GELU',gsd='ln(meters)',review='reports/hypernetwork_preflight_review.json'),execution_candidates=dict(precision='fp32',microbatch_observations=1,local_batch_size=64,chunk_pixels=512,native_mode='grouped',layout_mode='indexed',indexed_decoder=True),training_members=[member(r) for r in train_rows],validation_members=[member(r) for r in val_rows],rectangle_proposal=dict(count=1,target_patch_fraction=.5,aspect_ratio='log-uniform [0.5,2.0]',position='uniform feasible patch-aligned rectangle',size_rounding='nearest integer patch count, clipped to observation patch layout'),schedule=schedule,validation_panels=panels,B2=None,limitations=['Candidate only: no updates authorized by this file','Intact smaller observations selected for initial fitting; not representative of maximum shape or all sources','Engineering TRAIN/VAL roles retained; V3 full-machine admission incomplete','B2 and unseen-band-combination formal panels unresolved','Mask count/aspect ratio and exact members need user confirmation','Optimizer precision, resource and validation-uncertainty acceptance remain unresolved'])
    canonical=json.dumps(study,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode();study['candidate_sha256']=hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(study,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=study['status'],candidate_sha256=study['candidate_sha256'],training_members=len(train_rows),validation_members=len(val_rows),schedule_updates=len(schedule),optimizer_updates=0),indent=2))
