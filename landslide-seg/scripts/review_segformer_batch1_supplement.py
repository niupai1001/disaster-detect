"""Supplement existing batch-one review: dates, metric fixtures, object closeups."""
import json,re
from pathlib import Path
from collections import Counter
import numpy as np
import h5py
import matplotlib.pyplot as plt
from review_segformer_batch1 import ROOT,OUT,BASE,dump,measure,read_manifest,label_objects,object_table,match_objects,assign_bin,stretch
from landslide_seg.training.segformer_metrics import evaluate_tile,summarize_tiles

def fixtures():
    rec=dict(sample_id='synthetic_metric_fixture',event_id='fixture',split='fixture',days_after_event=0)
    g=np.zeros((16,16),bool);g[6:8,6:8]=True;v=np.ones_like(g);p=g.astype(float)
    old=evaluate_tile(p,g,v);new=measure(p,g,v,rec)
    assert old['tiny']['matched']==1 and old['tiny']['matched_eval']==0
    assert new['bins']['tiny']['matched']==1
    # A one-pixel overlap below matching IoU still contributes to pixel recall.
    g[:]=False;g[6:9,6:9]=True;p[:]=0;p[6,6]=1
    old=evaluate_tile(p,g,v);new=measure(p,g,v,rec)
    assert old['bins']['small']['px_hit']==0 and new['bins']['small']['hit']==1
    # An invalid extension changes prediction area in the old object metric.
    g[:]=False;g[6:8,6:8]=True;p[:]=0;p[6:8,6:8]=1;p[6,8:]=1;p[2:14,10:]=1;v[:,10:]=False
    old=evaluate_tile(p,g,v);new=measure(p,g,v,rec)
    assert old['n_matched_evaluable']==0 and new['matched']==1
    out={'passed':3,'tiny_matched_eval_bug':True,'scale_pixel_recall_matching_dependency':True,'invalid_prediction_changes_object_match':True,'note':'Synthetic arrays test metric definitions only; not model performance.'}
    dump('metric_regression_checks.json',out)

def dates(records):
    multi=[r for r in records if len(re.findall(r'\d{4}-\d{2}-\d{2}',r['event_date']))>1]
    early=[r for r in multi if r['date']<max(re.findall(r'\d{4}-\d{2}-\d{2}',r['event_date']))]
    post_before=[r for r in records if (r.get('days_at_post') or 0)<0]
    dump('multidate_audit.json',dict(multi_date_count=len(multi),by_region=dict(Counter(r['event_id'] for r in multi)),chosen_before_latest_count=len(early),chosen_before_latest_by_region=dict(Counter(r['event_id'] for r in early)),chosen_before_latest=early,post_reference_before_first_date_count=len(post_before),post_reference_before_first_date_by_region=dict(Counter(r['event_id'] for r in post_before))))
    return early

def object_crops(records):
    cases=json.loads((OUT/'case_selection.json').read_text());out=[]
    pp={e:np.load(OUT/f'{e}_prob_float32.npy',mmap_mode='r') for e in ('A0','A1','B0','B1')}
    for split in ('train','val','test'):
        selected=[]
        for cat in ('tiny_miss','small_miss','medium_miss','large_miss'):
            case=next((r for r in cases if r['split']==split and r['category']==cat),None)
            if case is None:continue
            i=case['index'];d=np.load(records[i]['cache']);g=d['mask'].astype(bool);v=d['valid'].astype(bool)
            gl,gn=label_objects(g);objs=[o for o in object_table(gl,gn,v) if not o['truncated']]
            pl,pn=label_objects((pp['B1'][i]>=.5)&v);matches=match_objects(gl,objs,pl,object_table(pl,pn,v),.25)
            b=cat.split('_')[0];objs=[o for o in objs if o['id'] not in matches and (1<=o['area']<=4 if b=='tiny' else (5<=o['area']<50 if b=='small' else assign_bin(o['area'])==b))]
            for o in sorted(objs,key=lambda o:o['area'])[:(3 if b=='tiny' else 1)]:selected.append((i,o,cat))
        fig,axs=plt.subplots(len(selected),6,figsize=(12,2.2*len(selected)),squeeze=False)
        for axes,(i,o,cat) in zip(axs,selected):
            d=np.load(records[i]['cache']);g=d['mask'].astype(bool);v=d['valid'].astype(bool);rgb=stretch(d['x10'][[2,1,0]],v)
            y0,x0,y1,x1=o['bbox'];cy=(y0+y1)//2;cx=(x0+x1)//2;half=max(8,(max(y1-y0,x1-x0)+1)//2+4)
            ys=slice(max(0,cy-half),min(128,cy+half));xs=slice(max(0,cx-half),min(128,cx+half))
            panels=[rgb,g]
            for e in pp:
                pr=pp[e][i]>=.5;a=rgb*.3;a[g&pr&v]=[0,1,0];a[~g&pr&v]=[1,0,0];a[g&~pr&v]=[0,.5,1];a[~v]=.45;panels.append(a)
            for ax,a,title in zip(axes,panels,('RGB','GT','A0','A1','B0','B1')):
                ax.imshow(a[ys,xs],interpolation='nearest',cmap='gray');ax.set_title(title,fontsize=8);ax.set_xticks([]);ax.set_yticks([])
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((x0-xs.start-.5,y0-ys.start-.5),x1-x0,y1-y0,fill=False,edgecolor='yellow',linewidth=.7))
            axes[0].set_ylabel(f"{records[i]['sample_id']}\n{cat}: id={o['id']} area={o['area']} px",fontsize=8)
            out.append(dict(sample_id=records[i]['sample_id'],split=split,category=cat,**o))
        fig.suptitle('Missed objects at IoU >= .25 | native pixels magnified, no super-resolution | yellow=object bbox',fontsize=10)
        fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(OUT/f'object_crops_{split}.png',dpi=130);plt.close(fig)
    dump('object_crops.json',out)

def data_provenance(records,early):
    ids=['china_s2_1004','indonesia_s2_157','indonesia_s2_455','china_s2_642','hiroshima_s2_1427','hiroshima_s2_1428','italy_s2_1449']
    out=[]
    for sid in ids:
        r=next(r for r in records if r['sample_id']==sid)
        with h5py.File(r['source_uri'],'r') as f:
            def cv(x):
                if isinstance(x,bytes):return x.decode(errors='replace')
                if isinstance(x,(list,tuple)):return [cv(y) for y in x]
                if hasattr(x,'tolist'):return cv(x.tolist())
                if isinstance(x,(int,float,bool,str)) or x is None:return x
                return str(x)
            out.append(dict(sample_id=sid,manifest=r,attributes={k:cv(v) for k,v in f.attrs.items()},band_attributes={k:cv(v) for k,v in f['B04'].attrs.items()},mask_unique=np.unique(f['MASK'][:]).tolist()))
    dump('source_metadata_examples.json',out)

def compare_legacy():
    current=json.loads((OUT/'metrics_valid_region.json').read_text());rows=[]
    for exp in ('A0','A1','B0','B1'):
        for split in ('val','test'):
            old=json.loads((BASE/exp/'1234/eval'/f'eval_{split}.json').read_text(encoding='utf-8'))['overall'];new=current[exp][split]
            rows.append(dict(exp=exp,split=split,old_iou=old['pixel']['iou'],new_iou=new['iou'],old_obj=old['objects']['object_recall_evaluable'],new_obj=new['object_recall_evaluable'],old_tiny=old['objects']['tiny_1_4px'],new_tiny=new['bins']['tiny']))
    dump('legacy_comparison.json',rows)

if __name__=='__main__':
    fixtures();records=read_manifest(ROOT/'manifests/s2_10band_v1.jsonl');early=dates(records);data_provenance(records,early);object_crops(records);compare_legacy();print('Supplement complete; 3 metric fixtures passed.')
