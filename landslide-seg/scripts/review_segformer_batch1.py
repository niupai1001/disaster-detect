"""Read-only checkpoint audit, valid-region metrics, failure galleries and sensitivity.

Run from project root with the existing tm Python environment. No training or
checkpoint writes. All new artifacts go to review_batch1, separate from reports.
Sensitivity uses mean substitution at inference, not a retrained RGB ablation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from landslide_seg.data.segformer_dataset import S2SegDataset, collate, read_manifest, load_stats
from landslide_seg.training.segformer_trainer import load_checkpoint
from landslide_seg.training.segformer_metrics import label_objects, object_table, match_objects, assign_bin, boundary_metrics

BASE = ROOT / 'outputs/segformer/s2_10band_v1'
OUT = BASE / 'review_batch1'
def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def ratio(a,b): return float(a/b) if b else None
def pixels(tp,fp,fn,tn):
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),iou=ratio(tp,tp+fp+fn),dice=ratio(2*tp,2*tp+fp+fn),precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),background_fpr=ratio(fp,fp+tn))

def measure(prob, gt, valid, rec):
    g = gt.astype(bool) & valid
    p = (prob >= .5) & valid
    tp,fp,fn,tn = ((p&g).sum(),(p&~g).sum(),(~p&g).sum(),(~p&~g&valid).sum())
    row = dict(sample_id=rec['sample_id'],event=rec['event_id'],split=rec['split'],**pixels(tp,fp,fn,tn),gt_valid=int(g.sum()),gt_raw=int(gt.sum()),invalid_prediction=int(((prob>=.5)&~valid).sum()))
    # Preserve original GT component identities/areas; only untruncated GT are
    # eligible for primary object metrics. Predictions are masked BEFORE labeling.
    gl,gn = label_objects(gt)
    pl,pn = label_objects(p)
    gos = object_table(gl,gn,valid); pos = object_table(pl,pn,valid)
    eligible = [o for o in gos if not o['truncated']]
    matches = match_objects(gl,eligible,pl,pos,.25)
    bins = {}
    for name in ('tiny','small','medium','large'):
        objs = [o for o in eligible if (1<=o['area']<=4 if name=='tiny' else assign_bin(o['area'])==name)]
        ids = [o['id'] for o in objs]
        gm = np.isin(gl,ids) & valid
        bins[name] = dict(n=len(objs),matched=sum(o['id'] in matches for o in objs),px=int(gm.sum()),hit=int((gm&p).sum()))
    row.update(bins=bins,objects=len(gos),objects_evaluable=len(eligible),matched=len(matches),max_area=max([o['area'] for o in eligible],default=0),boundary=boundary_metrics(p,g,valid,1))
    row.update(days_after_event=rec['days_after_event'],days_delta_vs_post=rec.get('days_delta_vs_post'),temporal_fallback=rec.get('temporal_fallback'))
    return row

def aggregate(rows):
    res=pixels(*[sum(r[k] for r in rows) for k in ('tp','fp','fn','tn')])
    res.update(n=len(rows),complete_misses=sum(r['tp']==0 and r['gt_valid']>0 for r in rows),gt_empty_on_valid=sum(r['gt_valid']==0 for r in rows),invalid_prediction=sum(r['invalid_prediction'] for r in rows))
    res['bins']={}
    for b in ('tiny','small','medium','large'):
        a={k:sum(r['bins'][b][k] for r in rows) for k in ('n','matched','px','hit')}
        a.update(object_recall=ratio(a['matched'],a['n']),pixel_recall=ratio(a['hit'],a['px']))
        res['bins'][b]=a
    res['object_recall_evaluable']=ratio(sum(r['matched'] for r in rows),sum(r['objects_evaluable'] for r in rows))
    bd={k:sum(r['boundary'][k] for r in rows) for k in ('gt_px','pred_px','prec_tp','rec_tp')}
    bp=ratio(bd['prec_tp'],bd['pred_px']) or 0; br=ratio(bd['rec_tp'],bd['gt_px']) or 0
    res['boundary_f1']=ratio(2*bp*br,bp+br)
    return res

def review_weights():
    from safetensors.torch import load_file
    pretrained=load_file(str(ROOT/'weights/segformer/segformer-b0-finetuned-ade-512-512/model.safetensors'))
    rows=[]
    for exp in ('A0','A1','B0','B1'):
        for seed in (1234,2026,3407):
            d=BASE/exp/str(seed); ck=torch.load(d/'best.pt',map_location='cpu',weights_only=False)
            state=ck['model']; hist=[json.loads(l) for l in (d/'metrics.jsonl').read_text().splitlines()]
            changed={}
            for category,prefix in [('encoder','model.segformer.'),('decoder','model.decode_head.')]:
                pairs=[(v,pretrained[k[6:]]) for k,v in state.items() if k.startswith(prefix) and k[6:] in pretrained and v.shape==pretrained[k[6:]].shape and v.is_floating_point()]
                changed[category]={'comparable_tensors':len(pairs),'changed_tensors':sum(not torch.equal(a,b) for a,b in pairs),'max_abs_change':max(float((a-b).abs().max()) for a,b in pairs)}
            optim=ck['optimizer']; steps=[float(v['step']) for v in optim['state'].values() if 'step' in v]
            row=dict(exp=exp,seed=seed,epoch=ck['epoch'],params=ck['params'],load_info=ck['load_info'],changed=changed,optimizer_tensors=len(steps),optimizer_steps_min=min(steps),optimizer_steps_max=max(steps),optimizer_initial_lrs=[g.get('initial_lr') for g in optim['param_groups']],best_val_iou=max(h['val_iou'] for h in hist),last_val_iou=hist[-1]['val_iou'],train_loss_at_best=hist[ck['epoch']]['train_loss'],train_loss_final=hist[-1]['train_loss'])
            if exp.startswith('B'):
                w=state['projection.proj.weight'][:,:,0,0]
                row['projection_weight']=w.tolist()
                row['seven_nonRGB_max_abs']=float(w[:,3:].abs().max())
                row['seven_nonRGB_norm']=float(w[:,3:].norm())
            else:
                w=state['model.segformer.encoder.patch_embeddings.0.proj.weight']
                row['seven_nonRGB_max_abs']=float(w[:,3:].abs().max())
                row['seven_nonRGB_norm']=float(w[:,3:].norm())
            if exp.endswith('1'): row['fusion_out_norm']=float(state['fusion.out.weight'].norm())
            rows.append(row)
    dump('checkpoint_audit.json',rows)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for exp in ('A0','A1','B0','B1'):
        curves=[[json.loads(l) for l in (BASE/exp/str(s)/'metrics.jsonl').read_text().splitlines()] for s in (1234,2026,3407)]
        for ax,key in zip(axs,('train_loss','val_iou')):
            y=np.array([[r[key] for r in c] for c in curves]); ax.plot(y.mean(0),label=exp); ax.fill_between(np.arange(y.shape[1]),y.mean(0)-y.std(0),y.mean(0)+y.std(0),alpha=.15);ax.set(xlabel='epoch (zero-based)',ylabel=key)
    axs[0].legend();axs[1].legend();fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=150);plt.close(fig)

def stretch(x,valid=None,limits=None):
    h=x.transpose(1,2,0).astype(float)
    z=h[valid] if valid is not None and valid.any() else h.reshape(-1,3)
    lo,hi=np.percentile(z,[2,98],axis=0) if limits is None else limits
    return np.clip((h-lo)/np.maximum(hi-lo,1),0,1)

def select_cases(rows, probs, ds):
    chosen=[]; coverage={}
    for split in ('train','val','test'):
        sub=[(i,r) for i,r in enumerate(rows) if r['split']==split]
        used=set()
        def add(category,cands,key,reverse=True):
            ranked=sorted(cands,key=lambda x:key(x[1]),reverse=reverse)
            pick=next((x for x in ranked if x[0] not in used),ranked[0] if ranked else None)
            coverage[split+'/'+category]='absent' if pick is None else pick[1]['sample_id']
            if pick is not None:
                used.add(pick[0]);chosen.append(dict(index=pick[0],category=category,**pick[1]))
        add('complete_miss',[x for x in sub if x[1]['tp']==0 and x[1]['gt_valid']>=8],lambda r:r['gt_valid'])
        add('partial_detection',[x for x in sub if .15<(x[1]['recall'] or 0)<.65 and x[1]['tp']>=10],lambda r:r['gt_valid'])
        add('false_positive',[x for x in sub if x[1]['fp']>20],lambda r:r['fp'])
        # Shift is ONLY a candidate ranking heuristic, not proof of registration error.
        candidates=sorted([x for x in sub if x[1]['tp']>10 and .05<(x[1]['iou'] or 0)<.65],key=lambda x:x[1]['tp'],reverse=True)[:100]
        shifted=[]
        for i,r in candidates:
            g=ds._mask[i].astype(bool);v=ds._valid[i].copy();v[:3]=False;v[-3:]=False;v[:,:3]=False;v[:,-3:]=False
            p=probs[i]>=.5; base=ratio((p&g&v).sum(),((p|g)&v).sum()) or 0
            best=(base,0,0)
            for dy in range(-3,4):
                for dx in range(-3,4):
                    vv=v&np.roll(ds._valid[i],(dy,dx),(0,1));pp=np.roll(p,(dy,dx),(0,1))
                    shifted_iou=ratio((pp&g&vv).sum(),((pp|g)&vv).sum()) or 0
                    unshifted_iou=ratio((p&g&vv).sum(),((p|g)&vv).sum()) or 0
                    if shifted_iou-unshifted_iou>best[0]-base:best=(base+shifted_iou-unshifted_iou,dy,dx)
            rr=dict(r,shift_gain=best[0]-base,shift_dy=best[1],shift_dx=best[2]);shifted.append((i,rr))
        add('boundary_offset_candidate',shifted,lambda r:r['shift_gain'])
        add('temporal_candidate',[x for x in sub if (x[1]['days_delta_vs_post'] or 0)!=0],lambda r:abs(r['days_delta_vs_post']))
        for b in ('tiny','small','medium','large'):
            add(b+'_miss',[x for x in sub if x[1]['bins'][b]['n']>x[1]['bins'][b]['matched']],lambda r:r['bins'][b]['n']-r['bins'][b]['matched'])
        add('successful_control',[x for x in sub if x[1]['gt_valid']>=10],lambda r:r['iou'] or 0)
        if split=='train':
            represented={r['event'] for r in chosen if r['split']=='train'}
            for ev in sorted({r['event'] for _,r in sub}-represented):
                add('event_'+ev,[x for x in sub if x[1]['event']==ev],lambda r:r['fn'])
    dump('case_selection.json',chosen);dump('coverage.json',coverage)
    return chosen

def galleries(cases, records, ds, probs_all):
    pages=[]
    for split in ('train','val','test'):
        sub=[r for r in cases if r['split']==split]
        for page,start in enumerate(range(0,len(sub),4),1):
            chunk=sub[start:start+4];fig,axs=plt.subplots(len(chunk),7,figsize=(17,3*len(chunk)),squeeze=False)
            for axes,r in zip(axs,chunk):
                i=r['index'];g=ds._mask[i].astype(bool);v=ds._valid[i];raw=ds._x[i]
                rgb=stretch(raw[[2,1,0]],v);false=stretch(raw[[6,2,1]],v)
                mask=np.stack([g*.95,g*.95,g*.95],-1);mask[~v]=[.3,.3,.3]
                panels=[rgb,false,mask]
                for exp in ('A0','A1','B0','B1'):
                    p=probs_all[exp][i]>=.5;overlay=rgb*.4
                    overlay[p&g&v]=[0,1,0];overlay[p&~g&v]=[1,0,0];overlay[~p&g&v]=[0,.5,1];overlay[~v]=[.45,.45,.45]
                    panels.append(overlay)
                for ax,p,title in zip(axes,panels,('RGB','NIR-R-G','GT / invalid gray','A0','A1','B0','B1')):
                    ax.imshow(p,interpolation='nearest');ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
                axes[0].set_ylabel(f"{r['sample_id']}\n{r['category']}\nB1 IoU={r['iou'] or 0:.3f} R={r['recall'] or 0:.3f}\nGTvalid={r['gt_valid']} TP/FP/FN={r['tp']}/{r['fp']}/{r['fn']}",fontsize=8)
            fig.suptitle(f'{split} seed=1234 best checkpoints | green=TP red=FP blue=FN gray=invalid | threshold=.5',fontsize=11)
            fig.tight_layout(rect=[0,0,1,.97]);name=f'{split}_cases_{page}.png';fig.savefig(OUT/name,dpi=120);plt.close(fig);pages.append(name)
    dump('gallery_files.json',pages)
    # Same stretch across the post-reference and selected dates; same inventory
    # polygon is shown in yellow, NOT claimed to be an observation-date annotation.
    import h5py
    from datetime import date,timedelta
    for r in [r for r in cases if r['category']=='temporal_candidate']:
        rec=records[r['index']]
        with h5py.File(rec['source_uri'],'r') as f:
            indexes=list(dict.fromkeys([int(rec['post_index']),int(rec['time_index'])]))
            images=[np.stack([f[b][t] for b in ('B04','B03','B02')]) for t in indexes]
            units=f['time'].attrs['units'];units=units.decode() if isinstance(units,bytes) else str(units)
            ref=date.fromisoformat(units.split('since')[-1].strip()[:10])
            dates=[str(ref+timedelta(days=int(f['time'][t]))) for t in indexes]
            scl=[np.asarray(f['SCL'][t]) for t in indexes] if 'SCL' in f else None
        lim=np.percentile(np.concatenate([x.transpose(1,2,0).reshape(-1,3) for x in images]),[2,98],axis=0)
        fig,axs=plt.subplots(2,len(indexes),figsize=(5*len(indexes),8),squeeze=False)
        for j,(im,dt,t) in enumerate(zip(images,dates,indexes)):
            axs[0,j].imshow(stretch(im,limits=lim));axs[0,j].contour(ds._mask[r['index']],levels=[.5],colors='yellow',linewidths=.6);axs[0,j].set_title(f'{dt} index={t} | '+('chosen' if t==rec['time_index'] else 'post reference'))
            if scl is not None:axs[1,j].imshow(scl[j],vmin=0,vmax=11,cmap='tab20',interpolation='nearest');axs[1,j].set_title('SCL (categorical)')
        for ax in axs.ravel():ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f"{rec['sample_id']} | inventory outline, temporal mismatch UNCONFIRMED");fig.tight_layout();fig.savefig(OUT/f"temporal_{r['split']}.png",dpi=130);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:1');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    checkpoints=list(BASE.glob('*/*/best.pt'))+list(BASE.glob('*/*/last.pt'))
    before={str(p.relative_to(ROOT)):sha(p) for p in checkpoints};dump('checkpoint_sha256_before.json',before)
    review_weights();print('Checkpoint audit done',flush=True)
    records=read_manifest(ROOT/'manifests/s2_10band_v1.jsonl');stats=load_stats(ROOT/'data/s2_10band_v1/stats.json')
    ds=S2SegDataset(records,stats,preload=True);loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0,collate_fn=collate)
    probs_all={};allrows={};summaries={};sensitivity={}
    for exp in ('A0','A1','B0','B1'):
        model,ck=load_checkpoint(BASE/exp/'1234/best.pt',args.device)
        cache=OUT/f'{exp}_prob_float32.npy'
        # Always recompute: an interrupted or stale cache is not evidence.
        probs=np.lib.format.open_memmap(cache,mode='w+',dtype='float32',shape=(len(ds),128,128));offset=0
        with torch.inference_mode():
            for batch in loader:
                x=batch['image'].to(args.device);p=model(x)['logits'][:,0].sigmoid().cpu().numpy();probs[offset:offset+len(p)]=p;offset+=len(p)
        probs.flush()
        probs_all[exp]=probs
        rows=[measure(probs[i],ds._mask[i],ds._valid[i],rec) for i,rec in enumerate(records)];allrows[exp]=rows;dump(f'{exp}_per_tile.json',rows)
        summaries[exp]={split:aggregate([r for r in rows if r['split']==split]) for split in ('train','val','test')}
        summaries[exp]['by_event']={ev:aggregate([r for r in rows if r['event']==ev]) for ev in sorted({r['event'] for r in rows})}
        print(exp,{k:round(v['iou'],5) for k,v in summaries[exp].items() if k!='by_event'},flush=True)
        if exp=='B1':
            for mode in ('all_nonRGB_mean','projection_nonRGB_mean','stem_nonRGB_mean','base_logits_only'):
                def hook(module,inputs):
                    x=inputs[0].clone();x[:,3:]=0;return (x,)
                handle=None
                if mode=='projection_nonRGB_mean':handle=model.projection.register_forward_pre_hook(hook)
                if mode=='stem_nonRGB_mean':handle=model.stem.register_forward_pre_hook(hook)
                sums={s:np.zeros(5,dtype=np.float64) for s in ('train','val','test')};offset=0
                with torch.inference_mode():
                    for batch in loader:
                        x=batch['image'].to(args.device)
                        if mode=='all_nonRGB_mean':x[:,3:]=0
                        out=model(x);pp=out['base_logits' if mode=='base_logits_only' else 'logits'][:,0].sigmoid().cpu().numpy()
                        for j,p in enumerate(pp):
                            i=offset+j;v=ds._valid[i];g=ds._mask[i].astype(bool)&v;pred=(p>=.5)&v
                            sums[records[i]['split']]+=np.array([(pred&g).sum(),(pred&~g).sum(),(~pred&g).sum(),(~pred&~g&v).sum(),np.abs(p-probs[i])[v].sum()])
                        offset+=len(pp)
                if handle:handle.remove()
                sensitivity[mode]={s:dict(**pixels(*a[:4]),mean_absolute_probability_change=float(a[4]/sum(ds._valid[i].sum() for i,r in enumerate(records) if r['split']==s))) for s,a in sums.items()}
                print('sensitivity',mode,{s:round(v['iou'],5) for s,v in sensitivity[mode].items()},flush=True)
        del model,ck;torch.cuda.empty_cache()
    dump('metrics_valid_region.json',summaries);dump('sensitivity_B1_seed1234.json',sensitivity)
    cases=select_cases(allrows['B1'],probs_all['B1'],ds);galleries(cases,records,ds,probs_all)
    dump('negative_tiles.json',[dict(records[i],gt_valid=int((ds._mask[i]&ds._valid[i]).sum()),gt_raw=int(ds._mask[i].sum())) for i in range(len(ds)) if (ds._mask[i]&ds._valid[i]).sum()==0])
    after={str(p.relative_to(ROOT)):sha(p) for p in checkpoints};dump('checkpoint_integrity.json',{'unchanged':before==after,'n_checkpoints':len(before),'after':after})
    assert before==after,'Unexpected checkpoint modification'
    print('Done: checkpoints unchanged; artifacts at',OUT,flush=True)

if __name__=='__main__': main()
