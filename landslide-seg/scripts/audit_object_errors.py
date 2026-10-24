"""Audit frozen TRAIN/DEV data and baseline object errors; never train or edit inputs."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from landslide_seg.config import resolve_path, file_sha256
from landslide_seg.data.segformer_dataset import S2SegDataset, _augment, collate, read_manifest
from landslide_seg.data.segformer_dataset import S2_10_BANDS, SCL_VALID
from landslide_seg.training.segformer_trainer import load_checkpoint, _seg_loss
from landslide_seg.training.segformer_metrics import (
    label_objects, object_table, match_objects, evaluate_tile_v2,
    summarize_tiles_v2, summarize_by_event_v2,
)

RUN = ROOT / 'outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234'
OUT = ROOT / 'outputs/data_review'
ASSET = ROOT / 'docs/assets/research/object_data_review'
BINS = [('1_4',1,5),('5_9',5,10),('10_19',10,20),('20_49',20,50),('50_199',50,200),('ge200',200,None)]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def size_bin(area):
    return next(k for k,lo,hi in BINS if area >= lo and (hi is None or area < hi))


def describe_objects(prob, gt, valid):
    """Factual flags; merge/fragment flags can coexist and are not causal diagnoses."""
    gl, gn = label_objects(gt)
    objs = object_table(gl, gn, valid)
    pred = (prob >= .5) & valid
    pl, pn = label_objects(pred)
    po = object_table(pl, pn)
    matched = match_objects(gl, [o for o in objs if not o['truncated']], pl, po, .25)
    pa = np.bincount(pl.ravel())
    touches = {p['id']: len(set(np.unique(gl[pl == p['id']]))-{0}) for p in po}
    rows = []
    for o in objs:
        region = (gl == o['id']) & valid
        values = prob[region]
        ids, hits = np.unique(pl[region & pred], return_counts=True)
        overlaps = [(int(pid), int(n), n/(o['area']+pa[pid]-n)) for pid,n in zip(ids,hits)]
        merged = any(touches[pid] > 1 for pid,_,_ in overlaps)
        hit = int((region & pred).sum())
        maxp = float(values.max()) if values.size else None
        is_matched = o['id'] in matched
        if is_matched:
            category = 'matched'
        elif not values.size:
            category = 'no_valid_pixels'
        elif hit == 0:
            category = 'no_hit_below_0.1' if maxp < .1 else 'no_hit_0.1_to_0.5'
        else:
            category = 'overlap_but_unmatched'
        border = (gl == o['id']) & ~ndimage.binary_erosion(gl == o['id'], structure=np.ones((3,3))) & valid
        inside = region & ~border
        ring = ndimage.binary_dilation(gl == o['id'], structure=np.ones((3,3)), iterations=2) & (gl != o['id']) & valid
        rows.append({**o, 'bin':size_bin(o['area']), 'valid_px':int(region.sum()),
            'hit_px':hit, 'max_prob':maxp, 'mean_prob':float(values.mean()) if values.size else None,
            'matched':is_matched, 'category':category,
            'max_iou':float(max((v[2] for v in overlaps), default=0)),
            'overlap_components':len(overlaps), 'merge_flag':merged,
            'fragment_flag':len(overlaps)>1, 'nearby_prediction_2px':bool((ring & pred).any()),
            'boundary_px':int(border.sum()), 'boundary_fn':int((border & ~pred).sum()),
            'interior_px':int(inside.sum()), 'interior_fn':int((inside & ~pred).sum())})
    return rows


class FixedAugment:
    def __init__(self, horizontal, vertical, rotation):
        self.flips = iter([0. if horizontal else 1., 0. if vertical else 1.])
        self.rotation = rotation

    def random(self):
        return next(self.flips)

    def integers(self, *args):
        return self.rotation


def pipeline_audit(records, cfg, checkpoint):
    import h5py
    exposure = json.loads((RUN/'sample_exposures.json').read_text())
    state = checkpoint.get('stream_state', {})
    best_exposure = state.get('exposures', state.get('sample_exposures'))
    if isinstance(best_exposure, list):
        train_ids = [r['sample_id'] for r in records if r['split']=='train']
        assert len(best_exposure)==len(train_ids)
        assert train_ids==checkpoint['meta']['train_samples']
        assert sum(best_exposure)==checkpoint['global_step']*cfg['segformer']['train']['batch_size']
        best_exposure = dict(zip(train_ids,best_exposure))
    grouped = defaultdict(Counter)
    samples = []
    hashes = []
    source_checks = Counter()
    for idx, rec in enumerate(records):
        with np.load(rec['cache']) as z:
            gt = z['mask'].astype(bool); valid = z['valid'].astype(bool)
            cached_image=z['x10']
        # Verify frozen selected-index arrays only; do not assess dates or annotation timing.
        with h5py.File(rec['source_uri'],'r') as f:
            raw_mask=f['MASK'][rec['time_index']]
            raw_image=np.stack([f[b][rec['time_index']] for b in S2_10_BANDS])
            scl_ok=np.isin(f['SCL'][rec['time_index']],SCL_VALID) if 'SCL' in f else np.ones_like(valid)
        expected_valid=scl_ok & np.isfinite(raw_image).all(axis=0) & (raw_image>0).all(axis=0) & (raw_mask!=255)
        assert np.array_equal(np.where(raw_mask==255,0,raw_mask),gt),rec['sample_id']
        assert np.array_equal(raw_image.astype(np.int16),cached_image),rec['sample_id']
        assert np.array_equal(expected_valid,valid),rec['sample_id']
        source_checks['mask_image_valid_exact']+=1
        source_checks['unknown_mask_pixels']+=int((raw_mask==255).sum())
        assert gt.shape == valid.shape == (128,128)
        digest = file_sha256(rec['cache'])
        assert digest == rec['cache_sha256'], rec['sample_id']
        hashes.append([rec['sample_id'],digest])
        gl, gn = label_objects(gt); objs = object_table(gl, gn, valid)
        # Coordinate IDs prove every mask/valid pixel follows the same permutation as image.
        coord = np.arange(gt.size).reshape(1,*gt.shape)
        for h in (False,True):
            for v in (False,True):
                for k in range(4):
                    image, mask, vm = _augment(coord,gt,valid,FixedAugment(h,v,k))
                    assert np.array_equal(mask,gt.ravel()[image[0]])
                    assert np.array_equal(vm,valid.ravel()[image[0]])
                    assert np.array_equal(np.sort(image.ravel()),np.arange(gt.size))
        edge = np.zeros_like(gt); edge[[0,-1],:]=True; edge[:,[0,-1]]=True
        edge_near = ndimage.binary_dilation(edge)
        invalid_near = ndimage.binary_dilation(~valid)
        for o in objs:
            region = gl == o['id']
            for group in [rec['split']+'/'+size_bin(o['area']),rec['split']+'/'+rec['event_id']+'/'+size_bin(o['area'])]:
                a = grouped[group]; a['objects']+=1; a['truncated']+=int(o['truncated'])
                a['direct_edge']+=int((region & edge).any())
                a['edge_or_1px_near']+=int((region & edge_near).any())
                a['invalid_or_adjacent']+=int((region & invalid_near).any())
                a['gt_px']+=int(region.sum()); a['valid_gt_px']+=int((region & valid).sum())
        if rec['split']=='train':
            samples.append({'sample_id':rec['sample_id'],'event_id':rec['event_id'],
                'final_exposure':int(exposure[rec['sample_id']]),
                'best_exposure':int(best_exposure[rec['sample_id']]) if best_exposure else None,
                'bins':sorted({size_bin(o['area']) for o in objs}),
                'foreground_px':int((gt & valid).sum()),'valid_px':int(valid.sum())})
        if idx % 500 == 0: print('data audit',idx,len(records),flush=True)
    event_stats = {}
    for event in sorted({s['event_id'] for s in samples}):
        sub=[s for s in samples if s['event_id']==event]
        event_stats[event]={'tiles':len(sub),'final_exposures':sum(s['final_exposure'] for s in sub),
            'foreground_px':sum(s['foreground_px'] for s in sub),'valid_px':sum(s['valid_px'] for s in sub)}
    exposure_by_size={k:{'tiles':sum(k in s['bins'] for s in samples),
        'final_min':min(s['final_exposure'] for s in samples if k in s['bins']),
        'final_max':max(s['final_exposure'] for s in samples if k in s['bins'])} for k,_,_ in BINS}
    gradients = {}
    for area in [1,4,9,19,49,199,400]:
        mask=torch.zeros(1,128,128);mask.reshape(-1)[1000:1000+area]=1
        logits=torch.zeros_like(mask,requires_grad=True);valid=torch.ones_like(mask,dtype=torch.bool)
        loss=_seg_loss(cfg,logits,mask,valid);loss.backward()
        g=logits.grad[mask.bool()]
        assert torch.isfinite(g).all() and (g<0).all()
        gradients[str(area)]={'positive_abs_gradient_sum':float(g.abs().sum()),'loss':float(loss.detach())}
    return {'cache_hashes':hashes,'augmentation_combinations_per_tile':16,'tiles_checked':len(records),
        'selected_source_array_checks':dict(source_checks),
        'prepare_script_sha256':file_sha256(ROOT/'scripts/prepare_segformer_data.py'),
        'edge_counts':dict(grouped),'exposure_by_event':event_stats,'exposure_by_size':exposure_by_size,
        'sample_exposures':samples,'best_stream_state_keys':list(state),
        'best_exposure_available':best_exposure is not None,'gradient_probe':gradients,
        'gradient_probe_note':'Synthetic zero logits at native 128x128 with actual configured loss; output gradient proves supervision inclusion, not model learning.'}


def fixed_cases(records):
    candidates=defaultdict(list)
    for rec in records:
        with np.load(rec['cache']) as z:
            lab,n=label_objects(z['mask']);objs=object_table(lab,n,z['valid'])
        for o in objs:
            if not o['truncated']:
                candidates[(rec['event_id'],size_bin(o['area']))].append({'sample_id':rec['sample_id'],
                    'event_id':rec['event_id'],'object_id':o['id'],'area':o['area'],'bbox':o['bbox'],'bin':size_bin(o['area'])})
    selected=[]; remaining=defaultdict(list)
    for key,rows in sorted(candidates.items()):
        # Four per event/size, no model output is read during selection.
        rows.sort(key=lambda r:hashlib.sha256(f"object-data-review:{r['sample_id']}:{r['object_id']}".encode()).hexdigest())
        selected.extend({**r,'selection':'size_stratum'} for r in rows[:4])
        remaining[key[0]].extend(rows[4:])
    # Some events lack tiny objects: retain empty strata, supplement to 24 using GT IDs only.
    for event,rows in remaining.items():
        rows.sort(key=lambda r:hashlib.sha256(f"object-data-review:{r['sample_id']}:{r['object_id']}".encode()).hexdigest())
        needed=24-sum(c['event_id']==event for c in selected)
        selected.extend({**r,'selection':'event_supplement'} for r in rows[:needed])
    ranks={k:i for i,(k,_,_) in enumerate(BINS)}
    return sorted(selected,key=lambda r:(r['event_id'],ranks[r['bin']],r['sample_id'],r['object_id']))


def plots(cases, records, probabilities, objects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    by_id={r['sample_id']:i for i,r in enumerate(records)}
    od={(r['sample_id'],r['id']):r for r in objects}
    for event in sorted({c['event_id'] for c in cases}):
        ec=[c for c in cases if c['event_id']==event]
        for page in range((len(ec)+5)//6):
            fig,axs=plt.subplots(6,4,figsize=(12,17),layout='constrained')
            for ax in axs.ravel():ax.set_axis_off()
            for row,c in enumerate(ec[page*6:page*6+6]):
                i=by_id[c['sample_id']]
                with np.load(records[i]['cache']) as z:
                    raw=z['x10'];gt=z['mask'].astype(bool);valid=z['valid'].astype(bool)
                rgb=raw[[2,1,0]].transpose(1,2,0).astype(float)
                lo,hi=np.percentile(rgb,[2,98],axis=(0,1));rgb=np.clip((rgb-lo)/np.maximum(hi-lo,1),0,1)
                p=probabilities[i];pred=(p>=.5)&valid
                err=np.zeros((*gt.shape,3));err[gt & pred]=[.2,.75,.35];err[~gt & pred]=[.9,.25,.2];err[gt & ~pred & valid]=[.2,.45,1.];err[~valid]=[.5,.5,.5]
                y0,x0,y1,x1=c['bbox'];sl=np.s_[max(0,y0-10):min(128,y1+10),max(0,x0-10):min(128,x1+10)]
                for col,arr in enumerate([rgb,gt,p,err]):
                    axs[row,col].imshow(arr[sl],interpolation='nearest',**({'cmap':'gray','vmin':0,'vmax':1} if col==1 else {'cmap':'magma','vmin':0,'vmax':1} if col==2 else {}))
                o=od[(c['sample_id'],c['object_id'])]
                category_cn={'matched':'匹配成功','no_hit_below_0.1':'最大概率<0.1','no_hit_0.1_to_0.5':'最大概率0.1–0.5','overlap_but_unmatched':'有重叠但未匹配'}[o['category']]
                axs[row,0].set_title(f"RGB：{c['sample_id']} / 对象{c['object_id']}\n{c['area']}像素；{category_cn}",fontsize=8)
                for col,title in enumerate(['原始标注','概率 0–1','绿TP / 红FP / 蓝FN / 灰无效'],start=1):
                    axs[row,col].set_title(title,fontsize=9)
                # Mark the selected object even when the crop contains neighbours.
                from matplotlib.patches import Rectangle
                sy,sx=max(0,y0-10),max(0,x0-10)
                axs[row,1].add_patch(Rectangle((x0-sx-.5,y0-sy-.5),x1-x0,y1-y0,fill=False,edgecolor='#ffcf33',linewidth=1))
            fig.suptitle(f'{event} 固定案例 {page+1}：每行同一区域；不插值放大；上下文含其他对象',fontsize=12)
            fig.savefig(ASSET/f'cases_{event}_{page+1}.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    cats=['matched','no_hit_below_0.1','no_hit_0.1_to_0.5','overlap_but_unmatched']
    labels=['匹配成功','目标内最大概率<0.1','目标内最大概率0.1–0.5','有重叠但未匹配']
    bottom=np.zeros(6)
    for cat,label in zip(cats,labels):
        values=np.array([sum(o['category']==cat and o['bin']==k and not o['truncated'] for o in objects) for k,_,_ in BINS])
        den=np.array([sum(o['bin']==k and not o['truncated'] for o in objects) for k,_,_ in BINS])
        v=values/den*100;axs[0].bar(range(6),v,bottom=bottom,label=label);bottom+=v
    axs[0].set_xticks(range(6),['1–4','5–9','10–19','20–49','50–199','≥200']);axs[0].set_ylabel('可评价对象占比 %');axs[0].legend(fontsize=8);axs[0].set_title('对象漏检不等于完全没有响应')
    for name,label in [('boundary','一像素边界'),('interior','内部')]:
        vals=[]
        for k,_,_ in BINS:
            sub=[o for o in objects if o['bin']==k];den=sum(o[name+'_px'] for o in sub)
            vals.append(100*sum(o[name+'_fn'] for o in sub)/den if den else np.nan)
        axs[1].plot(range(6),vals,'-o',label=label)
    axs[1].set_xticks(range(6),['1–4','5–9','10–19','20–49','50–199','≥200']);axs[1].set_ylabel('有效标注像素中漏检比例 %');axs[1].legend();axs[1].set_title('大对象的内部也存在漏分')
    fig.savefig(ASSET/'diagnosis.png',dpi=160);plt.close(fig)
    pipeline=json.loads((ASSET/'pipeline.json').read_text())
    summary=json.loads((ASSET/'summary.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    events=pipeline['exposure_by_event'];names=list(events);y=np.arange(len(names))
    total_px=sum(v['foreground_px'] for v in events.values())
    axs[0].barh(y-.18,[events[e]['tiles']/2223*100 for e in names],height=.36,label='训练图块份额')
    axs[0].barh(y+.18,[events[e]['foreground_px']/total_px*100 for e in names],height=.36,label='有效前景像素份额')
    axs[0].set_yticks(y,names,fontsize=8);axs[0].legend(fontsize=8);axs[0].set_xlabel('%');axs[0].set_title('自然采样保留了事件不均衡')
    for split,label in [('train','训练'),('val','验证')]:
        rows=[pipeline['edge_counts'][split+'/'+k] for k,_,_ in BINS]
        axs[1].plot(range(6),[100*r['direct_edge']/r['objects'] for r in rows],'-o',label=label)
    axs[1].set_xticks(range(6),['1–4','5–9','10–19','20–49','50–199','≥200']);axs[1].set_ylabel('原始对象中直接触边的比例 %');axs[1].legend();axs[1].set_title('原始小连通域包含大量边缘对象')
    for pos,(t,label) in enumerate([('.5','阈值0.5'),('.25','阈值0.25')]):
        p=summary['threshold_diagnostic'][t]['pixel']
        axs[2].bar(np.arange(3)+(pos-.5)*.35,[p[k]/1000 for k in ['tp','fp','fn']],width=.35,label=label)
    axs[2].set_xticks(range(3),['TP正确前景','FP误检','FN漏检']);axs[2].set_ylabel('千像素');axs[2].legend();axs[2].set_title('降阈值增加的误检多于找回的前景')
    fig.savefig(ASSET/'data_path.png',dpi=150);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);ASSET.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    cfg=json.loads((RUN/'resolved_config.json').read_text())
    records=read_manifest(resolve_path(cfg['segformer']['data']['manifest']))
    dev=[r for r in records if r['split']=='val']
    assert len(dev)==669 and sum(r['split']=='train' for r in records)==2223
    checkpoint=torch.load(RUN/'best.pt',map_location='cpu',weights_only=False)
    assert checkpoint['config']==cfg
    cases=fixed_cases(dev);write_json(ASSET/'fixed_cases.json',cases)
    pipeline=pipeline_audit(records,cfg,checkpoint);write_json(ASSET/'pipeline.json',pipeline)
    source_paths=[RUN/'best.pt',RUN/'resolved_config.json',RUN/'sample_exposures.json',
        resolve_path(cfg['segformer']['data']['manifest']),resolve_path(cfg['segformer']['data']['stats']),
        ROOT/'src/landslide_seg/data/segformer_dataset.py',ROOT/'src/landslide_seg/models/segformer.py',
        ROOT/'src/landslide_seg/training/segformer_metrics.py',ROOT/'src/landslide_seg/training/segformer_trainer.py',
        ROOT/'src/landslide_seg/training/losses.py',ROOT/'src/landslide_seg/training/update_training.py']
    identity={p.relative_to(ROOT).as_posix():file_sha256(p) for p in source_paths}
    cache=OUT/'baseline_dev_probabilities.npz';meta=OUT/'baseline_dev_probabilities.json'
    if cache.exists() and meta.exists() and json.loads(meta.read_text())==identity:
        with np.load(cache) as z: probabilities=z['probabilities'];assert list(z['sample_ids'])==[r['sample_id'] for r in dev]
    else:
        model,_=load_checkpoint(RUN/'best.pt',device='cpu')
        ds=S2SegDataset(dev,checkpoint['stats'],augment=False,preload=True)
        probabilities=[]
        with torch.inference_mode():
            for start in range(0,len(ds),32):
                batch=collate([ds[i] for i in range(start,min(start+32,len(ds)))])
                output=model(batch['image'])['logits'].float()
                assert output.shape[1:]==(1,128,128)
                probabilities.extend(torch.sigmoid(output[:,0]).numpy())
                print('DEV inference',start+len(output),len(ds),flush=True)
        probabilities=np.asarray(probabilities,dtype=np.float32)
        np.savez_compressed(cache,probabilities=probabilities,sample_ids=[r['sample_id'] for r in dev]);write_json(meta,identity)
    objects=[];tile_rows=[];threshold_rows=[]
    for r,p in zip(dev,probabilities):
        with np.load(r['cache']) as z:gt=z['mask'].astype(bool);valid=z['valid'].astype(bool)
        objects.extend({**o,'sample_id':r['sample_id'],'event_id':r['event_id']} for o in describe_objects(p,gt,valid))
        for threshold,target in [(.5,tile_rows),(.25,threshold_rows)]:
            target.append(evaluate_tile_v2(p,gt,valid,threshold=threshold,sample_id=r['sample_id'],event_id=r['event_id']))
    baseline=json.loads((RUN/'eval/eval_best_val.json').read_text())
    overall=summarize_tiles_v2(tile_rows);by_event=summarize_by_event_v2(tile_rows)
    for k,_,_ in BINS:
        assert sum(o['matched'] and o['bin']==k for o in objects)==overall['objects']['fine_bins'][k]['n_matched_evaluable']
    deltas={k:overall['pixel'][k]-baseline['overall']['pixel'][k] for k in ['tp','fp','fn','tn']}
    # CPU/GPU numerical ties can differ at threshold; report exact discrepancies, never hide them.
    assert max(abs(v) for v in deltas.values()) <= 50, deltas
    grouped={}
    for group in ['all']+sorted({o['event_id'] for o in objects}):
        grouped[group]={}
        for k,_,_ in BINS:
            sub=[o for o in objects if o['bin']==k and (group=='all' or o['event_id']==group)]
            ev=[o for o in sub if not o['truncated']]
            grouped[group][k]={'all_objects':len(sub),'evaluable':len(ev),'categories':dict(Counter(o['category'] for o in ev)),
                'unmatched_merge_flag':sum(not o['matched'] and o['merge_flag'] for o in ev),
                'unmatched_fragment_flag':sum(not o['matched'] and o['fragment_flag'] for o in ev),
                'zero_hit_nearby_2px':sum(o['hit_px']==0 and o['nearby_prediction_2px'] for o in ev),
                **{f:sum(o[f] for o in sub) for f in ['valid_px','hit_px','boundary_px','boundary_fn','interior_px','interior_fn']}}
    threshold_summary={}
    for t,rows in [('.5',tile_rows),('.25',threshold_rows)]:
        ov=summarize_tiles_v2(rows);events=summarize_by_event_v2(rows)
        threshold_summary[t]={'pixel':ov['pixel'],'event_mean_iou':float(np.mean([e['pixel']['iou'] for e in events.values()])),
            'by_event':{e:v['pixel'] for e,v in events.items()},'objects':ov['objects']}
    result={'identity':identity,'audit_script_sha256':file_sha256(Path(__file__)),
        'command':'python -X utf8 scripts/audit_object_errors.py','device':'cpu','threads':4,
        'checkpoint_step':checkpoint['global_step'],'threshold':.5,'object_iou':.25,'grouped':grouped,
        'reproduction_pixel_count_delta':deltas,'threshold_diagnostic':threshold_summary,
        'limitations':['0.1 is a descriptive probability cutoff, not a calibrated confidence boundary.',
            'Merge/fragment flags overlap; nearby prediction does not prove registration error.',
            'No original full-scene object identity; edge components cannot be declared independent landslides.',
            'Threshold .25 is a diagnostic only, not a newly selected model setting.',
            'Fixed cases selected by GT IDs only: up to four per event/size, supplemented to 24 per event where strata are sparse; not independent events.']}
    with (OUT/'per_object.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(objects[0]));writer.writeheader();writer.writerows(objects)
    write_json(ASSET/'summary.json',result)
    plots(cases,dev,probabilities,objects)
    print(json.dumps({'pixel_count_delta':deltas,'grouped':grouped['all'],'best_stream_state_keys':pipeline['best_stream_state_keys']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
