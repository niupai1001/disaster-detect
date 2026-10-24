"""只读目标尺寸审计；用既有评价分解漏检，不重训或修改标签。"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from landslide_seg.training.segformer_metrics import label_objects, object_table

BINS = [('1_4',1,5),('5_9',5,10),('10_19',10,20),('20_49',20,50),('50_199',50,200),('ge200',200,None)]


def blank():
    return {k:dict(n_objects=0,n_objects_evaluable=0,gt_px=0,n_tiles=0) for k,_,_ in BINS}


def main():
    manifest=ROOT/'manifests/v4r1/pool_100.jsonl'
    records=[json.loads(l) for l in manifest.read_text(encoding='utf-8-sig').splitlines()]
    groups=defaultdict(blank); tiles=defaultdict(int); valid_px=defaultdict(int); hashes=[]
    for r in records:
        with np.load(r['cache']) as z:
            gt=np.asarray(z['mask'],dtype=bool);valid=np.asarray(z['valid'],dtype=bool)
        lab,n=label_objects(gt);objs=object_table(lab,n,valid)
        keys=[r['split'],r['split']+'/'+r['event_id']]
        hashes.append({'sample_id':r['sample_id'],'mask_valid_sha256':hashlib.sha256(gt.tobytes()+valid.tobytes()).hexdigest()})
        for key in keys:
            tiles[key]+=1;valid_px[key]+=int(valid.sum())
        for name,lo,hi in BINS:
            os=[o for o in objs if o['area']>=lo and (hi is None or o['area']<hi)]
            pixels=int((np.isin(lab,[o['id'] for o in os]) & valid).sum()) if os else 0
            for key in keys:
                b=groups[key][name];b['n_objects']+=len(os);b['n_objects_evaluable']+=sum(not o['truncated'] for o in os)
                b['gt_px']+=pixels;b['n_tiles']+=bool(os)
    src=ROOT/'outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234/eval/eval_best_val.json'
    ev=json.loads(src.read_text());fine=ev['overall']['objects']['fine_bins'];pixel=ev['overall']['pixel']
    for k,_,_ in BINS:
        for f in ('n_objects','n_objects_evaluable','gt_px'):
            assert groups['val'][k][f]==fine[k][f],(k,f,groups['val'][k],fine[k])
    assert sum(b['gt_px'] for b in fine.values())==pixel['tp']+pixel['fn']
    assert sum(b['gt_px']-b['hit_px'] for b in fine.values())==pixel['fn']
    rows=[]
    for k,_,_ in BINS:
        b=fine[k];fn=b['gt_px']-b['hit_px']
        rows.append({'bin':k,**b,'fn':fn,'foreground_share':b['gt_px']/(pixel['tp']+pixel['fn']),
            'fn_share':fn/pixel['fn'],'evaluable_object_share':b['n_objects_evaluable']/ev['overall']['objects']['n_gt_evaluable']})
    oracle={}
    for label,keys in [('1_4',['1_4']),('1_19',['1_4','5_9','10_19']),('1_49',['1_4','5_9','10_19','20_49'])]:
        repaired=sum(fine[k]['gt_px']-fine[k]['hit_px'] for k in keys)
        per_event={}
        for event,e in ev['by_event'].items():
            f=e['objects']['fine_bins'];p=e['pixel'];delta=sum(f[k]['gt_px']-f[k]['hit_px'] for k in keys)
            per_event[event]=(p['tp']+delta)/(p['tp']+p['fp']+p['fn'])
        oracle[label]={'pooled_iou':(pixel['tp']+repaired)/(pixel['tp']+pixel['fp']+pixel['fn']),
                       'event_mean_iou':float(np.mean(list(per_event.values()))),'by_event':per_event}
    result={'scope':'原TRAIN2223/DEV669；不检查日期标注一致性，不涉及外部9通道参考数据',
        'definition':'8邻接图块内连通对象；面积按原GT计，像素统计限定valid；evaluable排除边缘及无效区邻接对象，沿用metric_v2。图块对象不等于全景独立滑坡。',
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'tile_counts':dict(tiles),
        'valid_pixels':dict(valid_px),'distribution':dict(groups),'mask_hashes':hashes,
        'baseline_eval':src.relative_to(ROOT).as_posix(),'baseline_eval_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'baseline_pixel':pixel,'baseline_event_mean_iou':ev['event_mean_iou'],'baseline_bins':rows,
        'oracle_fn_repair':oracle,'oracle_limit':'仅将指定尺寸GT中的全部FN改TP、保持所有FP和其他像素不变的算术诊断；不是可实现模型表现或消除全部小目标错误的上限。'}
    out=ROOT/'docs/assets/research';(out/'size_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    labels=['1–4','5–9','10–19','20–49','50–199','≥200'];x=np.arange(6)
    for split,color in [('train','#5285b3'),('val','#dd9146')]:
        den=sum(b['n_objects_evaluable'] for b in groups[split].values());offset=-.18 if split=='train' else .18
        axs[0].bar(x+offset,[100*groups[split][k]['n_objects_evaluable']/den for k,_,_ in BINS],width=.36,label={'train':'训练','val':'验证'}[split],color=color)
    axs[0].set_title('可评价对象数量占比');axs[0].set_ylabel('%');axs[0].legend()
    axs[1].plot(x,[100*r['object_recall_evaluable'] for r in rows],'-o',label='对象召回')
    axs[1].plot(x,[100*r['pixel_recall'] for r in rows],'-s',label='前景像素Recall');axs[1].set_title('成熟单期参照的检出能力');axs[1].set_ylabel('%');axs[1].legend()
    axs[2].bar(x,[100*r['fn_share'] for r in rows],color='#ad6756');axs[2].set_title('全部漏检像素来自哪些尺寸');axs[2].set_ylabel('占所有FN的百分比')
    for a in axs:a.set_xticks(x,labels);a.set_xlabel('图块内对象面积（像素）');a.grid(axis='y',alpha=.2)
    fig.savefig(out/'size_diagnosis.png',dpi=160);plt.close(fig)
    print(json.dumps({'distribution':{k:groups[k] for k in ['train','val']},'baseline_bins':rows,'oracle':oracle},ensure_ascii=False))


if __name__=='__main__':main()
