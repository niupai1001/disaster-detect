"""Read-only review of saved V5 results; writes separate review artifacts only."""
from pathlib import Path
import ast, csv, hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'docs/event_cv_v5_review_2026-10-20'
OUT.mkdir(exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')
hashes = {}
def read(p):
    p = ROOT / p
    b = p.read_bytes()
    hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(b).hexdigest()
    return json.loads(b)
def lines(p):
    p = ROOT / p
    b = p.read_bytes()
    hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(b).hexdigest()
    return [json.loads(s) for s in b.decode('utf-8').splitlines() if s.strip()]
def ids(rows): return {x['sample_id'] for x in rows}
def iou(c): return c['tp'] / (c['tp'] + c['fp'] + c['fn'])

families = {'control': ('event_cv_v5', 'SegFormer-LateDetail-EventCVV5'),
            'segformer_rgb': ('event_cv_v5_probe', 'Probe-segformer_b0'),
            'dofa_rgb': ('event_cv_v5_probe', 'Probe-dofa_rgb'),
            'dofa_10_wrong_wavelength': ('event_cv_v5_probe', 'Probe-dofa_10')}
review = {'scope': 'saved artifacts + CPU targeted input checks; no training or GPU inference',
          'folds': {}, 'runs': {}, 'temporal': {}, 'findings': {}}
outer_union = set()
baseline_tiles = {}
for f in ['F1','F2','F3','F4']:
    p = f'manifests/event_cv_v5/{f}'
    pool, outer = lines(p+'/pool_100.jsonl'), lines(p+'/test.jsonl')
    tr = [r for r in pool if r['split']=='train']
    va = [r for r in pool if r['split']=='val']
    sets = [ids(tr),ids(va),ids(outer)]
    assert all(not sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)])
    events = [{r['event_id'] for r in x} for x in [tr,va,outer]]
    assert all(not events[a]&events[b] for a,b in [(0,1),(0,2),(1,2)])
    stats = read(p+'/stats_train.json')
    fit_key = next(k for k in stats if isinstance(stats[k],list) and stats[k] and isinstance(stats[k][0],str) and '_s2_' in stats[k][0])
    assert set(stats[fit_key]) <= sets[0]
    assert not outer_union & sets[2]
    outer_union |= sets[2]
    review['folds'][f] = {'n_train':len(tr),'n_inner':len(va),'n_outer':len(outer),
                         'outer_events':sorted(events[2]),'train_only_stats':True}
    for key,(sub,name) in families.items():
        d = f'outputs/segformer/{sub}/{name}-{f}/1234'
        s = read(d+'/train_summary.json')
        cfg = read(d+'/resolved_config.json')
        assert s['complete'] and s['global_step']==3600 and s['seed']==1234
        assert all((ROOT/d/(tag+'.pt')).is_file() for tag in ['best','last'])
        h=s['history']; best=max(h,key=lambda r:r['val_iou'])
        assert best['global_step']==s['best_step']
        integrity=[r.get('probe_integrity') for r in h if r.get('probe_integrity')]
        assert all(x['encoder_unchanged'] and not x['encoder_has_grad'] and not x['encoder_training'] for x in integrity)
        records={}
        for tag in ['best_val','best_test','last_val','last_test']:
            epath=ROOT/d/'eval'/f'eval_{tag}.json'
            if not epath.exists(): continue
            e=read(epath)
            cp=ROOT/d/'eval'/f'per_tile_{tag}.csv'
            with cp.open(encoding='utf-8-sig',newline='') as fh: tiles=list(csv.DictReader(fh))
            hashes[cp.relative_to(ROOT).as_posix()]=hashlib.sha256(cp.read_bytes()).hexdigest()
            expected=sets[1] if tag.endswith('val') else sets[2]
            assert {r['sample_id'] for r in tiles}==expected and len(tiles)==len(expected)
            identity={r['sample_id']:(r['event_id'],r['gt_px'],r['valid_px']) for r in tiles}
            bk=(f,tag)
            if key=='control': baseline_tiles[bk]=identity
            else: assert identity==baseline_tiles[bk]
            counts={k:sum(int(r[k]) for r in tiles) for k in ['tp','fp','fn','tn']}
            assert abs(iou(counts)-e['overall']['pixel']['iou'])<1e-12
            records[tag]={'iou':iou(counts),'ap':e['overall']['pixel']['average_precision'],
                          'by_event':{k:v['pixel'] for k,v in e['by_event'].items()}}
        review['runs'][key+'_'+f]={'best_step':s['best_step'],'seconds':s['seconds'],
            'probe_checks':len(integrity),'params':s['params'],'eval':records,
            'history': [{'step':r['global_step'],'train':r['train_clean']['pooled']['iou'],
                         'inner':r['val_iou']} for r in h]}
review['unique_outer_samples']=len(outer_union)

for mode in ['single','multi']:
    d=f'outputs/utae_pair/utae_{mode}/1234'
    s=read(d+'/train_summary.json'); e=read(d+'/eval_summary.json')
    h=lines(d+'/metrics.jsonl')
    assert len(h)==75 and [r['epoch'] for r in h]==list(range(1,76))
    assert sum(r['skipped'] for r in h)==0
    assert max(h,key=lambda r:r['val']['pooled_iou'])['epoch']==s['best_epoch']
    for ck in ['best','last']:
        assert (ROOT/d/(ck+'.pt')).is_file()
        for split in ['val','test']:
            x=e[ck][split]
            for scope in ['valid_scope','all_pixels']: assert abs(iou(x[scope])-x[scope]['iou'])<1e-12
    review['temporal'][mode]={'best_epoch':s['best_epoch'],'seconds':s['seconds'],
        'init_digest':s['meta']['init_digest'],'eval':e,
        'history':[{'epoch':r['epoch'],'iou':r['val']['pooled_iou']} for r in h]}
assert review['temporal']['single']['init_digest']==review['temporal']['multi']['init_digest']

# Execute preprocessing contract without loading pretrained encoders.
from landslide_seg.models.frozen_probe import FrozenProbe, S2_WAVELENGTHS_10, RGB_CH, RGB_WAVELENGTHS
import torch
obj=object.__new__(FrozenProbe)
torch.nn.Module.__init__(obj)
obj.stats={'mean':[0]*10,'std':[1]*10}; obj.encoder_kind='dofa_10'
_, ten_waves=obj._prep(torch.zeros(1,10,2,2))
review['findings']['waves']={'actual':list(ten_waves),'rgb_reference':list(RGB_WAVELENGTHS),
    'ten_at_rgb_channels':[ten_waves[i] for i in RGB_CH],
    'matches_rgb_contract':tuple(ten_waves[i] for i in RGB_CH)==RGB_WAVELENGTHS}
assert not review['findings']['waves']['matches_rgb_contract']

# One sample per event: actual date tensors, frame equality, and raw missing values.
from landslide_seg.data.temporal_dataset import TemporalPairDataset
rows=lines('manifests/official_single/trainval.jsonl')+lines('manifests/official_single/test.jsonl')
sample={}
for r in rows: sample.setdefault(r['event_id'],r)
selected=list(sample.values())
stats=read('manifests/official_single_geo/stats_dem.json')
single=TemporalPairDataset(selected,stats,mode='single')
multi=TemporalPairDataset(selected,stats,mode='multi')
idx=read('data/s2_temporal_v5/index.json')
lookup={r['sample_id']:r for r in idx['rows']}
import xarray as xr
checks=[]
for j,r in enumerate(selected):
    a,b=single[j],multi[j]; t=int(r['time_index'])
    assert torch.equal(a['image'][0],b['image'][t])
    assert torch.equal(a['mask'],b['mask']) and torch.equal(a['valid'],b['valid'])
    with xr.open_dataset(ROOT/'data/sen12'/f"{r['sample_id']}.nc",engine='h5netcdf') as ds:
        ds=ds.sortby('time'); bands=stats['bands'][:10]
        ar=np.stack([ds[k].values for k in bands],axis=1)
        times=ds.time.values
        scl=ds['SCL'].values
        record={'event':r['event_id'],'id':r['sample_id'],'selected_index':t,
            'source_time_dtype':str(times.dtype),'first_dates':lookup[r['sample_id']]['dates'][:3],
            'single_date_tensor':a['dates'].tolist(),'multi_date_tensor':b['dates'].tolist(),
            'raw_nonfinite':int((~np.isfinite(ar)).sum()),
            'zero_fraction_per_frame':(ar==0).mean(axis=(1,2,3)).tolist(),
            'scl_invalid_fraction_per_frame':np.isin(scl,[0,1,3,8,9,10,11,255]).mean(axis=(1,2)).tolist()}
        checks.append(record)
review['targeted_input_checks']=checks
review['findings']['object_metric']='UTAE scale_recall is any-pixel GT hit, not one-to-one IoU matching; boundary metric absent'
review['findings']['temporal_dates']='arange(T), real dates ignored; single always zero'
review['findings']['missing_values']='builder directly casts raw arrays to int16; no explicit missing mask or mean fill'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
labels=['完整系统：单期十波段','冻结 SegFormer：RGB','冻结 DOFA：RGB','冻结 DOFA：十波段（波长错误）']
for k,label in zip(families,labels):
    vals=[review['runs'][k+'_'+f]['eval']['best_test']['iou'] for f in ['F1','F2','F3','F4']]
    ax[0].plot(range(1,5),vals,marker='o',label=label,linestyle='--' if 'wrong' in k else '-')
ax[0].set(xticks=[1,2,3,4],xticklabels=['Italy留出','吉尔吉斯等','新西兰等','非洲/加勒比等'],ylabel='外部留出 pooled IoU',title='每折同集合比较；错误波长组仅保留历史记录')
ax[0].legend(fontsize=8)
for mode,label in [('single','单期 + DEM'),('multi','15期 + DEM（序号编码）')]:
    h=review['temporal'][mode]['history']
    ax[1].plot([r['epoch'] for r in h],[r['iou'] for r in h],label=label)
ax[1].set(xlabel='Epoch',ylabel='VAL pooled IoU',title='多帧收益明显；真实日期尚未接入')
ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT/'results.png',dpi=160); plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
for ax,f in zip(axes.flat,['F1','F2','F3','F4']):
    h=review['runs']['control_'+f]['history']
    ax.plot([x['step'] for x in h],[x['train'] for x in h],label='TRAIN pooled IoU')
    ax.plot([x['step'] for x in h],[x['inner'] for x in h],label='INNER event mean IoU')
    ax.set(title=f+' 完整系统',xlabel='成功更新步数',ylabel='IoU'); ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'control_curves.png',dpi=160);plt.close(fig)
for rel in ['src/landslide_seg/models/frozen_probe.py','src/landslide_seg/data/temporal_dataset.py',
            'scripts/build_temporal_cache.py','scripts/evaluate_utae.py','scripts/train_utae.py',
            'scripts/review_event_cv_v5.py']:
    hashes[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
review['sha256']=hashes
(OUT/'verification.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
compact={'outer_samples':len(outer_union),'verified_runs':len(review['runs'])+len(review['temporal']),
         'folds':review['folds'],'waves':review['findings']['waves'],
         'sample_checks':len(checks),'sample_raw_nonfinite':sum(x['raw_nonfinite'] for x in checks),
         'sample_frames_with_over_half_zero':sum(sum(v>.5 for v in x['zero_fraction_per_frame']) for x in checks)}
print(json.dumps(compact,ensure_ascii=False,indent=2))
