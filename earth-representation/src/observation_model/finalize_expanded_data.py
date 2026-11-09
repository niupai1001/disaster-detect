"""Freeze and document the expanded data only after its queue stops."""
from collections import Counter,defaultdict
from datetime import datetime
import json
from pathlib import Path
import hashlib
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window,bounds
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.strtree import STRtree

from .expansion_data import ROOT,OUT,read,save,lines,digest


def main():
    status=read(OUT/'status.json')
    if status['status'] not in ['all_data_tasks_attempted_model_training_disabled','download_cap_reached_remaining_tasks_pending','current_local_snapshot_queue_retained']:
        raise RuntimeError('Data queue must finish or reach its approved cap before final release')
    if (OUT/'release.json').exists():raise FileExistsError('Preserve finalized data release')
    rows=[read(p) for p in sorted((OUT/'cache').glob('*.json'))]
    calibration=read(OUT/'s2_calibrated_v2/summary.json')
    corrected={r['id']:r for r in [read(p) for p in (OUT/'s2_calibrated_v2/cache').glob('*.json')]}
    history={r['id']:r for r in lines(OUT/'s2_calibrated_v2/reference_results.jsonl')}
    unresolved={k:v for k,v in history.items() if k not in corrected and v['status']=='unresolved'}
    save(OUT/'calibration_pending.json',list(unresolved.values()))
    rows=[corrected.get(r['id'],r) for r in rows if r['id'] not in unresolved]
    original=lines(OUT/'canonical_input.jsonl')
    held=[r for r in original if r['effective_partition'] in ['GUARD','TEST_ISOLATION']]
    held_geos=[box(*r['bbox']) for r in held];held_tree=STRtree(held_geos)
    geometries=[]
    for row in rows:
        extents=[]
        for band in row['bands']:
            h,w=band['shape'];extent=bounds(Window(0,0,w,h),Affine(*band['transform']))
            extents.append(transform_bounds(band['crs'],'EPSG:4326',*extent))
        extent=[min(a[0] for a in extents),min(a[1] for a in extents),max(a[2] for a in extents),max(a[3] for a in extents)]
        row['source_role']=row['role'];row['actual_native_bbox']=extent;geometries.append(box(*extent))
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def merge(i,j):
        a,b=find(i),find(j);parent[max(a,b)]=min(a,b)
    acquisitions={};inherited={};tree=STRtree(geometries);guard=set()
    for i,row in enumerate(rows):
        for store,k in [(acquisitions,row['acquisition_key']),(inherited,row['isolation_group'])]:
            if k in store:merge(i,store[k])
            else:store[k]=i
        g=geometries[i]
        for j in tree.query(g,predicate='intersects'):
            j=int(j)
            if j>i and g.intersection(geometries[j]).area>1e-12:merge(i,j)
        if any(g.intersection(held_geos[int(j)]).area>1e-12 for j in held_tree.query(g)):guard.add(i)
    groups=defaultdict(set)
    for i,row in enumerate(rows):groups[find(i)].add('GUARD' if i in guard else row['role'])
    role_changes=[]
    for i,row in enumerate(rows):
        effective='GUARD' if 'GUARD' in groups[find(i)] else 'VAL' if 'VAL' in groups[find(i)] else 'TRAIN'
        if effective!=row['role']:role_changes.append(dict(id=row['id'],before=row['role'],after=effective,reason='actual native support/acquisition isolation'))
        row['role']=effective;row['release_isolation_group']='expanded_group_'+str(find(i))
    aliases=[];retained=[];seen={}
    for row in rows:
        signature=json.dumps([row['instrument'],row['acquisition_key'],row['processing_domain'],[(b['band_key'],b['crs'],b['transform'],b['shape']) for b in row['bands']]],sort_keys=True)
        duplicate=None
        for prior in seen.get(signature,[]):
            with np.load(row['cache']) as a,np.load(prior['cache']) as b:
                equal=set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
            if equal:duplicate=prior;break
        if duplicate is not None:aliases.append(dict(id=row['id'],canonical_id=duplicate['id']));continue
        seen.setdefault(signature,[]).append(row);retained.append(row)
    rows=retained
    assert len({r['id'] for r in rows})==len(rows)
    assert not ({r['release_isolation_group'] for r in rows if r['role']=='TRAIN'}&{r['release_isolation_group'] for r in rows if r['role']=='VAL'})
    path=OUT/'members.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    stats=defaultdict(lambda:[0,0.,0.])
    for i,row in enumerate(rows):
        if row['role']!='TRAIN':continue
        with np.load(row['cache'],allow_pickle=False) as z:
            for b in row['bands']:
                k=b['band_key'];v=z[k][z[k+'__valid']].astype('float64');s=stats[row['instrument']+':'+k]
                s[0]+=len(v);s[1]+=float(v.sum());s[2]+=float((v*v).sum())
        if i%1000==0:print('TRAIN_STATISTICS',i,flush=True)
    statistics={k:dict(valid_pixels=n,mean=s/n,std=float(np.sqrt(max(q/n-(s/n)**2,0)))) for k,(n,s,q) in stats.items()}
    if any(s['std']<=0 for s in statistics.values()):raise ValueError('nonpositive TRAIN-only standard deviation')
    save(OUT/'statistics.json',dict(scope='TRAIN-only physical reflectance by instrument and band; no held-out fit',bands=statistics))
    local=lines(OUT/'local_events.jsonl');remote=lines(OUT/'remote_events.jsonl')
    attempted={r['id'] for r in remote if r['status'] in ['accepted','excluded']}
    tasks=lines(OUT/'tasks.jsonl');pending=[dict(id=t['id'],kind=t['kind']) for t in tasks if t['kind'] in ['cached_landsat','landsat_for_local_anchor'] and t['id'] not in attempted]
    save(OUT/'pending_tasks.json',pending);save(OUT/'duplicate_aliases.json',aliases);save(OUT/'role_adjustments.json',role_changes)
    config=read(ROOT/'observation_model/config.json')
    assert config['download_enabled'] is False and config['recover_nonconforming_data'] is False
    baseline=read(ROOT/'observation_model/outputs/encoder_v2/G0_audit/execution_boundary.json')['policy_sha256']
    assert digest(ROOT/'observation_model/execution_policy.json')==baseline
    authorization=read(OUT/'authorization.json')
    total_accounted=(read(OUT/'network.json')['received_bytes']+
        read(OUT/'s2_calibration_network.json')['received_bytes']+
        authorization['prior_bytes']+2693212+32*1024**2)
    assert total_accounted<=authorization['network_cap_bytes'], 'Aggregate transfer cap exceeded'
    release=dict(data_release_accepted=True,model_training_authorized=False,G1_started=False,optimizer_updates=0,
        manifest_sha256=digest(path),statistics_sha256=digest(OUT/'statistics.json'),
        counts=dict(Counter(r['role']+'/'+r['sensor'] for r in rows)),
        provenance_counts=dict(Counter(r['provenance'] for r in rows)),band_counts=dict(Counter(str(r['band_count']) for r in rows)),
        duplicate_aliases=len(aliases),role_adjustments=len(role_changes),pending_tasks=len(pending),
        stage_status=status['status'],network=read(OUT/'network.json'),
        S2_calibration=calibration,calibration_pending=len(unresolved),
        total_new_metered_bytes=read(OUT/'network.json')['received_bytes']+read(OUT/'s2_calibration_network.json')['received_bytes'],
        interrupted_read_reserve_bytes=32*1024**2,
        total_budget_accounted_bytes=total_accounted,
        local_exclusion_reasons=dict(Counter(r.get('error') for r in local if r['status']=='excluded')),
        remote_status_counts=dict(Counter(r['status'] for r in remote)),
        remote_exclusion_reasons=dict(Counter(r.get('error') for r in remote if r['status']=='excluded')),
        actual_test_pixel_reads=0,srf_physical_ingestion='pending_G1',normalization_statistics_fitted='TRAIN only',
        HLS='Existing raw HLS contains partial bands and held-out geography; authenticated acquisition was not retried. Landsat path used.',
        semantics='Partial real band observations are valid. Interpolated channels omitted from native targets. New Landsat anchors are geography-associated, not certified same-date cross-source pairs.',
        old_source_policy_sha256=baseline)
    save(OUT/'release.json',release)
    from .expanded_dataset import ExpandedOpticalDataset
    train=ExpandedOpticalDataset(OUT,'TRAIN');val=ExpandedOpticalDataset(OUT,'VAL')
    for dataset in [train,val]:
        item=dataset[0]
        assert item['bands'] and all(b['reflectance'].shape==b['pixel_valid'].shape for b in item['bands'])
        assert all(np.isfinite(b['reflectance'][b['pixel_valid']]).all() for b in item['bands'])
    save(OUT/'loader_acceptance.json',dict(status='PASS',train=len(train),val=len(val),finite_real_input=True,optimizer_updates=0))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    instruments=['S2','L8','L9'];x=np.arange(3)
    a=[release['counts'].get('TRAIN/'+s,0) for s in instruments];b=[release['counts'].get('VAL/'+s,0) for s in instruments]
    bars=axes[0].bar(x-.18,a,.35,label='训练');axes[0].bar_label(bars,padding=3)
    bars=axes[0].bar(x+.18,b,.35,label='验证');axes[0].bar_label(bars,padding=3);axes[0].set_xticks(x,instruments);axes[0].legend();axes[0].set_ylabel('实际接入观测数')
    counts=Counter(str(r['band_count']) for r in rows if r['role']=='TRAIN');keys=sorted(counts,key=int)
    bars=axes[1].bar(keys,[counts[k] for k in keys]);axes[1].bar_label(bars,padding=3);axes[1].set_xlabel('每份观测真实可用波段数');axes[1].set_ylabel('训练观测数')
    fig.suptitle('扩量后的真实输入：保留原生网格与有效区；未启动模型训练')
    fig.savefig(OUT/'data_coverage.png',dpi=150);plt.close(fig)
    net=release['network'];text=['# Encoder V2数据扩量交付','',
        '本轮按用户新增授权接入本地真实波段、补取Landsat QA和既有位置原生窗口；没有启动G1模型或SRF解析，没有优化更新，没有恢复旧训练。数据发布与模型训练许可分离。','',
        '| 实际数据角色 | S2 | Landsat8 | Landsat9 |','|---|---:|---:|---:|']
    for role,label in [('TRAIN','训练'),('VAL','验证'),('GUARD','新增隔离')]:text.append('| '+label+' | '+' | '.join(str(release['counts'].get(role+'/'+s,0)) for s in instruments)+' |')
    text+=['','![真实数据覆盖](data_coverage.png)','',
        '左图是最终角色隔离后的观测数；右图是训练成员每份实际波段数。四波段S2是旧缓存中真实10米的蓝、绿、红、近红外，原20米插值通道仍保留在旧数据，未改称10米原生。六波段Landsat是合法真实子集，不因缺coastal波段而排除。完整原生多波段观测与这些部分观测共用同一数据字段。','',
        '旧Landsat缓存逐份核对30米网格及原网格相位，继承已冻结的QA_PIXEL/RADSAT有效区并补取原生aerosol QA。旧S2四条10米通道保留已冻结SCL有效区；准备阶段源缓存SHA和通道选择一致检查通过，但独立定标对照发现旧提供方偏移重复扣除，最终发布采用单独修复版本，不能声称最终反射率与错误旧值不变。负反射率不硬裁剪；源DN先转float32再缩放。QA使用最近邻，反射率数据没有为统一尺寸而插值。Landsat C2反射率定标依据[USGS官方说明](https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products)。','',
        '重要修复：EarthSearch旧COG可能已应用BOA偏移，但资产仍声明-0.1，旧缓存再次扣除导致系统偏低。同次采集Collection1对照中S2A/B/C三个控制均显示0.1偏差，修正后RMSE约1e-8；已应用标记明确者依据该政策修复，标记矛盾者逐观测数值核对后才纳入。原缓存与初始准备结果保留，修复在s2_calibrated_v2/；该提供方可能在生成旧COG时截断过极暗值，不能声称恢复了上游丢失的负值。未能确认的记录单列calibration_pending。提供方问题记录见[官方仓库讨论](https://github.com/Element84/earth-search/discussions/26)及[问题71](https://github.com/Element84/earth-search/issues/71)。','',
        '![S2独立定标控制](s2_calibration_controls.png)','',
        '每组柱对应一个S2A/B/C同次采集控制，每个控制1024个有效红波段像元；纵轴是对数坐标，应看修复前后的系统偏差。三个控制用于核实明确标记的提供方处理政策，不代表逐份独立验证了全部S2；标记矛盾的成员另有逐份参考核对记录。','',
        '对旧时序缓存的训练地理位置，按实际日期前后16天检索Landsat8，在不接触既有留出采集/区域的前提下，最多尝试3个实际场景并按固定QA规则接入。原生30米窗口保留实际CRS、transform、原始大小；这些观测只证明同地理位置，不能作为≤2天、配准合格的跨卫星正配对。','',
        '最终按实际原生支持再合并采集/地理组件，只有从TRAIN限制到VAL/GUARD，没有留出回流。元信息中的source_role及role_adjustments保留变化。F盘已有角色冲突继续隔离；TEST没有读取像元。HLS的[官方分发](https://registry.opendata.aws/nasa-hlss30/)需要Earthdata访问，已存在原生控制只含部分波段且位于保护地域；本轮采用可公开访问的Landsat补全路径，没有反复尝试HLS认证。','',
        f'本轮数据及定标参考实际计量HTTP响应体合计{release["total_new_metered_bytes"]}字节；另为数据队列改为QA优先时的中断读取保守预留32MiB，不冒充已测实际字节。限额从此前10GiB总约定扣除既有记录和SRF获取，数据主队列及参考分别计量。失败与重复读取计入实际读到的响应体；未完成数据任务{len(pending)}条，另有定标待核实{len(unresolved)}条，均不写成已通过。','',
        'members.jsonl、release.json、statistics.json分别冻结真实成员、准入/预算和TRAIN-only统计。expanded_dataset.py读取每波段实际大小和有效区，不补零通道、不强迫共同最细网格；source_local及source_remote保存执行版本。critical_checks和loader_acceptance提供最小正确性证据。','',
        '数据已具备统一读取格式，但SRF响应及历史适用性仍待G1；model_training_authorized=false，不能凭数据扩量启动旧队列或宣称模型收益。旧文件、旧checkpoint和统计未覆盖。']
    (OUT/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    source_names=['expand_training_data.py','expansion_data.py','repair_expansion_s2.py','expanded_dataset.py','finalize_expanded_data.py']
    save(OUT/'delivery_source_hashes.json',{name:digest(ROOT/'observation_model'/name) for name in source_names})
    main_readme=ROOT/'observation_model/README.md'
    content=main_readme.read_text(encoding='utf-8')
    start='<!-- ENCODER_V2_DATA_EXPANSION_START -->';end='<!-- ENCODER_V2_DATA_EXPANSION_END -->'
    train_count=sum(v for k,v in release['counts'].items() if k.startswith('TRAIN/'))
    val_count=sum(v for k,v in release['counts'].items() if k.startswith('VAL/'))
    block=(start+'\n## Encoder V2：数据扩量执行入口\n\n'
        f'按用户追加的数据扩量授权完成[扩量数据发布](outputs/encoder_v2/data_expansion/README.md)：最终训练{train_count}份、验证{val_count}份；实际角色、逐波段网格、定标修复、QA及来源以冻结成员为准。'
        '新读取器保留每波段原生大小和有效区；下载仅限本数据任务，未恢复旧队列。'
        'S2重复偏移在独立版本修复，未确认成员隔离。数据准入通过不等于G1模型获准，优化更新仍为0。\n'+end)
    if start not in content or end not in content:raise ValueError('Stable README expansion entry missing')
    before,rest=content.split(start,1);_,after=rest.split(end,1)
    main_readme.write_text(before+block+after,encoding='utf-8')
    save(OUT/'delivery_hashes.json',{p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name in ['members.jsonl','release.json','statistics.json','README.md','loader_acceptance.json','critical_checks.json']})
    print(json.dumps(dict(status='EXPANDED_DATA_RELEASED_NO_TRAINING',counts=release['counts'],pending=len(pending),received_bytes=net['received_bytes']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
