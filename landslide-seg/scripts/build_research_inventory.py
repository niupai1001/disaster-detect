"""CPU-only documentation inventory. Never imports models or starts training.

Refresh stable aliases, provenance and figures beneath docs/assets/research.
The four maintained Markdown documents contain human-reviewed interpretation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets/research'
DATE = '2026-10-24'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return None if isinstance(obj, float) and not math.isfinite(obj) else obj


def route(config):
    data = config.get('data', {})
    identity = str(data.get('manifest', '')).replace('\\', '/')
    if 'official_single' in identity:
        return 'OFFICIAL_USABLE'
    if 'event_cv_v5' in identity:
        return 'EVENT_CV_V5'
    if 'p13/' in identity:
        return 'EVENT_HOLDOUT_P13'
    if 'v4r1/' in identity:
        return 'EVENT_HOLDOUT_V4R1'
    return 'LEGACY:' + str(data.get('version', 'UNRESOLVED'))


def naming(config):
    m = config.get('model', {})
    typ = m.get('type', 'segformer')
    backbone = str(m.get('backbone', 'b0')).upper()
    probe = m.get('probe') or {}
    if typ == 'use':
        u=m.get('use',{})
        attention=u.get('attention',True);queries=u.get('queries',4);metadata=u.get('metadata',True)
        identity=u.get('identity','physical');hidden=u.get('hidden',32)
        structure=('波段自注意力' if attention else ('扩宽共享逐波段编码' if hidden!=32 else '共享逐波段编码'))
        structure+='＋固定查询汇聚' if queries else '＋带掩膜均值汇聚'
        if metadata:structure+='＋'+('物理描述' if identity=='physical' else '固定随机身份描述')
        mode=m.get('use_downstream','bridge10')
        if mode in ('linear','shallow','fpn','unet'):
            consumer={'linear':'线性1×1卷积头','shallow':'浅层3×3卷积读取头','fpn':'下游多尺度适配＋全新FPN解码器',
                      'unet':'32维直接输入＋全新可训练U-Net（GroupNorm）'}[mode]
            return f'use-{mode}-{attention}-{queries}-{metadata}-{identity}-{hidden}', f'USE｜可变光学波段集合｜{structure}＋{consumer}'
        if mode in ('rgb3','patch32'):
            consumer={'rgb3':'32→3兼容适配器＋原RGB预训练MiT首层','patch32':'全新32输入MiT首层＋后续预训练MiT'}[mode]
            return f'segformer-b0-use-{mode}-{attention}-{queries}-{metadata}-{identity}-{hidden}', f'SegFormer B0｜USE32空间特征｜{structure}＋{consumer}＋原网格细节'
        return f'segformer-b0-use-{attention}-{queries}-{metadata}-{identity}-{hidden}', f'SegFormer B0｜光学波段集合经USE统一入口｜{structure}＋桥接＋原网格细节'
    if typ == 'probe':
        enc = str(probe.get('encoder', 'unknown'))
        label = {'segformer_b0': 'SegFormer B0 RGB', 'segformer_b2': 'SegFormer B2 RGB',
                 'dofa_rgb': 'DOFA RGB', 'dofa_10': 'DOFA十波段'}.get(enc, enc)
        return f'probe-{enc}', f'冻结 {label}＋线性读取头'
    if typ == 'dofa':
        return 'dofa-pyramid-upernet', 'DOFA＋特征金字塔＋UPerNet'
    if typ == 'unet':
        return 'unet-resnet18', 'ResNet18 U-Net'
    temporal = typ == 'temporal_spectral'
    source = '单张十波段直接输入' if m.get('adapter', 'direct10') == 'direct10' else '十波段可学习投影为三通道'
    if temporal:
        tc = config.get('temporal', {})
        mode = tc.get('mode')
        source = ('截至参考日期的观测' if mode == 'history' else
                  f"{tc.get('n_frames', 15)}期" if mode == 'multi' else '单期') + '十波段共享入口'
    bands = m.get('band_order', [])
    if any('slope' in str(b).lower() for b in bands):
        source += '＋坡度'
    if any('dem' == str(b).lower() for b in bands):
        source += '＋DEM'
    stem = temporal or bool(m.get('stem', False)) or typ == 'single'
    refine = m.get('refine', 'none')
    fusion = m.get('fusion', 'sum')
    exchange = m.get('exchange', 'none')
    suffix = 'standard'
    structure = '标准解码预测'
    if stem:
        suffix, structure = 'late-detail', '原始网格细节修正'
    if typ == 'terrain_fusion':
        source = '单张十波段；高程和坡度独立输入'
        suffix, structure = 'late-detail-terrain-score', '影像细节修正＋地形预测分数相加'
    if refine != 'none':
        suffix = 'refine-' + str(refine)
        structure = '粗概率条件精分残差' if refine == 'guided' else '零条件精分残差'
    if typ == 'single':
        suffix = 'single-' + fusion
        structure = '单头特征相加' if fusion == 'sum' else '单头AFF融合'
    if exchange != 'none':
        suffix += '-early-' + exchange
        structure += '＋' + ('早期双向交换' if exchange == 'cross' else '早期分支内残差')
    if temporal:
        mode = config.get('temporal', {}).get('mode', 'unknown')
        suffix += '-temporal-' + str(mode)
        aggregation = config.get('temporal', {}).get('aggregation', 'date')
        structure += '＋' + ('仅参考影像的时间接口' if mode == 'single' else
                            {'mean': '有效帧均值汇聚', 'content': '仅内容时间加权',
                             'date': '内容与观测日期间隔时间加权'}[aggregation])
        suffix += '-' + aggregation
        if m.get('spectral'):
            suffix += '-band-relation'
            structure += '＋光谱关系编码'
    if probe:
        step = probe.get('source_step')
        suffix += '-frozen-' + (str(step) if step is not None else 'encoder')
        structure += f'（冻结第{step}步编码器、重训解码器）' if step is not None else '（冻结编码器、重训解码器）'
    return f'segformer-{backbone.lower()}-{m.get("adapter", "direct10")}-{suffix}', f'SegFormer {backbone}｜{source}｜{structure}'


def metric_rows(run):
    rows = []
    for p in sorted(set((run / 'eval').glob('eval_best*.json')) | set((run / 'eval').glob('eval_dev_*.json'))
                    | set((run / 'eval').glob('eval_last*.json')) | set((run / 'eval').glob('eval_independent_recheck*.json'))):
        d = read(p)
        parts = {key: d[key] for key in ('val', 'test') if isinstance(d.get(key), dict)}
        if not parts:
            parts = {str(d.get('split', p.stem)): d}
        for split, part in parts.items():
            overall = part.get('overall', {})
            pix = overall.get('pixel', {})
            rows.append({'source': relative(p), 'split': split,
                         'n_samples': part.get('n_samples'), 'pixel': pix, 'band_subset': d.get('band_subset'),
                         'event_mean_iou': part.get('event_mean_iou', d.get(split + '_event_mean_iou')),
                         'objects': {k: v for k, v in overall.get('objects', {}).items() if k in ('n_gt_evaluable', 'n_matched_evaluable', 'object_recall_evaluable')},
                         'boundary_f1': overall.get('boundary', {}).get('f1'),
                         'metric_version': d.get('metric_version', overall.get('metric_version')),
                         'manifest_sha256': d.get('manifest_sha256'),
                         'checkpoint_sha256_recorded': d.get('checkpoint_sha256')})
    return rows


def inventory():
    records, excluded = [], []
    paths=list((ROOT / 'outputs/segformer').rglob('resolved_config.json'))
    paths+=list((ROOT / 'outputs/use/s2_10band_v4r1_p20').rglob('resolved_config.json'))
    for directory in ('probes','adapters','unet_control'):
        paths+=list((ROOT / 'outputs/use_independent_frozen'/directory).rglob('resolved_config.json'))
    for p in sorted(paths):
        run = p.parent
        path = relative(run)
        if any(token in path.lower() for token in ('smoke', 'validation_', '/toy/', '_test_temp', '_regression_tmp', '_probe_test_tmp', '_pilot')):
            excluded.append(path)
            continue
        cfg = read(p).get('segformer', {})
        if not cfg:
            raise ValueError(f'Missing segformer config: {p}')
        sp = run / 'train_summary.json'
        summary = read(sp) if sp.exists() else {}
        steps = summary.get('global_step', 0)
        if sp.exists() and steps < 1000 and not summary.get('complete'):
            excluded.append(path)
            continue
        arch_id, name = naming(cfg)
        tags = []
        if cfg.get('model',{}).get('type')=='use':
            tags.append('3/4/6/10波段混合训练' if cfg.get('data',{}).get('mixed_bands') else '完整十波段训练')
            stage=cfg['model'].get('use_stage')
            tags.append('USE随机初始化' if stage=='independent_random' else
                        'USE独立无标签预训练初始化' if stage=='independent_ssl' else
                        'USE外部遮挡波段重建初始化' if cfg['model'].get('use_pretrained') else 'USE随机初始化')
            if cfg['model'].get('freeze_use'):tags.append('USE全部参数与buffer冻结')
        no_bn_consumer=(cfg.get('model',{}).get('type')=='use'
                        and cfg['model'].get('use_downstream') in ('linear','shallow','fpn','unet'))
        if cfg.get('train',{}).get('freeze_bn_running_stats') and not no_bn_consumer:
            tags.append('BN运行统计冻结，仿射参数训练')
        loss = cfg.get('loss', {})
        labels = {'coarse_weight': '基础输出辅助监督权重', 'boundary_weight': '边界加权', 'pos_weight': 'BCE正类权重'}
        for k in labels:
            if loss.get(k) and (k != 'pos_weight' or loss[k] != 1):
                tags.append(f'{labels[k]}={loss[k]}')
        if cfg.get('optim', {}).get('first_conv_lr'):
            tags.append('首层独立学习率=' + str(cfg['optim']['first_conv_lr']))
        if cfg.get('optim', {}).get('encoder_lr') == 6e-6:
            tags.append('预训练encoder低学习率')
        if cfg.get('sampling', {}).get('mode') == 'balanced':
            tags.append('事件／尺度均衡采样')
        terrain = cfg.get('model', {}).get('terrain') or {}
        if terrain:
            start = terrain.get('start')
            if start == 'random':
                tags.append('地形分支不先单独训练，直接随分割任务学习')
            elif start == 'ssl_supervised':
                tags.append('编码器先做地形补全训练；评分头先用滑坡标注训练，未迁移该阶段编码器')
            elif start == 'ssl_scoring_full_v2':
                tags.append('有效区区块占比监督v2；成对迁移评分阶段编码器与预测头')
            elif start == 'ssl':
                tags.append('分割前先遮挡地形、训练补回原高程' + ('6000步' if 'extra' in str(terrain.get('weights')) else '3000步'))
            tags.append('地形融合系数可学习' if terrain.get('alpha_learnable', True) else '地形融合系数固定0.01')
        # Config is preserved verbatim; this label cannot silently change its meaning.
        state = '完成训练' if summary.get('complete') or (sp.exists() and steps >= 1000 and 'complete' not in summary) else '未完成／需核对恢复状态'
        if '_blocked_' in path:
            state = '历史阻碍／权重身份待闭环'
        if 'event_cv_v5_probe/Probe-dofa_10' in path:
            state += '；旧波长错误，不作光谱结论'
        train_cfg = cfg.get('train', {})
        if summary and not summary.get('complete'):
            target = train_cfg.get('max_steps')
            if target and steps < target:
                state = '未完成／未达到目标更新数'
        model_cfg = cfg.get('model', {})
        temporal = model_cfg.get('type') == 'temporal_spectral'
        effective_stem = temporal or bool(model_cfg.get('stem')) or model_cfg.get('type') == 'single'
        if model_cfg.get('type')=='use':
            effective_stem=model_cfg.get('use_downstream','bridge10') in ('bridge10','rgb3','patch32')
        effective_input = 'RGB' if model_cfg.get('type') == 'probe' and model_cfg.get('probe', {}).get('encoder') != 'dofa_10' else 'S2十波段及配置声明的额外通道'
        if model_cfg.get('type')=='use':effective_input='可变光学波段集合；实际训练组合及评价子集见配置与评价文件'
        treatment = '；'.join(tags) or '原训练处理，详见配置'
        data_context = str(cfg.get('data', {}).get('version', '未声明数据版本'))
        manifest = str(cfg.get('data', {}).get('manifest', '')).replace('\\', '/')
        fold = re.search(r'event_cv_v5/(F[1-4])/', manifest)
        if fold:
            pd = read(ROOT/'manifests/event_cv_v5'/fold.group(1)/'protocol.json')
            data_context = '外部留出：' + '、'.join(pd['design']['outer'])
        if '/p13/few/' in manifest:
            data_context = '三个训练事件，同1251块'
        if '/p13/all/' in manifest:
            data_context = '九个训练事件，同1251块'
        run_name = f"{name}｜{route(cfg)}｜{data_context}｜TRAIN={summary.get('n_train', '?')}｜seed={summary.get('seed', run.name)}｜{treatment}"
        r = {'run_id': hashlib.sha256(path.encode()).hexdigest()[:16], 'architecture_id': arch_id,
             'display_name': name, 'run_display_name': run_name,
             'treatment': treatment, 'protocol': route(cfg),
             'data_context': data_context,
             'data_version_recorded': cfg.get('data', {}).get('version'),
             'seed': summary.get('seed', cfg.get('seed', run.name)), 'status': state,
             'run_dir': path, 'legacy_experiment': cfg.get('experiment', run.parent.name),
             'config_path': relative(p), 'config_sha256': sha(p),
             'manifest': cfg.get('data', {}).get('manifest'), 'stats': cfg.get('data', {}).get('stats'),
             'input': {k: cfg.get('model', {}).get(k) for k in ('adapter', 'in_chans', 'band_order')},
             'effective_structure': {'stem': effective_stem, 'input_seen_by_encoder': effective_input,
                                     'source': '配置并结合temporal_spectral／probe类的实际构造语义'},
             'model_config': cfg.get('model'), 'temporal': cfg.get('temporal'),
             'loss': loss, 'optim': cfg.get('optim'), 'train': cfg.get('train'), 'sampling': cfg.get('sampling'),
             'summary_source': relative(sp) if sp.exists() else None,
             'global_step': steps if sp.exists() else None,
             'best_selection_value': summary.get('best_val_iou'),
             'selection_metric': summary.get('selection_metric', '历史VAL IoU，见原run'),
             'n_train': summary.get('n_train'), 'n_val': summary.get('n_val'),
             'params': summary.get('params'), 'seconds': summary.get('seconds'),
             'best_exists': (run / 'best.pt').exists(), 'last_exists': (run / 'last.pt').exists(),
             'metrics_jsonl': relative(run / 'metrics.jsonl') if (run / 'metrics.jsonl').exists() else None,
             'evaluations': metric_rows(run)}
        records.append(r)
    records.extend(independent_pretraining_inventory())
    return records, excluded


def independent_pretraining_inventory():
    """Independent SSL has no task labels or SegFormer; keep its metric family explicit."""
    records=[]
    for p in sorted((ROOT/'outputs/use_independent_frozen/pretraining').glob('*/resolved_config.json')):
        cfg=read(p);run=p.parent;path=relative(run)
        summary_path=run/'summary.json';summary=read(summary_path) if summary_path.exists() else {}
        acceptance_path=run/'acceptance.json';accepted=read(acceptance_path).get('passed') is True if acceptance_path.exists() else False
        u=cfg['use_config'];attention=u.get('attention',True);queries=u.get('queries',4);metadata=u.get('metadata',True)
        identity=u.get('identity','physical')
        structure=('波段注意力' if attention else '共享逐波段卷积')+('＋查询汇聚' if queries else '＋掩膜均值汇聚')
        if metadata:structure+='＋'+('物理描述' if identity=='physical' else '固定随机身份')
        name='USE｜可变光学波段集合｜'+structure+'；临时重建读出'
        sensors='+'.join(cfg['train_ids'])
        treatment='独立无标签遮挡波段重建'+('＋空间遮挡重建' if cfg.get('spatial_objective') else '')
        n_train=sum(len(ids) for ids in cfg['train_ids'].values());n_val=sum(len(ids) for ids in cfg['validation_ids'].values())
        status='完成训练并通过正式产物验收' if accepted else '完成训练；正式产物验收待核对' if summary.get('complete') else '未完成／进程与冻结配置状态待核对'
        records.append(dict(run_id=hashlib.sha256(path.encode()).hexdigest()[:16],
            architecture_id=f'use-ssl-{attention}-{queries}-{metadata}-{identity}',display_name=name,
            run_display_name=f'{name}｜无标签冻结地理成员｜{sensors}｜seed={cfg["seed"]}｜{treatment}',
            treatment=treatment,protocol='USE_INDEPENDENT_SSL',data_context='无标签预训练地理成员；不是滑坡DEV',
            seed=cfg['seed'],status=status,run_dir=path,legacy_experiment='use_independent_pretraining_'+cfg['variant'],
            config_path=relative(p),config_sha256=sha(p),model_config=dict(type='use_ssl',use=u),
            effective_structure=dict(stem=False,input_seen_by_encoder='真实无标签光学波段集合',source='独立预训练实际配置'),
            summary_source=relative(summary_path) if summary_path.exists() else None,
            global_step=summary.get('updates'),best_selection_value=summary.get('best_score'),
            selection_metric='equal_sensor_unlabeled_standardized_mbm_mae',n_train=n_train,n_val=n_val,
            best_exists=(run/'best.pt').exists(),last_exists=(run/'last.pt').exists(),evaluations=[],
            initialization_sha256=cfg.get('initialization_sha256'),members_sha256=cfg['members_sha256'],
            acceptance_source=relative(acceptance_path) if acceptance_path.exists() else None))
    return records


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib import font_manager
    fp = Path('C:/Windows/Fonts/msyh.ttc')
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(fp)).get_name()
    plt.rcParams['axes.unicode_minus'] = False

    def canvas(title, height=8):
        fig, ax = plt.subplots(figsize=(16, height))
        ax.set(xlim=(0, 16), ylim=(0, height)); ax.axis('off')
        fig.suptitle(title, fontsize=17, y=.98)
        return fig, ax

    def box(ax, x, y, label, w=2, h=.65, color='#e5eefb'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.05', fc=color, ec='#667085', lw=1.1))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=10)

    def arrow(ax, start, end, label=None, rad=0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12, color='#475467', connectionstyle=f'arc3,rad={rad}'))
        if label:
            ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.10, label, ha='center', fontsize=8, color='#344054')

    def save(fig, stem):
        for ext in ('png', 'svg'):
            fig.savefig(OUT / f'{stem}.{ext}', bbox_inches='tight', dpi=150)
        plt.close(fig)

    fig, ax = canvas('两种数据协议：开发与官方比较各回答一个问题', 5.5)
    box(ax, .3, 2.4, 'Sen12Landslides\n固定源数据', 2.3, .9)
    box(ax, 4, 3.6, '项目协议\n整事件留出', 2.8, .9, '#d8f1e4')
    box(ax, 8.1, 3.6, '默认开发与消融\n逐事件比较', 2.7, .9, '#d8f1e4')
    box(ax, 12, 3.6, '有明显进步？\n检查AP、误报、成本', 3.3, .9, '#fff0cb')
    box(ax, 4, .9, '官方split\n官方完整集 ≠ 本地可用子集', 3.2, 1)
    box(ax, 8.1, .9, '冻结候选后训练\n官方划分比较', 2.7, 1)
    box(ax, 12, .9, '对齐成员、模态、指标\n才可比较官方基线', 3.3, 1)
    arrow(ax, (2.6, 3.1), (4, 4)); arrow(ax, (2.6, 2.6), (4, 1.4))
    arrow(ax, (6.8, 4), (8.1, 4)); arrow(ax, (10.8, 4), (12, 4))
    arrow(ax, (13.6, 3.6), (9.5, 1.9), '达到阶段收益门槛', -.05)
    arrow(ax, (7.2, 1.4), (8.1, 1.4)); arrow(ax, (10.8, 1.4), (12, 1.4))
    ax.text(.3, .15, '历史官方实验保留；当前规则只约束后续新增训练。两类分数不混排。', fontsize=11)
    save(fig, 'protocol_routes')

    fig, ax = canvas('输入如何改变：数据扩量、加地形、加时相是三种不同变化', 7)
    rows=[('样本／事件扩量', ['十波段单期影像','同一十波段入口','同一SegFormer','输入层不改变']),
          ('直接输入适配', ['十波段单期影像','扩展首层卷积','官方MiT编码器','官方解码器']),
          ('可学习投影', ['十波段单期影像','可训练光谱投影','原RGB入口＋MiT','官方解码器']),
          ('增加地形输入', ['影像＋DEM／坡度','扩展输入与统计','MiT＋细节分支','相同预测路径']),
          ('增加时间观测', ['多帧十波段＋日期','逐帧共享入口','共享MiT→时间聚合','官方解码器'])]
    for i,(lab, labels) in enumerate(rows):
        y=5.7-i*1.15; ax.text(.1, y+.35, lab, fontsize=11, weight='bold')
        for j,label in enumerate(labels):
            x=3+j*3.1; box(ax,x,y,label,w=2.65)
            if j: arrow(ax,(x-.45,y+.32),(x-.05,y+.32))
    save(fig, 'input_evolution')

    fig, ax = canvas('从标准SegFormer到原网格细节残差：基础结构', 6.5)
    box(ax,.2,3.3,'单期S2影像',2,.8)
    box(ax,3,3.3,'输入适配',1.8,.8)
    box(ax,5.6,3.3,'官方MiT\n分层编码器',2,.8)
    box(ax,8.4,3.3,'官方解码器\n多尺度特征融合',2.3,.8)
    box(ax,11.5,4.5,'基础分类与上采样',2.1,.75)
    box(ax,3,1.1,'原网格细节Stem\n不下采样',2.2,.9,'#d8f1e4')
    box(ax,8.4,1.1,'细节残差头\n融合解码特征与细节',2.6,.9,'#d8f1e4')
    box(ax,14,3.3,'相加\n输出Mask',1.6,.8,'#fff0cb')
    for a,b in [((2.2,3.7),(3,3.7)),((4.8,3.7),(5.6,3.7)),((7.6,3.7),(8.4,3.7)),((10.7,3.8),(11.5,4.8)),((13.6,4.8),(14.5,4.1)),((1.2,3.3),(3.2,2)),((5.2,1.55),(8.4,1.55)),((9.5,3.3),(9.5,2)),((11,1.55),(14.5,3.3))]: arrow(ax,a,b)
    ax.text(.3,.2,'关闭细节支路：基础分类结果直接输出；开启后：最终logits＝基础logits＋细节修正。\n原网格Stem是旁路，不是输入投影；两者可以同时存在。',fontsize=11)
    save(fig,'architecture_baseline_detail')

    fig, ax = canvas('三个后续结构的模块连接关系', 9)
    for y, title, center in [(7.2,'条件精分残差','解码特征＋细节＋粗概率\n精分修正，再加基础logits'),(4.55,'单头融合','解码特征与细节对齐\n相加或AFF→一个分类头'),(1.9,'早期交换','MiT阶段3–4 → 官方解码器\n保留末端细节残差')]:
        ax.text(.2,y+.95,title,fontsize=13,weight='bold')
        box(ax,.3,y,'影像',1.6,.75)
        box(ax,3.1,y,'MiT阶段1–2\n阶段2与细节双向交换' if title=='早期交换' else 'MiT＋官方解码特征',3,.75)
        box(ax,3.1,y-1,'原网格细节Stem',3,.65,'#d8f1e4')
        box(ax,8,y-.35,center,4,.95,'#fff0cb');box(ax,13.5,y,'Mask',1.8,.75)
        arrow(ax,(1.9,y+.35),(3.1,y+.35));arrow(ax,(1.2,y),(3.1,y-.65))
        arrow(ax,(6.1,y+.35),(8,y+.15));arrow(ax,(6.1,y-.65),(8,y-.1));arrow(ax,(12,y+.15),(13.5,y+.35))
        if title=='早期交换':
            arrow(ax,(4.1,y),(4.1,y-.35));arrow(ax,(5.1,y-.35),(5.1,y))
    save(fig,'architecture_variants')

    fig, ax = canvas('最新模型接口：光谱关系与时间聚合独立开关',6.6)
    for x,y,l,w,col in [(.2,3.8,'单张／多张影像\n观测日期',2.1,'#e5eefb'),(3,3.8,'可选光谱\n关系编码',2,'#fff0cb'),(5.8,3.8,'逐帧共享\n官方MiT',2,'#e5eefb'),(8.6,3.8,'观测日期间隔与内容\n特征加权',2.4,'#fff0cb'),(11.8,3.8,'官方解码器',2,'#e5eefb'),(3,1.2,'取参考影像',2,'#d8f1e4'),(5.8,1.2,'原网格细节Stem',2.3,'#d8f1e4'),(11.8,1.2,'末端细节残差\n输出目标Mask',3.5,'#d8f1e4')]:box(ax,x,y,l,w,.9,col)
    for a,b in [((2.3,4.25),(3,4.25)),((5,4.25),(5.8,4.25)),((7.8,4.25),(8.6,4.25)),((11,4.25),(11.8,4.25)),((4,3.8),(4,2.1)),((5,1.65),(5.8,1.65)),((8.1,1.65),(11.8,1.65)),((12.8,3.8),(12.8,2.1))]:arrow(ax,a,b)
    ax.text(.3,.2,'单期时聚合退化为恒等；多期共享同一空间编码器，不将全部帧堆成输入通道。\n此图说明已实现接口；多期和光谱组尚未全部完成，不能由图推断有效性。',fontsize=11)
    save(fig,'architecture_temporal_spectral')

    # Read actual run metadata for result plots; do not maintain a second numeric table here.
    run_records = read(OUT/'experiment_registry.json')['runs']

    def record(fragment):
        found = [r for r in run_records if fragment in r['run_dir']]
        if len(found) != 1:
            raise ValueError(f'Expected one run for {fragment}: {len(found)}')
        return found[0]

    def test_iou(fragment):
        r = record(fragment)
        vals = [e['pixel']['iou'] for e in r['evaluations'] if 'test' in e['split'] and e['pixel'].get('iou') is not None]
        if len(vals) != 1:
            raise ValueError(f'Expected one best TEST eval for {fragment}')
        return vals[0]

    pools = [record(f's2_10band_v4r1_p20/segformer_b0_direct10_detailres_n{p:03}_u3600_s1234/') for p in (25,50,75,100)]
    few = record('p13_events/segformer_b0_latedetail_fewevents_')
    many = record('p13_events/segformer_b0_latedetail_allevents_')
    fig, axes=plt.subplots(1,2,figsize=(13,4.5))
    axes[0].plot([r['n_train'] for r in pools],[r['best_selection_value'] for r in pools],marker='o')
    axes[0].set(xlabel='训练图块数（均为同九事件）',ylabel='DEV事件平均IoU',title='同事件增加图块：后续非单调',ylim=(.20,.36))
    axes[1].bar(['三个事件，同1251块','九个事件，同1251块'],[few['best_selection_value'],many['best_selection_value']],color=['#98a2b3','#199d76'])
    axes[1].set(ylabel='DEV事件平均IoU',title='固定N改变事件组成',ylim=(0,.36))
    fig.tight_layout();save(fig,'event_data_evidence')
    fig,ax=plt.subplots(figsize=(10,4.5))
    ax.barh(['Italy留出','中国／北海道／吉尔吉斯留出','广岛／新西兰等留出','非洲／加勒比等留出'],
            [test_iou(f'event_cv_v5/SegFormer-LateDetail-EventCVV5-F{k}/') for k in range(1,5)],color='#3984ba')
    ax.set(xlabel='各折外部留出 pooled IoU',xlim=(0,.35),title='SegFormer＋细节残差：四折跨事件参照');ax.invert_yaxis()
    fig.tight_layout();save(fig,'event_cv_baseline')
    fig,ax=plt.subplots(figsize=(10,4.5))
    official_runs = ['official_single/SegFormer-LateDetail-OfficialSplit-SingleDate/',
                     'official_single_geo/SegFormer-LateDetail-OfficialSingle-DEM/',
                     'official_single_geo/SegFormer-LateDetail-OfficialSingle-Slope/',
                     'official_single/SegFormer-LateDetail-OfficialSingle-PW5/',
                     'official_unet/UNet-ResNet18-OfficialSingle-Baseline/']
    ax.barh(['SegFormer十波段','SegFormer＋DEM','SegFormer＋坡度','SegFormer正类加权','U-Net十波段'],
            [test_iou(s) for s in official_runs],color=['#3984ba']*4+['#98a2b3'])
    ax.set(xlabel='本地官方可用子集 TEST pooled IoU',title='历史官方划分结果：集合与官方完整TEST不同',xlim=(0,.5));ax.invert_yaxis()
    fig.tight_layout();save(fig,'official_history')
    fig,ax=plt.subplots(figsize=(10,4.3))
    single_eval=read(ROOT/'outputs/utae_v6/utae_single_realdates/1234/eval_summary.json')['best']['test']['overall']['pixel']['iou']
    multi_eval=read(ROOT/'outputs/utae_v6/utae_multi_realdates/1234/eval_summary.json')['best']['test']['overall']['pixel']['iou']
    ax.barh(['旧日期接口：单期＋DEM','旧日期接口：15期＋DEM','观测日期编码：单期＋DEM','观测日期编码：15期＋DEM'],[.3324,.4770,single_eval,multi_eval],color=['#98a2b3','#98a2b3','#3984ba','#3984ba'])
    ax.set(xlabel='同945块TEST pooled IoU',xlim=(0,.6),title='U-TAE时间诊断：旧接口与观测日期编码接口');ax.invert_yaxis()
    fig.tight_layout();save(fig,'temporal_reference_status')


def check():
    d=read(OUT/'experiment_registry.json')
    ids=set()
    for r in d['runs']:
        assert r['run_id'] not in ids; ids.add(r['run_id'])
        p=ROOT/r['config_path'];assert p.exists() and sha(p)==r['config_sha256']
        assert r['display_name'] and r['protocol']
        if 'temporal_spectral' == r['model_config'].get('type'):
            assert '细节' in r['display_name'], 'Temporal model display must include its actual detail branch'
            assert r['effective_structure']['stem']
    documents = ['RESEARCH_INDEX.md','DATA_OFFICIAL.md','DATA_EVENT_PROTOCOL.md','MODEL_EVOLUTION.md']
    checked = 0
    for name in documents:
        p = ROOT/'docs'/name
        if not p.exists():
            continue
        body = p.read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',body):
            if target.startswith(('http:', 'https:', '#')):
                continue
            local = target.split('#')[0].strip('<>')
            assert (p.parent/local).exists(), (name,target)
            checked += 1
        assert body.count('![') >= 1, f'Missing figure: {name}'
    print(f'PASS: {checked} relative document links and image targets')
    (OUT/'verification.json').write_text(json.dumps({'as_of':DATE, 'run_identities_checked':len(ids),
        'relative_links_checked':checked,'config_hashes_checked':len(ids),
        'training_or_inference_executed':False, 'checkpoint_hashes_recomputed':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"PASS: {len(ids)} run identities, aliases and config hashes; no training/inference executed")


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    parser.add_argument('--no-figures', action='store_true', help='只刷新机器索引与别名，保留现有图')
    args=parser.parse_args()
    if args.check:
        check();return
    OUT.mkdir(parents=True,exist_ok=True)
    runs,excluded=inventory()
    policy=read(OUT/'research_policy.json')
    result={'as_of':DATE,'policy':'原固定事件留出基线默认开发；不默认四折，明显进步后才新增官方比较',
            'default_protocol':policy['default_protocol'],
            'scope':'磁盘现有SegFormer、独立USE预训练及冻结消费者resolved_config；U-TAE由对应协议文档及阶段证据另列',
            'evidence':'CPU读取配置/摘要/已有eval；不重新推理，不重哈希大型checkpoint',
            'runs':runs,'excluded_engineering_runs':excluded}
    (OUT/'experiment_registry.json').write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (OUT/'experiment_aliases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['run_id','display_name','run_display_name','treatment','architecture_id','protocol','data_context','seed','status','run_dir','legacy_experiment']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in runs)
    if not args.no_figures:
        figures()
    check()
    print(json.dumps({'runs':len(runs),'excluded_engineering_runs':len(excluded)},ensure_ascii=False))


if __name__=='__main__':
    main()
