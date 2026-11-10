"""TVLP 2.0 metadata and checkpoint-bound sessions for actual LR training."""
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid
import math
import torch
from .checkpoint import digest
from .tvlp_writer import RunWriter


METRICS = {
    'train_loss': dict(label='训练重建损失', unit='z²', direction='min', role='loss'),
    'val_loss': dict(label='固定验证重建损失（S2/Landsat）', unit='z²', direction='min', role='loss'),
    'learning_rate': dict(label='本步学习率', unit='', direction='none', role='other'),
    'gradient_norm': dict(label='裁剪前梯度范数', unit='L2', direction='none', role='other'),
    'update_seconds': dict(label='更新耗时', unit='s', direction='min', role='other'),
    'spent_updates': dict(label='实际参数更新次数', unit='次', direction='none', role='other'),
}
VARIANT = dict(
    id='lr-native-windows-joint309-compute-optimized',
    name='联合局部读出 · 原生窗口 · 计算优化', kind='modification',
    description='保持309维联合局部/全局读出和A/B1目标；同格网合批、纯几何与输入缓存、精确检索复用，修复学习率与恢复预算。')


def primary_process():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get('RANK', '0')) == 0


def training_runs_root():
    # Frozen runtimes live under outputs; keep their logs at the actual project root.
    source = Path(__file__).resolve()
    for parent in source.parents:
        if (parent/'pyproject.toml').is_file():
            return parent/'training-runs'
    raise ValueError('Cannot locate project training-runs directory')


def epoch_length(rows, config):
    cohort = set(config.get('e2_members', []))
    counts = Counter(
        'Landsat' if r['sensor'] in ('L8', 'L9') else r['sensor']
        for r in rows if r['role'] == 'TRAIN' and len(r['bands']) >= 2 and (not cohort or r['id'] in cohort))
    if not counts: raise ValueError('No TRAIN members for epoch accounting')
    return math.ceil(sum(counts.values())/config['batch_size']) if 'epochs' in config else max(counts.values())


def metadata(config):
    viewer = config.get('viewer', {})
    phase = config['phase'].lower()
    epoch_mode='epochs' in config
    experiment = viewer.get('experiment', dict(
        id=f'encoder-v3-lr-observation-equal-epochs-{phase}' if epoch_mode else f'encoder-v3-lr-native-optical-{phase}',
        name='Encoder V3 · 完整Epoch训练' if epoch_mode else 'Encoder V3 · 真实更新检查' if phase == 'e2' else 'Encoder V3 · 原生光学首轮训练'))
    variant = viewer.get('variant', dict(id='lr-joint309-epoch-observation-mean',name='完整Epoch · 观测等权',kind='modification',
        description='完整TRAIN遍历、批次观测平均梯度和完整VAL观测平均选模；保留309维结构。') if epoch_mode else dict(VARIANT))
    for value in (experiment, variant):
        if any(not isinstance(value.get(k), str) or not value[k].strip() for k in ('id','name')):
            raise ValueError('Viewer experiment and variant require stable id and readable name')
    if variant.get('kind') not in ('baseline','ablation','modification'):
        raise ValueError('Viewer variant kind must be baseline, ablation or modification')
    if variant['kind'] != 'baseline' and not variant.get('description','').strip():
        raise ValueError('Changed Viewer variants require a description')
    return experiment, variant


def training_writer(config, identity, state, run_dir, train_rows, *, resuming):
    if not primary_process(): return nullcontext(None)
    root = training_runs_root().resolve(); run_dir = Path(run_dir).resolve()
    reference = state.get('viewer')
    if reference is not None:
        if not resuming: raise ValueError('Viewer resume requires a loaded checkpoint')
        path = Path(reference['directory']).resolve()
        if not path.is_relative_to(root): raise ValueError('Viewer run is outside project training-runs')
        meta = json.loads((path/'run.json').read_text(encoding='utf-8'))
        if (meta['run_id'] != reference['run_id']
                or meta['config'].get('execution_identity') != digest(identity)
                or meta['config'].get('training_directory') != str(run_dir)):
            raise ValueError('Viewer run does not match the restored checkpoint')
        return RunWriter(path, resume=True)
    experiment, variant = metadata(config)
    epoch_mode='epochs' in config
    metrics={key:dict(value) for key,value in METRICS.items()}
    numbers=['learning_rate','gradient_norm','update_seconds','spent_updates']
    if epoch_mode:
        metrics['train_loss']['label']='训练批次平均损失（观测等权）'
        metrics['val_loss']['label']='完整验证集平均损失（观测等权）'
        metrics.update(batch_size=dict(label='本批观测数',unit='个',direction='none',role='other'),
            observations_seen=dict(label='累计训练观测数',unit='个',direction='none',role='other'),
            epoch_train_loss=dict(label='完整Epoch训练平均损失',unit='z²',direction='min',role='loss'),
            val_s2_loss=dict(label='完整S2验证损失',unit='z²',direction='min',role='loss'),
            val_landsat_loss=dict(label='完整Landsat验证损失',unit='z²',direction='min',role='loss'))
        numbers=['learning_rate','gradient_norm','val_s2_loss','val_landsat_loss']
    directory = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex
    writer = RunWriter(root/directory, experiment=experiment, variant=variant, metrics=metrics,
        charts=['train_loss','val_loss'], primary_metric='val_loss',
        numbers=numbers,
        config=dict(phase=config['phase'], **(dict(epochs=config['epochs'],batch_size=config['batch_size'],
            micro_batch_observations=1,validations_per_epoch=config['validations_per_epoch'],drop_last=False) if epoch_mode else dict(budget_updates=config['steps'])),
            training_directory=str(run_dir), execution_identity=digest(identity),
            training_config_sha256=digest(config), code_identity=identity['code'],
            learning_rate=config['lr'], weight_decay=config['weight_decay'],
            epoch_updates=epoch_length(train_rows, config),
            epoch_definition='完整TRAIN成员每轮随机无放回遍历一次；进度=累计处理观测数/TRAIN成员数' if epoch_mode else '完成更新数/最大来源TRAIN成员数；每次更新从每个来源各取1个观测，E2按绑定成员计算',
            objective='标准化域MSE；通道和观测等权，0.5*A+0.5*B1；不按来源重加权' if epoch_mode else '标准化域MSE；通道/观测/来源等权，0.5*A+0.5*B1',
            validation_scope='全部冻结S2/Landsat VAL观测；与训练同遮盖规则、固定验证随机状态；EnMAP无独立VAL' if epoch_mode else '固定S2/Landsat面板；EnMAP无独立VAL',
            restored_checkpoint_without_previous_viewer=resuming))
    state['viewer'] = dict(directory=str(writer.path), run_id=writer.meta['run_id'])
    return writer
