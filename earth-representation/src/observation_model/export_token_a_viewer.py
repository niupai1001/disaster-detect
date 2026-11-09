"""Honest TVLP import of A's saved points; never invent missing steps or times."""
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.ssl.tvlp_writer import TvlpRun

OUT=Path(__file__).parent/'outputs/token_A'


def main():
    source=OUT/'result.json';r=json.loads(source.read_text());sha=hashlib.sha256(source.read_bytes()).hexdigest();h=r['history']
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    x=[v['step'] for v in h]
    axes[0,0].plot(x,[v['loss'] for v in h],marker='o');axes[0,0].set_yscale('log');axes[0,0].set_ylabel('归一化MSE（对数）');axes[0,0].set_title('训练损失｜原始保存的11个节点')
    axes[0,1].plot(x,[v['gradient'] for v in h],marker='o');axes[0,1].set_yscale('log');axes[0,1].set_ylabel('裁剪前梯度范数');axes[0,1].set_title('梯度记录')
    axes[1,0].plot(x,[v['lr'] for v in h],marker='o');axes[1,0].set_ylim(0,.0012);axes[1,0].set_ylabel('学习率');axes[1,0].set_title('固定学习率1e-3')
    labels=['空间插值','光谱线性回归','模型'];scores=[r['val_controls'][k]['score'] for k in ['spatial','spectral']]+[r['val_model']['score']]
    axes[1,1].bar(labels,scores);axes[1,1].set_ylabel('固定地域验证MSE');axes[1,1].set_title('仅第200步评价一次；不是验证曲线')
    for a in axes.flat:a.grid(axis='y',alpha=.2)
    for a in [axes[0,0],axes[0,1],axes[1,0]]:a.set_xlabel('优化器更新步数（本轮亦为固定子集遍历数）')
    fig.suptitle('实验A｜48训练/32验证｜200步已完成｜原始节点连线，不补造逐步数据');fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=150);plt.close(fig)
    folder=OUT/'tvlp_import/token_A_completed_import'
    if (folder/'run.json').exists():
        previous=json.loads((folder/'run.json').read_text());assert previous['config']['source_sha256']==sha
        print('Existing identical import retained');return
    definitions={'train/loss':{'direction':'min'},'train/gradient_norm':{'direction':'none'},'train/lr':{'direction':'none'},'val/score':{'direction':'min'}}
    run=TvlpRun(folder.parent,project='conditional-observation',name='A单波段局部遮盖｜历史导入（已完成）',run_id=folder.name,seed=0,metric_definitions=definitions,
        config=dict(imported_after_training=True,source=str(source),source_sha256=sha,recorded_training_points=len(h),optimizer_steps=200,train_observations=48,val_observations=32,
            timestamp_meaning='export/import time, not original training wall time; original per-step timestamps were not recorded',
            missing_points='only step1 and multiples20 were saved; no interpolation of metrics; VAL only at final200',
            no_training_started=True,original_settings=r['settings']))
    for row in h:
        run.log('train',dict(loss=row['loss'],gradient_norm=row['gradient'],lr=row['lr']),global_step=row['step'])
    run.log('final_train',dict(score=r['train_model']['score']),global_step=200)
    run.log('val',dict(score=r['val_model']['score'],spatial_interpolation=r['val_controls']['spatial']['score'],spectral_regression=r['val_controls']['spectral']['score']),global_step=200)
    assets=folder/'artifacts';assets.mkdir(exist_ok=True)
    for name in ['training_curves.png','cases_key.png']:
        shutil.copyfile(OUT/name,assets/name)
        event=dict(type='artifact',timestamp=datetime.now(timezone.utc).isoformat(),session_id=run.session_id,path='artifacts/'+name,artifact_type='image',tag=name.removesuffix('.png'),phase='val' if name=='cases_key.png' else 'train',role='comparison',metadata=dict(imported_after_training=True,source_sha256=sha))
        run._events.write(json.dumps(event)+'\n')
    run.finish('completed','Historical results imported; no training executed; timestamps denote import time')
    (OUT/'viewer_import.json').write_text(json.dumps(dict(source_sha256=sha,run=str(folder),points=len(h),training_executed=False),indent=2));print(str(folder))


if __name__=='__main__':main()
