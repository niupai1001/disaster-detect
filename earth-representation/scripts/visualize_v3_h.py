"""Read-only native reconstruction atlas for a frozen V3-H best checkpoint."""
import ssl
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING','1')
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--records',type=Path,help='Reuse full-VAL records already reproduced from this exact checkpoint')
    args=parser.parse_args();release=args.release.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(release/'runtime/src'))
    from encoder_v3.hierarchical.train import load_data,identity,build_model,batch_loss,sample_from_dataset,EpochPlan,write
    from encoder_v3.latent_response.checkpoint import parameter_sha
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    run=release/'formal_run';state=json.loads((run/'status.json').read_text())
    assert state['status']=='STOPPED','Observe safe stop before visualization'
    config=json.loads((run/'config.json').read_text());torch.cuda.set_device(config['device'])
    source_hashes={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ('best.pt','last.pt','updates.jsonl')}
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    datasets,resolver=load_data();assert identity(config,datasets,resolver)==checkpoint['identity']
    plan=EpochPlan(datasets['TRAIN'].rows,config['seed'],config['batch_size'])
    best_epoch=plan.progress(checkpoint['runner']['cursor'])
    model=build_model(config).to(config['device']);model.load_state_dict(checkpoint['model']);model.eval()
    before=parameter_sha(model)
    if args.records:
        previous=json.loads((args.records.parent/'report.json').read_text())
        assert previous['status']=='PASS' and previous['checkpoint_sha256']==source_hashes['best.pt'] and previous['identity']==checkpoint['identity']
        records=json.loads(args.records.read_text())
        assert len(records)==len(datasets['VAL'].rows) and {r['id'] for r in records}=={r['id'] for r in datasets['VAL'].rows}
        metric=sum(r['loss'] for r in records)/len(records)
    else:
        with torch.no_grad():metric,records=batch_loss(model,datasets['VAL'],datasets['VAL'].rows,resolver,dict(config,seed=config['validation_seed']),epoch=0)
    expected=checkpoint['runner']['validation']['metric'];assert abs(metric-expected)<1e-6,(metric,expected)
    write(out/'val_members.json',records)
    report={'status':'RUNNING','checkpoint':str(run/'best.pt'),'checkpoint_sha256':source_hashes['best.pt'],
        'identity':checkpoint['identity'],'paused_cursor':state['cursor'],'paused_epoch':state['epoch'],
        'best_cursor':checkpoint['runner']['cursor'],'best_epoch':best_epoch,'full_val_loss':metric,'full_val_members':len(records),
        'optimizer_updates':0,'selection':'S2/L8: within-source weighted A/B1 full-VAL loss P25/P50/P90. L9/EnMAP: TRAIN native-pixel median, diagnostic only. S2 extreme: worst source VAL loss.',
        'full_val_records_reused_from':str(args.records.resolve()) if args.records else None,
        'display':'Truth and prediction use identical fixed physical-reflectance ranges. RGB0..0.3, singleband0..0.5, gamma2.2; hidden-only absolute error0..0.15. Extreme scene uses RGB0..1 and gray0..1. No independent image stretch. Visible truth in composite is not model prediction.',
        'cases':[],'spectra':[]}
    write(out/'report.json',report);print('FULL_VAL_REPRODUCED',metric,flush=True)
    lookup={role:{r['id']:i for i,r in enumerate(ds.rows)} for role,ds in datasets.items()}
    examples={};spectrum_ids={};figures=[]
    canonical=lambda name:re.sub(r'^B0+(\d)',r'B\1',name.removeprefix('SR_'))

    def stats(ds,row,name):
        bank=ds.datasets[row['sensor']=='EnMAP'] if hasattr(ds,'datasets') else ds
        value=bank.statistics['bands'][row['instrument']+':'+name]
        return value['mean'],value['std']

    def arrays(ds,row,sample,detail,indices):
        truth=[];prediction=[];valid=[];hidden=[]
        for i in indices:
            c=sample.channels[i];mu,std=stats(ds,row,c.name)
            truth.append(c.values.cpu().numpy()*std+mu);prediction.append(detail['predictions'][i].cpu().numpy()*std+mu)
            valid.append(c.valid.cpu().numpy());hidden.append(detail['target_masks'][i].cpu().numpy())
        return tuple(np.stack(v,-1) for v in (truth,prediction,valid,hidden))

    def panels(ds,row,sample,detail,kind,extreme=False):
        if kind=='A':
            desired=('B4','B3','B2');names={canonical(c.name):i for i,c in enumerate(sample.channels)}
            if row['sensor']=='EnMAP':
                candidates=[i for i,c in enumerate(sample.channels) if bool(c.valid.any())]
                indices=[min(candidates,key=lambda i:abs(sample.channels[i].physical.center_um-w)) for w in (.665,.56,.49)]
            else:indices=[names[name] for name in desired]
            assert len({sample.channels[i].grid for i in indices})==1
        else:
            candidates=[i for i,mask in enumerate(detail['target_masks']) if bool(mask.any())]
            indices=[min(candidates,key=lambda i:abs(sample.channels[i].physical.center_um-.85))]
        truth,prediction,valid,mask=arrays(ds,row,sample,detail,indices)
        good=valid.all(-1);hidden=mask.all(-1)&good;assert hidden.any()
        high=1. if extreme else (.3 if kind=='A' else .5)
        def display(values):
            image=np.clip(values/high,0,1)**(1/2.2)
            return image if image.shape[-1]==3 else np.repeat(image,3,-1)
        original=display(truth);predicted=display(prediction);original[~good]=.65
        visible=original.copy();visible[hidden]=.65
        pure=predicted.copy();pure[~hidden]=.65
        composite=original.copy();composite[hidden]=predicted[hidden]
        error=np.where(hidden,np.abs(prediction-truth).mean(-1),np.nan)
        return (original,visible,pure,composite,error),{'display_bands':[sample.channels[i].name for i in indices],
            'reflectance_display_upper':high,'hidden_display_pixels':int(hidden.sum()),
            'display_band_physical_mae':float(np.abs(prediction-truth)[hidden].mean()),
            'display_target_fraction_above_upper':float((truth[hidden]>high).mean())}

    titles=('原图','可见输入','仅隐藏区的模型预测','可见原图＋隐藏区预测','隐藏区绝对误差')
    def draw(axs,images,kind):
        for k,(ax,data) in enumerate(zip(axs,images)):
            if k==4:
                im=ax.imshow(data,cmap='magma',vmin=0,vmax=.15,interpolation='nearest')
                ax.figure.colorbar(im,ax=ax,fraction=.046,pad=.025,extend='max')
            else:ax.imshow(data,interpolation='nearest')
            ax.set_title('该目标波段输入（隐藏）' if kind=='B1' and k==1 else titles[k],fontsize=10);ax.axis('off')

    def evaluate(role,row,kind,label,record=None,extreme=False):
        ds=datasets[role];sample=sample_from_dataset(ds,lookup[role][row['id']],resolver,config['device'])
        with torch.no_grad():loss,details=model([sample],kind=kind,seed=config['validation_seed'],epoch=0,return_predictions=True)
        detail=details[0];value=float(loss[0]);assert np.isfinite(value)
        if record is not None:assert abs(value-record['tasks'][kind]['mae'])<1e-5
        baseline=[]
        for c,mask in zip(sample.channels,detail['target_masks']):
            if bool(mask.any()):baseline.append(float(c.values[mask].square().mean()))
        images,info=panels(ds,row,sample,detail,kind,extreme)
        report['cases'].append({'id':row['id'],'role':role,'source':row['sensor'],'selection':label,'task':kind,
            'loss':value,'same_target_train_mean_baseline_mse':float(np.mean(baseline)),**info})
        write(out/'report.json',report)
        return images

    for sensor in ('S2','L8'):
        ranked=sorted([r for r in records if r['sensor']==sensor],key=lambda r:(r['loss'],r['id']))
        chosen=[ranked[round((len(ranked)-1)*q)] for q in (.25,.5,.9)]
        spectrum_ids[sensor]=('VAL',chosen[1]['id'])
        for kind in ('A','B1'):
            fig,axes=plt.subplots(3,5,figsize=(17,10),squeeze=False)
            for j,(record,label) in enumerate(zip(chosen,('P25 较低误差','P50 典型','P90 较高误差'))):
                row=datasets['VAL'].rows[lookup['VAL'][record['id']]];images=evaluate('VAL',row,kind,label,record)
                draw(axes[j],images,kind);case=report['cases'][-1]
                axes[j,0].text(0,-.07,label+' / '+','.join(case['display_bands']),transform=axes[j,0].transAxes,fontsize=9)
                axes[j,2].text(0,-.07,f'任务MSE={case["loss"]:.4f} z²',transform=axes[j,2].transAxes,fontsize=9)
                if j==1:examples[(sensor,kind)]=images
            fig.suptitle(f'{sensor} 独立VAL / {kind}重建 / 最佳检查点：第{best_epoch:g}轮',fontsize=15)
            clipped=[c['display_target_fraction_above_upper'] for c in report['cases'] if c['source']==sensor and c['task']==kind and c['role']=='VAL']
            fig.text(.5,.019,'统一物理反射率显示，灰色为隐藏或无效；B1仅展示一个隐藏波段，其他可见波段未画出。',ha='center',fontsize=9)
            fig.text(.5,.003,'P25/P50/P90目标通道值超过显示上限：'+' / '.join(f'{v:.1%}' for v in clipped)+'；仅显示截断，误差和指标使用未截断值。',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,.95));filename=f'{sensor}_{kind}.png';fig.savefig(out/filename,dpi=150);plt.close(fig)
            figures.append((f'{sensor} VAL {kind}：误差分位样本',filename));print('RENDERED',filename,flush=True)

    for kind in ('A','B1'):
        fig,axes=plt.subplots(2,5,figsize=(17,7),squeeze=False)
        for j,sensor in enumerate(('S2','L8')):
            draw(axes[j],examples[(sensor,kind)],kind)
            axes[j,0].text(0,-.08,sensor+' 独立VAL典型样本（P50）',transform=axes[j,0].transAxes,fontsize=10)
        fig.suptitle(('A：空间遮盖重建' if kind=='A' else 'B1：整波段遮盖重建')+f' / 最佳权重，第{best_epoch:g}轮',fontsize=15)
        fig.text(.5,.025,'固定显示范围：RGB0—0.3，单波段0—0.5，γ=2.2；误差色标0—0.15反射率。拼合图的可见部分为原图。',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.07,1,.94));filename=f'overview_{kind}.png';fig.savefig(out/filename,dpi=160);plt.close(fig)
        figures.insert(0,(f'{kind}：独立VAL典型样本总览',filename))

    for sensor in ('L9','EnMAP'):
        candidates=sorted([r for r in datasets['TRAIN'].rows if r['sensor']==sensor],key=lambda r:(sum(np.prod(b['shape']) for b in r['bands']),r['id']))
        row=candidates[len(candidates)//2];spectrum_ids[sensor]=('TRAIN',row['id'])
        fig,axes=plt.subplots(2,5,figsize=(17,7),squeeze=False)
        for j,kind in enumerate(('A','B1')):
            images=evaluate('TRAIN',row,kind,'原生像元数量中位观测（非独立验证）');draw(axes[j],images,kind)
            axes[j,0].text(0,-.08,f'{kind} / TRAIN诊断',transform=axes[j,0].transAxes,fontsize=10)
        fig.suptitle(f'{sensor} TRAIN重建诊断 / 当前无该来源独立VAL / 最佳权重第{best_epoch:g}轮',fontsize=14)
        fig.tight_layout(rect=(0,.05,1,.94));filename=f'{sensor}_TRAIN.png';fig.savefig(out/filename,dpi=150);plt.close(fig)
        figures.append((f'{sensor}：训练样本诊断，不代表泛化',filename));print('RENDERED',filename,flush=True)

    worst=max([r for r in records if r['sensor']=='S2'],key=lambda r:r['loss']);row=datasets['VAL'].rows[lookup['VAL'][worst['id']]]
    fig,axes=plt.subplots(2,5,figsize=(17,7),squeeze=False)
    for j,kind in enumerate(('A','B1')):
        images=evaluate('VAL',row,kind,'S2最高综合误差观测',worst,extreme=True);draw(axes[j],images,kind)
        axes[j,0].text(0,-.08,f'{kind} / 统一反射率范围0—1',transform=axes[j,0].transAxes,fontsize=10)
    fig.suptitle('高影响S2验证观测：保留失败与极端样本 / '+worst['id'],fontsize=13)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(out/'extreme_S2_VAL.png',dpi=150);plt.close(fig)
    figures.append(('高影响S2验证观测：单独显示0—1范围', 'extreme_S2_VAL.png'))

    fig,axes=plt.subplots(4,1,figsize=(13,13))
    for ax,sensor in zip(axes,('S2','L8','L9','EnMAP')):
        role,identifier=spectrum_ids[sensor];ds=datasets[role];row=ds.rows[lookup[role][identifier]]
        sample=sample_from_dataset(ds,lookup[role][identifier],resolver,config['device'])
        with torch.no_grad():_,details=model([sample],kind='B1',seed=config['validation_seed'],epoch=0,return_predictions=True)
        groups={}
        for i,c in enumerate(sample.channels):groups.setdefault(c.grid,[]).append(i)
        indices=max(groups.values(),key=len);shape=sample.channels[indices[0]].values.shape
        coverage=np.stack([sample.channels[i].valid.cpu().numpy() for i in indices]).sum(0)
        candidates=np.argwhere(coverage>=max(1,.9*coverage.max()));center=(np.array(shape)-1)/2
        y,x=candidates[((candidates-center)**2).sum(-1).argmin()];waves=[];truth=[];baseline=[];prediction=[];hidden=[]
        for i in indices:
            c=sample.channels[i]
            if not bool(c.valid[y,x]):continue
            mu,std=stats(ds,row,c.name);is_hidden=bool(details[0]['target_masks'][i][y,x])
            waves.append(c.physical.center_um*1000);truth.append(float(c.values[y,x])*std+mu);baseline.append(mu)
            prediction.append(float(details[0]['predictions'][i][y,x])*std+mu if is_hidden else np.nan);hidden.append(is_hidden)
        order=np.argsort(waves);waves=np.array(waves)[order];truth=np.array(truth)[order];baseline=np.array(baseline)[order];prediction=np.array(prediction)[order];hidden=np.array(hidden)[order];assert hidden.any()
        ax.plot(waves,truth,'o-',color='#111827',ms=3,label='真实光谱');ax.plot(waves,baseline,'--',color='#9ca3af',label='TRAIN通道均值基线')
        ax.scatter(waves[hidden],prediction[hidden],color='#dc2626',s=20,label='模型预测：隐藏波段',zorder=4)
        ax.scatter(waves[~hidden],truth[~hidden],color='#2563eb',s=17,label='可见输入波段',zorder=3)
        ax.set_title(f'{sensor} / {role}'+('（非独立验证）' if role=='TRAIN' else '')+f' / 同一原生网格像元({y},{x})')
        ax.set_xlabel('中心波长 / nm');ax.set_ylabel('物理反射率');ax.grid(alpha=.2);ax.legend(fontsize=9,ncol=4)
        report['spectra'].append({'id':identifier,'role':role,'source':sensor,'pixel':[int(y),int(x)],'wavelength_nm':waves.tolist(),'truth':truth.tolist(),'baseline':baseline.tolist(),'hidden_prediction':[float(v) if np.isfinite(v) else None for v in prediction]})
    fig.tight_layout();fig.savefig(out/'spectral_profiles.png',dpi=150);plt.close(fig);figures.append(('原生像元光谱：仅隐藏波段画预测', 'spectral_profiles.png'))

    updates=[json.loads(line) for line in (run/'updates.jsonl').read_text().splitlines()];validation={}
    directory=run
    while True:
        for p in directory.glob('validation_*.json'):
            step=int(p.stem.split('_')[-1])
            if step<=state['cursor'] and step not in validation:validation[step]=json.loads(p.read_text())['metric']
        lineage=directory/'migration_lineage.json'
        if not lineage.exists():break
        directory=Path(json.loads(lineage.read_text())['source_run'])
    fig,axes=plt.subplots(1,3,figsize=(17,4.8),layout='constrained')
    x=np.array([r['epoch'] for r in updates]);y=np.array([r['loss'] for r in updates]);axes[0].plot(x,y,color='#cbd5e1',lw=.6,label='原始批次TRAIN')
    smooth=[np.mean(y[max(0,i-30):i+1]) for i in range(len(y))];axes[0].plot(x,smooth,color='#d97706',label='31批窗口均值')
    es=[];means=[]
    for e in range(state['cursor']//len(plan)):
        part=updates[e*len(plan):(e+1)*len(plan)];es.append(e+1);means.append(sum(r['loss']*r['batch_size'] for r in part)/sum(r['batch_size'] for r in part))
    axes[0].plot(es,means,'o-',color='#2563eb',label='完整Epoch观测均值');axes[0].set_yscale('log');axes[0].set_title('训练曲线（对数轴，保留尖峰）');axes[0].legend(fontsize=8)
    steps=sorted(k for k in validation if k>=175);axes[1].plot([plan.progress(k) for k in steps],[validation[k] for k in steps],'o-',ms=3,color='#dc2626',label='固定383VAL');axes[1].axhline(metric,color='#2563eb',ls='--',label=f'最佳{metric:.5f}');axes[1].set_title('固定验证：175步起的后段放大');axes[1].legend(fontsize=8)
    axes[2].plot(x,[r['lr'] for r in updates],color='#059669');axes[2].set_title('实际学习率：预热＋余弦');axes[2].set_ylabel('学习率')
    for ax in axes:ax.set_xlabel('Epoch进度');ax.grid(alpha=.2)
    for ax in axes[:2]:ax.set_ylabel('MSE / z²')
    fig.suptitle(f'已暂停于{state["epoch"]:.3f} Epoch；最佳检查点{best_epoch:g} Epoch（非10轮完成）')
    fig.savefig(out/'learning_curves.png',dpi=150);plt.close(fig);figures.insert(0,('暂停实验的真实训练/验证/学习率', 'learning_curves.png'))
    assert parameter_sha(model)==before
    for name,value in source_hashes.items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==value
    assert json.loads((run/'status.json').read_text())['status']=='STOPPED'
    report.update(status='PASS',parameters_unchanged=True,source_artifacts_unchanged=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figures=figures)
    write(out/'report.json',report)
    sections=''.join(f'<section><h2>{html.escape(title)}</h2><a href="{filename}"><img src="{filename}" loading="lazy"></a></section>' for title,filename in figures)
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V3-H最佳权重重建图集</title><style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#f4f6f9;color:#172033;max-width:1500px;margin:30px auto;padding:0 20px}section{background:white;padding:20px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}p{line-height:1.8}</style><h1>V3-H最佳权重重建图集</h1>'+f'<p>实验已暂停：{state["epoch"]:.3f} Epoch / {state["cursor"]}更新。最佳：第{best_epoch:g}轮 / {checkpoint["runner"]["cursor"]}步，完整383VAL={metric:.8f}。本次仅推理，optimizer更新0。</p><p>S2、L8按独立VAL误差分位选样；L9、EnMAP为TRAIN诊断。预测与原图固定同一物理显示范围，灰色为隐藏/无效；拼合图的可见部分来自原图。显示上限会截断亮区（L8 A P90约27.3%目标通道值超过0.3），应结合未截断误差/指标；误差色标顶部箭头表示超过0.15。验证曲线为175步起的后段放大。<a href="report.json">完整记录</a></p>'+sections+'</html>',encoding='utf-8')
    print('VISUALIZATION_PASS',str(out),flush=True)


if __name__=='__main__':main()
