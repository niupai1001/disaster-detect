"""Reproduce full VAL and render native-grid best-checkpoint reconstruction evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--records',type=Path,help='Reuse this run\'s already reproduced full-VAL records for rendering')
    args=parser.parse_args();release=args.release.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('Use a fresh visualization directory')
    out.mkdir(parents=True)
    sys.path.insert(0,str(release/'runtime/src'))
    from encoder_v3.shared_mae.data import load_data,build_model,sample_from_dataset,identity
    from encoder_v3.shared_mae.train import batch_loss,write
    from encoder_v3.latent_response.checkpoint import parameter_sha
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    config=json.loads((release/'formal_run/config.json').read_text())
    checkpoint_path=release/'formal_run/best.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    datasets,resolver=load_data()
    assert identity(config,datasets,resolver)==checkpoint['identity'],'Frozen code/config/data mismatch'
    model=build_model(config).to(config['device']);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    before=parameter_sha(model);fixed=dict(config,seed=config['validation_seed'])
    if args.records:
        records=json.loads(args.records.read_text());metric=sum(r['loss'] for r in records)/len(records)
        assert len(records)==len(datasets['VAL'].rows) and {r['id'] for r in records}=={r['id'] for r in datasets['VAL'].rows}
    else:
        with torch.no_grad():metric,records=batch_loss(model,datasets['VAL'],datasets['VAL'].rows,resolver,fixed,epoch=0)
    assert abs(metric-checkpoint['runner']['validation']['metric'])<1e-5
    write(out/'val_members.json',records)
    report=dict(status='RUNNING',checkpoint=str(checkpoint_path),checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        identity=checkpoint['identity'],full_val_loss=metric,full_val_members=len(records),optimizer_updates=0,
        selection='Within each source sort all VAL observations by mean A/B loss, then choose P25/P50/P90 (not selected by appearance)',
        display='Native targets/predictions. Fixed reflectance display: RGB 0..0.3, gray band 0..0.5, gamma2.2; no independent prediction stretch. Gray=unavailable/hidden. Error only on hidden valid targets, fixed scale 0..0.15. B input panel shows the hidden target band only, other visible input bands are not drawn.',cases=[])
    write(out/'report.json',report);print('Full383VAL reproduced:',metric,flush=True)
    lookups={role:{r['id']:i for i,r in enumerate(ds.rows)} for role,ds in datasets.items()}

    def stats(ds,row,name):
        bank=ds.datasets[row['sensor']=='EnMAP'] if hasattr(ds,'datasets') else ds
        stat=bank.statistics['bands'][row['instrument']+':'+name]
        return stat['mean'],stat['std']

    def raw_arrays(ds,row,sample,detail,indices):
        truth=[];pred=[];valid=[];hidden=[]
        for i in indices:
            c=sample.channels[i];mu,std=stats(ds,row,c.name)
            truth.append(c.values.detach().cpu().numpy()*std+mu)
            assert detail['predictions'][i] is not None
            pred.append(detail['predictions'][i].detach().cpu().numpy()*std+mu)
            valid.append(c.valid.cpu().numpy());hidden.append(detail['target_masks'][i].cpu().numpy())
        return np.stack(truth,-1),np.stack(pred,-1),np.stack(valid,-1),np.stack(hidden,-1)

    def panels(ds,row,sample,detail,kind):
        names={c.name:i for i,c in enumerate(sample.channels)}
        if kind=='A':
            rgb=('B04','B03','B02') if row['sensor']=='S2' else ('SR_B4','SR_B3','SR_B2')
            if not all(k in names for k in rgb):
                canonical=lambda k:k[0]+k[1:].lstrip('0')
                rgb=tuple(next(k for k in names if canonical(k)==canonical(want)) for want in rgb)
            indices=[names[k] for k in rgb]
            assert all(sample.channels[i].grid==sample.channels[indices[0]].grid for i in indices)
        else:
            preferred=('B08','B8','B11','B04','B4','SR_B5','SR_B6','SR_B4')
            candidates=[i for i,p in enumerate(detail['predictions']) if p is not None]
            chosen=next((names[k] for k in preferred if k in names and names[k] in candidates),candidates[0])
            indices=[chosen]
        truth,pred,valid,mask=raw_arrays(ds,row,sample,detail,indices)
        good=valid.all(-1);hidden=mask.any(-1)&good
        assert hidden.any()
        lo=np.zeros(len(indices));hi=np.full(len(indices),.3 if kind=='A' else .5)
        def stretch(data):
            image=np.clip((data-lo)/np.maximum(hi-lo,1e-6),0,1)**(1/2.2)
            return image if len(indices)==3 else np.repeat(image,3,-1)
        original=stretch(truth);prediction=stretch(pred);original[~good]=.65
        visible=original.copy();visible[hidden]=.65
        only=prediction.copy();only[~hidden]=.65
        composite=original.copy();composite[hidden]=prediction[hidden]
        error=np.where(hidden,np.abs(pred-truth).mean(-1),np.nan)
        band_names=[sample.channels[i].name for i in indices]
        return (original,visible,only,composite,error),dict(display_bands=band_names,
            hidden_pixels=int(hidden.sum()),physical_mae=float(np.abs(pred-truth)[hidden].mean()))

    selected={};examples={};spectra=[]
    labels=('P25 较低误差','P50 典型','P90 较高误差')
    for sensor in ('S2','L8'):
        ranked=sorted((r for r in records if r['sensor']==sensor),key=lambda r:(r['loss'],r['id']))
        chosen=[ranked[round((len(ranked)-1)*q)] for q in (.25,.5,.9)]
        selected[sensor]=chosen
        for kind in ('A','B1'):
            fig,axes=plt.subplots(3,5,figsize=(18,10),squeeze=False)
            for j,record in enumerate(chosen):
                index=lookups['VAL'][record['id']];row=datasets['VAL'].rows[index]
                sample=sample_from_dataset(datasets['VAL'],index,resolver,config['device'])
                with torch.no_grad():loss,detail=model([sample],kind=kind,seed=config['validation_seed'],epoch=0,return_predictions=True)
                assert np.isfinite(float(loss))
                images,info=panels(datasets['VAL'],row,sample,detail[0],kind)
                titles=('原图','可见输入（灰色为隐藏）' if kind=='A' else '该目标波段输入（整幅隐藏）','仅隐藏区预测','可见原图＋隐藏区预测','隐藏区绝对误差')
                for k,(data,title) in enumerate(zip(images,titles)):
                    if k==4:
                        im=axes[j,k].imshow(data,cmap='magma',vmin=0,vmax=.15,interpolation='nearest');fig.colorbar(im,ax=axes[j,k],fraction=.045,pad=.025)
                    else:axes[j,k].imshow(data,interpolation='nearest')
                    axes[j,k].set_title(title,fontsize=11);axes[j,k].axis('off')
                axes[j,0].text(0,-.05,labels[j]+' / '+','.join(info['display_bands']),transform=axes[j,0].transAxes,fontsize=10)
                axes[j,2].text(0,-.05,f'整观测MSE={float(loss):.4f} z²',transform=axes[j,2].transAxes,fontsize=10)
                report['cases'].append(dict(id=record['id'],role='VAL',source=sensor,quantile=labels[j],task=kind,loss=float(loss),**info))
                if j==1:examples[(sensor,kind)]=images
                del sample,detail
            title=('Sentinel-2' if sensor=='S2' else 'Landsat')+' 验证集 / '+('A：空间遮盖重建' if kind=='A' else 'B：随机隐藏波段重建')
            fig.suptitle(title+'（最佳权重，第10轮）',fontsize=16)
            fig.text(.5,.015,'固定反射率显示：RGB 0–0.3、单波段0–0.5，γ=2.2；第三列为纯预测。B第二列只画目标波段，其他可见输入波段未画出。',ha='center',fontsize=10)
            fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(out/f'{sensor}_{kind}.png',dpi=150);plt.close(fig)
            print(sensor,kind,'rendered',flush=True);write(out/'report.json',report)

    for kind in ('A','B1'):
        fig,axes=plt.subplots(2,5,figsize=(17,7))
        for j,sensor in enumerate(('S2','L8')):
            for k,image in enumerate(examples[(sensor,kind)]):
                axes[j,k].imshow(image,cmap='magma' if k==4 else None,vmin=0 if k==4 else None,vmax=.15 if k==4 else None,interpolation='nearest')
                axes[j,k].axis('off');axes[j,k].set_title(('原图','可见输入' if kind=='A' else '目标波段输入（隐藏）','仅隐藏区预测','可见原图＋预测','隐藏区误差')[k])
            axes[j,0].text(0,-.08,sensor+' 验证集典型样本（P50）',transform=axes[j,0].transAxes)
        fig.suptitle('空间遮盖重建' if kind=='A' else '随机隐藏波段重建',fontsize=16)
        fig.subplots_adjust(hspace=.35,wspace=.06,top=.9,bottom=.1)
        fig.text(.5,.02,'固定物理反射率显示，非各图单独拉伸。误差色标统一为0–0.15。'+(' B仍使用其他可见波段；第二列只画被隐藏的目标波段。' if kind=='B1' else ''),ha='center',fontsize=10)
        fig.savefig(out/f'overview_{kind}.png',dpi=150);plt.close(fig)

    by_id={r['id']:r for r in records}
    for sensor,chosen in selected.items():
        for record in chosen:
            losses=[r['loss'] for r in report['cases'] if r['id']==record['id']]
            assert len(losses)==2 and abs(sum(losses)/2-by_id[record['id']]['loss'])<1e-5

    fig,axes=plt.subplots(3,1,figsize=(12,11))
    for j,sensor in enumerate(('S2','L8','EnMAP')):
        role='TRAIN' if sensor=='EnMAP' else 'VAL';ds=datasets[role]
        if sensor=='EnMAP':
            rows=sorted((r for r in ds.rows if r['sensor']==sensor),key=lambda r:(max(np.prod(b['shape']) for b in r['bands']),r['id']));row=rows[len(rows)//2]
        else:row=ds.rows[lookups[role][selected[sensor][1]['id']]]
        sample=sample_from_dataset(ds,lookups[role][row['id']],resolver,config['device'])
        with torch.no_grad():loss,details=model([sample],kind='B1',seed=config['validation_seed'],epoch=0,return_predictions=True)
        shapes=[c.values.shape for c in sample.channels];shape=max(set(shapes),key=shapes.count)
        ids=[i for i,c in enumerate(sample.channels) if c.values.shape==shape]
        coverage=np.stack([sample.channels[i].valid.cpu().numpy() for i in ids]).sum(0)
        candidates=np.argwhere(coverage>=max(1,.9*coverage.max()));center=(np.array(shape)-1)/2
        y,x=candidates[((candidates-center)**2).sum(-1).argmin()]
        waves=[];truth=[];baseline=[];prediction=[];hidden=[]
        for i in ids:
            c=sample.channels[i]
            if not bool(c.valid[y,x]):continue
            mu,std=stats(ds,row,c.name);waves.append(c.physical.center_um*1000)
            truth.append(float(c.values[y,x])*std+mu);baseline.append(mu)
            p=details[0]['predictions'][i];is_hidden=p is not None and bool(details[0]['target_masks'][i][y,x])
            prediction.append(float(p[y,x])*std+mu if is_hidden else np.nan);hidden.append(is_hidden)
        order=np.argsort(waves);waves=np.array(waves)[order];truth=np.array(truth)[order];baseline=np.array(baseline)[order];prediction=np.array(prediction)[order];hidden=np.array(hidden)[order]
        assert hidden.any()
        ax=axes[j];ax.plot(waves,truth,'o-',color='black',markersize=3,label='真实光谱');ax.plot(waves,baseline,'--',color='.55',label='TRAIN逐波段均值基线')
        ax.scatter(waves[hidden],prediction[hidden],color='#d62728',s=16,label='模型预测（仅隐藏波段）',zorder=4)
        ax.scatter(waves[~hidden],truth[~hidden],color='#2070c0',s=18,label='可见波段',zorder=3)
        ax.set_title(sensor+(' 训练样本（无独立EnMAP验证）' if role=='TRAIN' else ' 验证样本')+f' / 同一原生网格像元({y},{x})，{len(waves)}个有效通道')
        ax.set_xlabel('中心波长 / nm');ax.set_ylabel('反射率');ax.legend(fontsize=8,ncol=4);ax.grid(alpha=.2)
        spectra.append(dict(id=row['id'],role=role,source=sensor,pixel=[int(y),int(x)],wavelength_nm=waves.tolist(),truth=truth.tolist(),baseline=baseline.tolist(),
            hidden_prediction=[None if not np.isfinite(v) else float(v) for v in prediction]))
    fig.tight_layout();fig.savefig(out/'spectral_profiles.png',dpi=150);plt.close(fig);write(out/'spectral_profiles.json',spectra)

    meta=json.loads((release/'formal_run/status.json').read_text());directory=Path(meta['viewer']['directory'])
    metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    val=[r for r in metrics if 'val_loss' in r['values']];epochs=[r for r in metrics if 'epoch_train_loss' in r['values']];lr=[r for r in metrics if 'learning_rate' in r['values']]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([r['epoch'] for r in val],[r['values']['val_loss'] for r in val],label='完整VAL（固定遮盖）')
    axes[0].plot([r['epoch'] for r in epochs],[r['values']['epoch_train_loss'] for r in epochs],'o-',label='Epoch平均TRAIN（随机遮盖）')
    axes[0].set(xlabel='Epoch',ylabel='MSE / z²',title='实际训练与验证曲线');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].semilogy([r['epoch'] for r in lr],[r['values']['learning_rate'] for r in lr]);axes[1].set(xlabel='Epoch',ylabel='学习率（对数）',title='预热＋余弦下降：末期学习率接近零');axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=150);plt.close(fig)
    assert parameter_sha(model)==before
    report.update(status='PASS',parameter_sha_before=before,parameter_sha_after=before,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(out/'report.json',report);print('Visualization PASS',flush=True)


if __name__=='__main__':main()
