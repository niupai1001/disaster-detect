"""A's three controls, fixed member coverage, actual reconstruction and provenance."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import NativeSet
from .token_prototype import digest
from .token_experiment_a import OUT,select,features


def main():
    torch.set_num_threads(4);r=json.loads((OUT/'result.json').read_text());members=json.loads((OUT/'members.json').read_text());d=json.loads((OUT/'diagnostics.json').read_text())
    stats=torch.load(OUT/'model.pt',map_location='cpu',weights_only=False)['statistics'];cases=torch.load(OUT/'cases.pt',map_location='cpu',weights_only=False);fits=torch.load(OUT/'spectral_fits.pt',map_location='cpu',weights_only=False)
    rows=[]
    for split in ['train','val']:
        for method,res in [('model',r[split+'_model']),*r[split+'_controls'].items()]:
            for b,mse in res['bands'].items():
                rs=[x for x in res['rows'] if x['band']==b]
                rows.append(dict(split=split,method=method,band=b,count=len(rs),normalized_mse=mse,physical_rmse=np.sqrt(mse)*stats[b]['std'],physical_bias=np.mean([x['bias'] for x in rs])))
    with (OUT/'band_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    val,vm=select(NativeSet('val',device='cpu'),stats,16);assert vm==[x for x in members if x['split']=='val']
    pred_controls={};hashes={}
    for split,records in cases.items():
        for key,v in records.items():hashes[split+'/'+key]=dict(band=v['band'],input_sha256=v['input_sha256'],output_sha256=v['output_sha256'],valid_sha256=digest(v['valid']))
    for key,v in cases['val'].items():
        sensor,index=key.rsplit('_',1);i=int(index);items,targets=val[sensor];j=int(targets[i]);b=items[j];g=b.native_gsd_m;p=60//g
        x,valid=features(items,j,i);fit=fits[b.name];x=torch.where(valid,x,fit['mean'].float());spectral=((x.double()-fit['mean'])@fit['w']+fit['ym']).float()
        yy,xx=torch.meshgrid(torch.arange(b.value.shape[-2]),torch.arange(b.value.shape[-1]),indexing='ij');xy=torch.stack([yy,xx],-1).float();keep=b.valid[i].clone();keep[p:2*p,p:2*p]=False
        distance,idx=torch.cdist(xy[p:2*p,p:2*p].reshape(-1,2),xy[keep]).topk(4,largest=False);weight=distance.clamp_min(1e-6).pow(-2)
        spatial=((b.value[i][keep][idx]*weight).sum(1)/weight.sum(1)).reshape(p,p)
        pred_controls[key]=(spatial,spectral)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    for keys,filename in [(['S2_0','S2_3','L8_4'],'cases_key.png'),(list(cases['val']),'cases_all.png')]:
        fig,axes=plt.subplots(len(keys),7,figsize=(17.5,2.5*len(keys)),squeeze=False)
        for row,key in enumerate(keys):
            v=cases['val'][key];s=stats[v['band']];p=60//v['gsd'];truth=v['target'].numpy()*s['std']+s['mean'];inp=v['input'].numpy()*s['std']+s['mean'];inp[p:2*p,p:2*p]=np.nan
            target=truth[p:2*p,p:2*p];pred=v['prediction'][p:2*p,p:2*p].numpy()*s['std']+s['mean'];sp,sl=pred_controls[key];sp=sp.numpy()*s['std']+s['mean'];sl=sl.numpy()*s['std']+s['mean']
            lo,hi=np.quantile(truth[v['valid'].numpy()],[.02,.98]);hi=max(hi,lo+1e-4)
            for col,(pic,title) in enumerate(zip([truth,inp,target,sp,sl,pred,abs(pred-target)],['真实240m窗口','目标波段输入（空白为遮盖）','真实隐藏区60m','空间插值60m','其他波段线性回归60m','Transformer恢复60m','模型绝对误差'])):
                a=axes[row,col];a.imshow(pic,cmap='magma' if col==6 else 'viridis',vmin=0 if col==6 else lo,vmax=max(.01,(hi-lo)/2) if col==6 else hi);a.set_title(title,fontsize=8);a.set_xticks([]);a.set_yticks([])
                if col==0:a.set_ylabel(v['band'])
        fig.tight_layout();fig.savefig(OUT/filename,dpi=125);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.5));h=r['history'];ax[0].plot([x['step'] for x in h],[x['loss'] for x in h]);ax[0].set_yscale('log');ax[0].set_ylabel('A固定训练MSE')
    methods=['spatial','spectral','model'];names=['同波段空间插值','同位置光谱线性回归','Transformer+线性头'];values=[r['val_controls'][x]['score'] if x!='model' else r['val_model']['score'] for x in methods]
    ax[1].bar(names,values);ax[1].set_ylabel('A固定地域验证MSE')
    for a in ax:a.grid(axis='y',alpha=.2)
    ax[0].set_xlabel('更新数（最多200）');fig.tight_layout();fig.savefig(OUT/'learning_and_validation.png',dpi=150);plt.close(fig)
    populations={}
    for split in ['train','val']:
        ms=[x for x in members if x['split']==split];populations[split]=dict(observations=len(ms),groups=len({x['observation']['group'] for x in ms}),acquisitions=len({x['observation']['acquisition_id'] for x in ms}))
    result=dict(status='A_FIT_AND_SPECTRAL_DEPENDENCE_CONFIRMED_GENERALIZATION_BEHIND_SPATIAL',members=populations,hashes=hashes,
        checks=dict(hidden_value_change=r['hidden_value_intervention'],same_location_spectral_removal_score=d['spectral_region_removed']['score'],optimizer_updates=200,additional_diagnostic_updates=0),
        mask_distribution='fixed central60m target-only region in both TRAIN and VAL; no spatial mask distribution shift',
        limitation='per-target TRAIN2to4 scenes,VAL1to3;636 supervised pixels; spectrum-only regression fixedlambda.01 on identical TRAIN target pixels; cannot compare causally with old4/2 B or conclude generic semantics',
        continuation='C and all cross-satellite/physics queues remain disabled; no new training until next user instruction',
        source_hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['token_experiment_a.py','analyze_token_a.py','report_token_a.py','token_prototype.py']},checkpoint_sha256=hashlib.sha256((OUT/'model.pt').read_bytes()).hexdigest())
    (OUT/'delivery.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],members=populations)))


if __name__=='__main__':main()
