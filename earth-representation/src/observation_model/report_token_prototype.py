"""Actual native-grid reconstructions, variance, hashes and bounded-stage report."""
import csv
import hashlib
import json
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .token_prototype import OUT,digest


def main():
    r=json.loads((OUT/'result.json').read_text());members=json.loads((OUT/'members.json').read_text())
    metadata={b['name']:b for m in members for b in m['observation']['bands']}
    stats=torch.load(OUT/'complete_input.pt',map_location='cpu',weights_only=False)['statistics']
    rows=[];cases={};baseline={}
    for stage in r['stages']:
        cases[stage]=torch.load(OUT/(stage+'_cases.pt'),map_location='cpu',weights_only=False)
        for split in ['train','val']:
            for band,v in cases[stage][split].items():
                s=stats[band];g=metadata[band]['native_gsd_m'];p=60//g;mask=torch.zeros_like(v['valid']);mask[:,p:2*p,p:2*p]=True
                hidden=mask if stage=='fixed_mask' else torch.zeros_like(mask)
                actual_input=torch.where(hidden,0,v['input']);input_valid=v['valid']&~hidden
                target=v['target']*s['std']+s['mean'];pred=v['prediction']*s['std']+s['mean'];valid=v['evaluation_valid'];error=(pred-target)[valid]
                row=dict(stage=stage,split=split,band=band,gsd_m=g,rmse=float(error.square().mean().sqrt()),bias=float(error.mean()),
                    prediction_variance=float(pred[valid].var(unbiased=False)),target_variance=float(target[valid].var(unbiased=False)),
                    normalized_mse=float((error/s['std']).square().mean()),actual_input_sha256=digest(actual_input),input_valid_sha256=digest(input_valid),
                    effective_tokenizer_input_sha256=digest(torch.where(input_valid,actual_input,0)),output_sha256=digest(v['prediction']))
                rows.append(row)
                np.savez_compressed(OUT/(stage+'_'+split+'_'+band.replace(':','_')+'.npz'),input=actual_input.numpy(),target=target.numpy(),prediction=pred.numpy(),valid=valid.numpy())
                if stage=='fixed_mask' and split=='val':
                    ring=torch.zeros_like(mask);ring[:,:3*p,:3*p]=True;ring&=v['valid']&~mask
                    average=(v['input']*ring).sum((1,2))/ring.sum((1,2)).clamp_min(1)
                    errors=(average[:,None,None]-v['target'])[valid]
                    baseline[band]=float(errors.square().mean())
    with (OUT/'band_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    means={s:np.mean([v for k,v in baseline.items() if k.startswith(s+':')]) for s in ['S2','L8']}
    neighbor=float(np.mean(list(means.values()))) if means else None
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    for all_bands,name in [(False,'cases_key.png'),(True,'cases_all_bands.png')]:
        bands=list(cases['complete_input']['val']) if all_bands else ['S2:B02','S2:B08','L8:SR_B5']
        cols=7 if 'fixed_mask' in cases else 3
        fig,axes=plt.subplots(len(bands),cols,figsize=(2.6*cols,2.3*len(bands)),squeeze=False)
        for i,b in enumerate(bands):
            s=stats[b];v=cases['complete_input']['val'][b];truth=v['target'][0].numpy()*s['std']+s['mean'];pred=v['prediction'][0].numpy()*s['std']+s['mean'];ok=v['valid'][0].numpy();lo,hi=np.quantile(truth[ok],[.02,.98]);hi=max(hi,lo+1e-4)
            pictures=[truth,pred,np.abs(pred-truth)];titles=['真实观测','完整输入：模型重建','完整输入：绝对误差']
            if 'fixed_mask' in cases:
                m=cases['fixed_mask']['val'][b];h=60//metadata[b]['native_gsd_m'];inp=truth.copy();inp[h:2*h,h:2*h]=np.nan;mp=m['prediction'][0].numpy()*s['std']+s['mean']
                region=np.zeros_like(ok);region[h:2*h,h:2*h]=True
                pictures += [inp,np.where(region,truth,np.nan),np.where(region,mp,np.nan),np.where(region,np.abs(mp-truth),np.nan)];titles += ['局部遮盖：可见输入','遮盖内：真实目标','遮盖内：模型恢复','遮盖内：绝对误差']
            for j,(pic,title) in enumerate(zip(pictures,titles)):
                a=axes[i,j];a.imshow(np.where(ok,pic,np.nan),cmap='magma' if j in [2,6] else 'viridis',vmin=0 if j in [2,6] else lo,vmax=max(.01,(hi-lo)/2) if j in [2,6] else hi)
                a.set_title(title,fontsize=9);a.set_xticks([]);a.set_yticks([])
                if j==0:a.set_ylabel(b,fontsize=10)
        fig.suptitle('固定验证成员：真实反射率｜完整输入与局部遮盖｜每行一条波段；输入遮盖显示为空白')
        fig.tight_layout();fig.savefig(OUT/name,dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for name,v in r['stages'].items():
        axes[0].plot([x['step'] for x in v['history']],[x['loss'] for x in v['history']],label=name)
        axes[1].plot([x['step'] for x in v['history']],[x['gradient'] for x in v['history']],label=name)
    axes[0].set_ylabel('训练标准化MSE');axes[0].set_yscale('log');axes[1].set_ylabel('裁剪前梯度范数');axes[1].set_yscale('log')
    for a in axes:a.set_xlabel('本关更新数（各200步）');a.legend();a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'learning.png',dpi=150);plt.close(fig)
    report=dict(status='BOUNDED_PROTOTYPE_COMPLETE_NO_AUTO_CONTINUATION',neighbor_mean_val_mse=neighbor,model_mask_val_mse=r['stages'].get('fixed_mask',{}).get('validation'),masked_generalization_beats_neighbor=False,
        conclusion='native disjoint tokenization and shared encoder can transmit/reconstruct spectra; fixed TRAIN mask can be fitted; held-out generalization remains limited',
        code_hashes={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['token_prototype.py','test_token_prototype.py','report_token_prototype.py','execution_gate.py','execution_policy.json']},
        checkpoint_hashes={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['complete_input.pt','fixed_mask.pt'] if (OUT/n).exists()},
        source_version_note='executed trainer frozen under source/; working copy additionally rejects unsupported isolated20m token masks; no extra optimizer steps',
        constraints='no formal training, no whole-band or cross-source continuation; measured-value copy is a separate deterministic reference')
    (OUT/'delivery.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
