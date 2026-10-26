"""CPU inference on six previously fixed validation examples, without training."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_use_v2 as rt
from landslide_seg.models.use_v2 import USEv2, FrozenReadout
from landslide_seg.models.frozen_use import state_sha256

torch.set_num_threads(4)
plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
job='downstream_tail_attention_context_fine'
folder=rt.BASE/'runs'/job
summary=rt.read(folder/'summary.json')
ckpt=folder/'best.pt'
assert rt.sha(ckpt)==summary['checkpoints']['best.pt']
saved=torch.load(ckpt,map_location='cpu',weights_only=False)
model=FrozenReadout(USEv2(),fine=True)
model.load_state_dict(saved['model'],strict=True)
# Frozen wrapper was constructed before state loading; establish the loaded-state boundary.
model.use.reference_sha256=state_sha256(model.use.encoder)
model.eval();before=model.use.verify()
config=rt.read(folder/'resolved_config.json')['configuration']
_,dataset=rt.datasets(config)
stats=rt.load_stats(rt.ROOT/config['segformer']['data']['stats'])
cases=rt.read(rt.ROOT/'docs/assets/research/use/case_provenance.json')['cases']
lookup={r['sample_id']:i for i,r in enumerate(dataset.records)}
dest=rt.BASE/'case_view';dest.mkdir(exist_ok=True)
fig,axes=plt.subplots(len(cases),4,figsize=(12,3*len(cases)),layout='constrained')
results=[]
for i,case in enumerate(cases):
    sample=dataset[lookup[case['sample_id']]]
    with torch.no_grad():
        probability=model(rt.collate([sample])['image'])[0,0].sigmoid().numpy()
    gt=sample['mask'].numpy().astype(bool);valid=sample['valid'].numpy().astype(bool)
    x=sample['image'].x.numpy()*np.array(stats['std'])[:,None,None]+np.array(stats['mean'])[:,None,None]
    rgb=np.clip(x[[2,1,0]].transpose(1,2,0)*.0001/.3,0,1)
    pred=probability>=.5
    for ax,data,title in zip(axes[i],[rgb,gt,probability,pred],['原RGB（固定反射率0–0.3）','滑坡真值（白色）','预测概率（固定0–1）','预测掩膜（阈值0.5）']):
        if data.ndim==3: ax.imshow(data,interpolation='nearest')
        else:
            cmap=plt.get_cmap('magma' if data.dtype!=bool else 'gray').copy()
            cmap.set_bad('#607d8b')
            im=ax.imshow(np.ma.masked_where(~valid,data),cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(title if i==0 else '');ax.set_xticks([]);ax.set_yticks([])
    axes[i,0].set_ylabel(case['sample_id'],fontsize=9)
    fig.colorbar(axes[i,2].images[0],ax=axes[i,2],shrink=.7)
    results.append(dict(sample_id=case['sample_id'],gt_pixels=int((gt&valid).sum()),predicted_pixels=int((pred&valid).sum()),true_positive=int((pred&gt&valid).sum()),max_probability=float(probability[valid].max())))
    np.savez_compressed(dest/(case['sample_id']+'.npz'),prob=probability,mask=gt,valid=valid,rgb=rgb)
assert before==model.use.verify()
fig.suptitle('USE v2波段注意力＋Fine：20k后续训10k；冻结编码器＋5313参数分割头\n既有固定案例，非按预测效果挑选；CPU只读推理',fontsize=13)
fig.savefig(dest/'segmentation_cases.png',dpi=130);plt.close(fig)
(dest/'provenance.json').write_text(json.dumps(dict(job=job,checkpoint_sha256=rt.sha(ckpt),use_sha256=before,cases=results,training_updates=0),ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(results,ensure_ascii=False))
