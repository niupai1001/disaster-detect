"""Read-only CPU visualization of accepted USE features on fixed validation rows."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_use_independent_ssl import observation
from pretrain_use_independent import ROOT, OUT, inputs, initialize
from audit_use_independent_data import read, sha
from landslide_seg.data.optical_observation import collate_optical
from landslide_seg.models.frozen_use import FrozenUSE

torch.set_num_threads(4)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
dest = ROOT/'outputs/use_feature_view'
dest.mkdir(exist_ok=True)
checkpoint = OUT/'pretraining/multisensor/best.pt'
acceptance = read(checkpoint.parent/'acceptance.json')
assert sha(checkpoint) == acceptance['checkpoints']['best']['file_sha256']
model = initialize('multisensor', 'cpu')
model.use.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['use'], strict=True)
encoder = FrozenUSE(model.use)
before = encoder.verify()
splits, stats, _, _ = inputs('multisensor')
rows = sorted(splits['PRETRAIN_VAL']['s2'], key=lambda r:r['file'])[:2]
records=[]
for row in rows:
    obs = observation(row, stats['s2'])
    with torch.inference_mode():
        out = encoder(collate_optical([obs]))
    f=out['features'][0].numpy(); valid=out['valid'][0].numpy()
    x=obs.x.numpy()*np.array(stats['s2']['std'])[:,None,None]+np.array(stats['s2']['mean'])[:,None,None]
    records.append((row,x,f,valid))
assert before == encoder.verify()
# Fit one common visualization basis across both images; never train model weights.
pixels=np.concatenate([f[:,v].T for _,_,f,v in records])
mean=pixels.mean(0)
_,_,vt=np.linalg.svd(pixels-mean,full_matrices=False)
basis=vt[:3].T
for i in range(3):
    if basis[np.argmax(abs(basis[:,i])),i]<0: basis[:,i]*=-1
scores=(pixels-mean)@basis
lo,hi=np.percentile(scores,[2,98],axis=0)
rgbpixels=np.concatenate([x[[2,1,0]][:,v].T for _,x,_,v in records])
rlo,rhi=np.percentile(rgbpixels,[2,98],axis=0)
fig,axes=plt.subplots(2,4,figsize=(14,7.5),layout='constrained')
for n,(row,x,f,v) in enumerate(records):
    rgb=np.clip((x[[2,1,0]].transpose(1,2,0)-rlo)/(rhi-rlo+1e-9),0,1)
    pca=np.clip(((f.transpose(1,2,0)-mean)@basis-lo)/(hi-lo+1e-9),0,1)
    rgb[~v]=0;pca[~v]=0
    norm=np.linalg.norm(f,axis=0)
    for ax,data,title in zip(axes[n],[rgb,pca,norm,v],['原观测 RGB（B04/B03/B02）','USE 32维 → PCA三色显示','USE特征 L2 范数（非概率）','输出有效区：白=有效']):
        im=ax.imshow(data,interpolation='nearest',**({'cmap':'gray','vmin':0,'vmax':1} if data.dtype==bool else {'cmap':'viridis'} if data.ndim==2 else {}))
        ax.set_title(title);ax.axis('off')
        if title.startswith('USE特征'):fig.colorbar(im,ax=ax,shrink=.7)
    axes[n,0].set_ylabel('固定验证样本 '+str(n+1))
    np.savez_compressed(dest/f'sample_{n+1}.npz',reflectance=x,features=f,valid=v)
fig.suptitle('同一冻结多传感器USE；前两个按路径排序的S2重建验证样本；无训练')
fig.savefig(dest/'comparison.png',dpi=160);plt.close(fig)
_,_,f,v=records[0]
fig,axes=plt.subplots(4,8,figsize=(16,8),layout='constrained')
for i,ax in enumerate(axes.flat):
    low,high=np.percentile(f[i][v],[2,98])
    ax.imshow(np.ma.masked_where(~v,f[i]),cmap='viridis',vmin=low,vmax=high,interpolation='nearest')
    ax.set_title(f'特征通道 {i+1}');ax.axis('off')
fig.suptitle('样本1的全部32通道；各通道独立2–98%色阶，亮度不能跨通道比较')
fig.savefig(dest/'channels.png',dpi=160);plt.close(fig)
(dest/'provenance.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),use_sha256=before,selection='first two path-sorted S2 PRETRAIN_VAL rows',samples=rows,shapes=[list(r[2].shape) for r in records],pca_basis=basis.tolist(),pca_mean=mean.tolist(),training_updates=0),ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'output':str(dest),'shapes':[list(r[2].shape) for r in records]}))

