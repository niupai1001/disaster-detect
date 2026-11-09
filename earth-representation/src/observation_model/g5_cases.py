"""Fixed native-window prediction cases; inference only, never checkpoint selection."""
import shutil
import numpy as np
import torch

from .g5_protocol import G5
from .audit_ridge_bands import DEST as AUDIT,normalization
from .g5_feature_probe import baseline_inputs,baseline_values
from .encoder_v2_tasks import band_task,input_hash,tensor_hash
from .expansion_data import save,digest


@torch.no_grad()
def publish_cases(model,stream,config,run,step):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model.eval();destination=G5/'cases'/f'step_{step:05d}';destination.mkdir(parents=True,exist_ok=True)
    fits=torch.load(AUDIT/'TRAIN_fits.pt',map_location=config['device'],weights_only=False);norm=normalization()
    entries=[];fig,axes=plt.subplots(4,5,figsize=(17,12),layout='constrained');i=0
    for sensor,count,targets in [('S2',10,['B08','B11']),('L8',7,['SR_B5','SR_B6'])]:
        candidates=[j for j in stream.primary if stream.val.rows[j]['sensor']==sensor and stream.val.rows[j]['band_count']==count]
        index=min(candidates,key=lambda j:(sum(np.prod(b['shape']) for b in stream.val.rows[j]['bands']),stream.val.rows[j]['id']))
        meta,image=stream.image(index,'VAL')
        for band in targets:
            run.heartbeat(step);task=band_task(image,band);target=task.targets[0]
            original=next(b for b in image.bands if b.band_key==band);visible=[b.band_key for b in task.visible.bands]
            _,support,_,_=baseline_inputs(image,original,visible,norm);q=target.valid&support.all(-1)
            ids=q.flatten().nonzero().flatten();assert len(ids)>0
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                representation=model.encode(task.visible);prediction,coverage=model.decoder(representation,target.grid)
            assert coverage[q].all()
            ridge,_,_=baseline_values(image,original,visible,fits,norm,ids)
            reference=torch.zeros_like(target.values);reference.flatten()[ids]=ridge
            truth=target.values.float();error=(prediction.float()-truth).abs()
            a=dict(visible=task.visible.bands[0].values.cpu().numpy(),visible_valid=task.visible.bands[0].keep().cpu().numpy(),
                target=truth.cpu().numpy(),valid=q.cpu().numpy(),prediction=prediction.float().cpu().numpy(),ridge=reference.cpu().numpy(),error=error.cpu().numpy())
            filename=f'{sensor}_{band}.npz';np.savez_compressed(destination/filename,**a)
            lo,hi=np.quantile(a['target'][a['valid']],[.01,.99]);hi=max(hi,lo+1e-6)
            panels=[('visible','Visible '+task.visible.bands[0].band_key),('target','True '+band),('ridge','TRAIN Ridge'),('prediction','G5 '+str(step)),('error','Absolute error')]
            for col,(name,title) in enumerate(panels):
                mask=a['visible_valid'] if name=='visible' else a['valid']
                limits={} if name=='visible' else (dict(vmin=0,vmax=max(float(np.quantile(a['error'][mask],.99)),1e-6)) if name=='error' else dict(vmin=lo,vmax=hi))
                axes[i,col].imshow(np.ma.masked_where(~mask,a[name]),cmap='magma' if name=='error' else 'viridis',**limits)
                axes[i,col].set_title(title);axes[i,col].set_xticks([]);axes[i,col].set_yticks([])
            axes[i,0].set_ylabel(f'VAL {sensor} {original.support[0]:g}m')
            entries.append(dict(member=meta['id'],target_band=band,source_cache_sha256=meta['cache_sha256'],
                input_sha256=input_hash(task),prediction_sha256=tensor_hash(prediction),arrays=filename,arrays_sha256=digest(destination/filename),
                native_shape=list(truth.shape),transform=original.transform,crs=original.crs,valid_pixels=int(q.sum()),
                model_mae=float(error[q].mean()),ridge_mae=float((reference[q]-truth[q]).abs().mean())))
            i+=1
    path=destination/'predictions.png';fig.savefig(path,dpi=140);plt.close(fig)
    save(destination/'index.json',dict(update=step,entries=entries,optimizer_updates_during_visualization=0,
        selection='Smallest native full-band primary VAL window per sensor, tie by ID; fixed before observing G5 outcomes',
        support='Full native target, target-valid AND all Ridge input-valid; distinct from1024-pixel primary score'))
    target=run.dir/'artifacts'/f'predictions_{step:05d}.png';shutil.copy2(path,target)
    run.artifact('artifacts/'+target.name,kind='image',tag='fixed-native-predictions',phase='validation',step=step)
