"""5750-step fixed-case remaining real bands, matched TRAIN Ridge support."""
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .g5_protocol import G5,ObservationStream
from .encoder_v2 import EncoderConfig
from .encoder_v2_fast_geometry import GeometryFastObservationEncoderV2
from .encoder_v2_tasks import band_task,input_hash,tensor_hash
from .encoder_v2_runtime import V2Run
from .audit_ridge_bands import DEST as AUDIT,normalization
from .g5_feature_probe import baseline_inputs,baseline_values
from .check_encoder_v2 import state_hash
from .expansion_data import read,save,digest


@torch.no_grad()
def main():
    config=read(G5/'config.json');torch.set_num_threads(4);torch.cuda.set_device(config['device'])
    base=G5/'cases/step_05750';destination=base/'remaining';destination.mkdir(exist_ok=True)
    checkpoint=base/'visualization_checkpoint.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False);assert payload['state']['update']==5750
    model=GeometryFastObservationEncoderV2(EncoderConfig(local_width=16,query_chunk=2048)).to(config['device']);model.load_state_dict(payload['model']);model.eval();before=state_hash(model)
    stream=ObservationStream(config['device']);original=read(base/'index.json');members={('S2' if r['target_band'].startswith('B') else 'L8'):r['member'] for r in original['entries']};indices={r['id']:i for i,r in enumerate(stream.val.rows)}
    fits=torch.load(AUDIT/'TRAIN_fits.pt',map_location=config['device'],weights_only=False);norm=normalization()
    run=V2Run('encoder_v2_G5_step5750_remaining_bands','5750步｜其余13波段与Ridge｜固定留出案例',dict(display_only=True,optimizer_updates=0,checkpoint_update=5750,checkpoint_sha256=digest(checkpoint)));(run.dir/'artifacts').mkdir(exist_ok=True)
    entries=[]
    for sensor,keys,name in [('S2',['B02','B03','B04','B12'],'S2_visible_SWIR'),('S2',['B05','B06','B07','B8A'],'S2_rededge_NIR'),('L8',['SR_B1','SR_B2','SR_B3','SR_B4','SR_B7'],'Landsat_remaining')]:
        meta,image=stream.image(indices[members[sensor]],'VAL');fig,axes=plt.subplots(len(keys),5,figsize=(17,3*len(keys)),squeeze=False,layout='constrained')
        for i,key in enumerate(keys):
            run.heartbeat(5750);task=band_task(image,key);target=task.targets[0];band=next(b for b in image.bands if b.band_key==key);visible=[b.band_key for b in task.visible.bands]
            _,support,_,_=baseline_inputs(image,band,visible,norm);q=target.valid&support.all(-1);ids=q.flatten().nonzero().flatten();assert len(ids)>0
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                representation=model.encode(task.visible);prediction,coverage=model.decoder(representation,target.grid)
            assert coverage[q].all()
            estimate,_,_=baseline_values(image,band,visible,fits,norm,ids);ridge=torch.zeros_like(target.values);ridge.flatten()[ids]=estimate
            truth=target.values.float();model_error=(prediction.float()-truth).abs();ridge_error=(ridge-truth).abs()
            arrays=dict(target=truth.cpu().numpy(),ridge=ridge.cpu().numpy(),prediction=prediction.float().cpu().numpy(),model_error=model_error.cpu().numpy(),ridge_error=ridge_error.cpu().numpy(),valid=q.cpu().numpy())
            filename=f'{sensor}_{key}.npz';np.savez_compressed(destination/filename,**arrays)
            low,high=np.quantile(arrays['target'][arrays['valid']],[.01,.99]);high=max(high,low+1e-6);error_high=max(float(np.quantile(np.r_[arrays['model_error'][arrays['valid']],arrays['ridge_error'][arrays['valid']]],.99)),1e-6)
            for j,(field,title) in enumerate([('target','Observed '+key),('ridge','TRAIN Ridge'),('prediction','G5 step5750'),('ridge_error','Ridge absolute error'),('model_error','G5 absolute error')]):
                ax=axes[i,j];err=field.endswith('error');im=ax.imshow(np.ma.masked_where(~arrays['valid'],arrays[field]),cmap='magma' if err else 'viridis',vmin=0 if err else low,vmax=error_high if err else high);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
                if err:fig.colorbar(im,ax=ax,shrink=.6)
            axes[i,0].set_ylabel(f'{sensor} {key} | {band.support[0]:g}m')
            entries.append(dict(sensor=sensor,band=key,member=meta['id'],visible_bands=visible,valid_pixels=int(q.sum()),model_mae=float(model_error[q].mean()),ridge_mae=float(ridge_error[q].mean()),model_rmse=float((prediction[q].float()-truth[q]).square().mean().sqrt()),ridge_rmse=float((ridge[q]-truth[q]).square().mean().sqrt()),input_sha256=input_hash(task),prediction_sha256=tensor_hash(prediction),arrays=filename,arrays_sha256=digest(destination/filename),target_transform=band.transform,target_crs=band.crs))
            del representation
        fig.savefig(destination/(name+'.png'),dpi=140);plt.close(fig);shutil.copy2(destination/(name+'.png'),run.dir/'artifacts'/(name+'.png'));run.artifact('artifacts/'+name+'.png',kind='image',tag='matched-single-band-Ridge',phase='diagnostic',step=5750)
        print('REMAINING_PAGE_DONE',name,flush=True)
    assert state_hash(model)==before
    save(destination/'index.json',dict(checkpoint_update=5750,checkpoint_sha256=digest(checkpoint),optimizer_updates=0,entries=entries,selection='Same fixed complete-band VAL members as previous1500/5750 figures; no prediction-based selection',support='Target-valid AND all Ridge covariates valid, equal for both methods; target fully excluded from Encoder',color_scales='Same row: observed/Ridge/G5 share reflectance scale; two error panels share error scale. Scales differ across rows.'))
    import csv
    with (destination/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=['sensor','band','model_mae','ridge_mae','model_rmse','ridge_rmse','valid_pixels']);writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in entries)
    run.finish('completed','13 remaining real bands; fixed checkpoint; zero optimizer updates');print('REMAINING_BANDS_PASS',flush=True)


if __name__=='__main__':main()
