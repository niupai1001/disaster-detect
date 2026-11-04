"""Read-only bidirectional zero-shot illustration, not a cross-source acceptance."""
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rasterio.warp import reproject,Resampling
from affine import Affine

from .g5_protocol import G5,ObservationStream
from .encoder_v2 import EncoderConfig,target_from_band
from .encoder_v2_fast_geometry import GeometryFastObservationEncoderV2
from .encoder_v2_tasks import tensor_hash
from .encoder_v2_runtime import V2Run
from .check_encoder_v2 import state_hash
from .expansion_data import read,save,digest


@torch.no_grad()
def main(checkpoint=None,step=1500):
    c=read(G5/'config.json');torch.set_num_threads(4);torch.cuda.set_device(c['device']);destination=G5/'cases'/f'step_{step:05d}'
    pairs=read(destination/'cross_source_pair_candidates.json');pair=pairs[0];assert pair['crs_equal'] and pair['days']<=2
    stream=ObservationStream(c['device']);indices={r['id']:i for i,r in enumerate(stream.val.rows)}
    s2row,s2=stream.image(indices[pair['s2']],'VAL');l8row,l8=stream.image(indices[pair['l8']],'VAL')
    checkpoint=checkpoint or G5/'acceleration/before_microbatch_last.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False);assert payload['state']['update']==step
    model=GeometryFastObservationEncoderV2(EncoderConfig(local_width=16,query_chunk=2048)).to(c['device']);model.load_state_dict(payload['model']);model.eval();before=state_hash(model)
    run=V2Run(f'encoder_v2_G5_step{step}_cross_source',f'{step}步跨卫星生成｜未验证零样本诊断',dict(display_only=True,optimizer_updates=0,checkpoint_sha256=digest(checkpoint),cross_source_supervision=False));(run.dir/'artifacts').mkdir(exist_ok=True)
    fig,axes=plt.subplots(6,5,figsize=(17,18),layout='constrained');entries=[];i=0
    for direction,source,source_row,target_image,target_row,keys,visible_keys in [
        ('S2 to Landsat',s2,s2row,l8,l8row,['SR_B4','SR_B5','SR_B6'],['B04','B08','B11']),
        ('Landsat to S2',l8,l8row,s2,s2row,['B04','B08','B11'],['SR_B4','SR_B5','SR_B6'])]:
        run.heartbeat(step)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['precision']=='bf16'):
            representation=model.encode(source)
            for key,visible_key in zip(keys,visible_keys):
                band=next(b for b in target_image.bands if b.band_key==key);visible=next(b for b in source.bands if b.band_key==visible_key)
                # Target values are never passed into the encoder or decoder.
                prediction,coverage=model.decoder(representation,target_from_band(band))
                source_valid=np.zeros(tuple(band.values.shape),dtype=np.uint8)
                reproject(visible.keep().cpu().numpy().astype(np.uint8),source_valid,src_transform=Affine(*visible.transform),src_crs=visible.crs,dst_transform=Affine(*band.transform),dst_crs=band.crs,resampling=Resampling.nearest)
                valid=band.valid&coverage&torch.as_tensor(source_valid.astype(bool),device=c['device']);assert valid.any()
                truth=band.values.float();error=(prediction.float()-truth).abs()
                arrays=dict(visible=visible.values.cpu().numpy(),visible_valid=visible.keep().cpu().numpy(),target=truth.cpu().numpy(),prediction=prediction.float().cpu().numpy(),error=error.cpu().numpy(),valid=valid.cpu().numpy())
                filename=direction.replace(' ','_')+'_'+key+'.npz';np.savez_compressed(destination/filename,**arrays)
                low,high=np.quantile(arrays['target'][arrays['valid']],[.01,.99]);high=max(high,low+1e-6)
                for col,(name,title) in enumerate([('visible','Source '+visible_key),('target','Observed '+key),('prediction','ZERO-SHOT estimate'),('error','Absolute difference'),('valid','Common valid support')]):
                    mask=arrays['visible_valid'] if name=='visible' else arrays['valid'];limits=dict(vmin=low,vmax=high) if name in ['target','prediction'] else (dict(vmin=0,vmax=max(float(np.quantile(arrays['error'][mask],.99)),1e-6)) if name=='error' else (dict(vmin=0,vmax=1) if name=='valid' else {}))
                    ax=axes[i,col];im=ax.imshow(arrays[name] if name=='valid' else np.ma.masked_where(~mask,arrays[name]),cmap='magma' if name=='error' else ('gray' if name=='valid' else 'viridis'),**limits);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
                    if name=='error':fig.colorbar(im,ax=ax,shrink=.6)
                axes[i,0].set_ylabel(direction+'\nTarget '+str(band.support[0])+'m');i+=1
                entries.append(dict(direction=direction,source=source_row['id'],target=target_row['id'],source_date=source_row['date'],target_date=target_row['date'],target_band=key,target_support_m=band.support,shape=list(truth.shape),valid_pixels=int(valid.sum()),diagnostic_mae=float(error[valid].mean()),source_cache_sha256=source_row['cache_sha256'],target_cache_sha256=target_row['cache_sha256'],prediction_sha256=tensor_hash(prediction),arrays=filename,arrays_sha256=digest(destination/filename),target_transform=band.transform,target_crs=band.crs))
    fig.suptitle(f'G5 step{step} | UNVALIDATED ZERO-SHOT CROSS-SATELLITE | time gap %.3f days\nNative target grids; no measured registration residual; differences include acquisition/domain changes'%pair['days'],fontsize=12)
    fig.savefig(destination/'cross_source_zero_shot.png',dpi=140);plt.close(fig);assert state_hash(model)==before
    save(destination/'cross_source_index.json',dict(checkpoint_update=step,checkpoint_sha256=digest(checkpoint),model_sha256=before,optimizer_updates=0,pair=pair,entries=entries,pair_selection='VAL-only; bbox overlap>80%; same CRS; prefer<=2days and complete10band S2, then time gap, size. No prediction-based case selection.',limitations=['Not cross-source-trained or validated','No registration residual measurement, so pair is not formally admitted','Separate acquisitions and processing domains; difference is not pure model error','Landsat-to-S2 10m/20m queries do not establish recovered fine-scale truth','Nearest-neighbor QA transfer defines display support only; no reflectance resampling into model inputs']))
    target=run.dir/'artifacts/cross_source_zero_shot.png';shutil.copy2(destination/target.name,target);run.artifact('artifacts/'+target.name,kind='image',tag='UNVALIDATED-zero-shot-cross-satellite',phase='diagnostic',step=step);run.finish('completed','Zero-shot illustration only, no optimizer updates')
    print('CROSS_SOURCE_FIGURE_PASS',destination/'cross_source_zero_shot.png',flush=True)


if __name__=='__main__':main()
