"""Read-only fixed checkpoint illustrations; no training or model selection."""
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .g5_protocol import G5,ObservationStream
from .encoder_v2 import EncoderConfig
from .encoder_v2_fast_geometry import GeometryFastObservationEncoderV2
from .encoder_v2_tasks import spectral_task,input_hash,tensor_hash
from .encoder_v2_runtime import V2Run
from .g5_cases import publish_cases
from .check_encoder_v2 import state_hash
from .expansion_data import read,save,digest


@torch.no_grad()
def main(checkpoint=None,step=1500):
    config=read(G5/'config.json');torch.set_num_threads(4);torch.cuda.set_device(config['device'])
    checkpoint=checkpoint or G5/'acceleration/before_microbatch_last.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['state']['update']==step
    model=GeometryFastObservationEncoderV2(EncoderConfig(local_width=16,query_chunk=2048)).to(config['device']);model.load_state_dict(payload['model']);model.eval();before=state_hash(model)
    stream=ObservationStream(config['device']);run=V2Run(f'encoder_v2_G5_step{step}_cases',f'G5第{step}步｜波段恢复与成组缺失｜只读可视化',dict(display_only=True,optimizer_updates=0,checkpoint_sha256=digest(checkpoint),checkpoint_update=step))
    (run.dir/'artifacts').mkdir(parents=True,exist_ok=True)
    destination=G5/'cases'/f'step_{step:05d}';publish_cases(model,stream,config,run,step)
    fig,axes=plt.subplots(4,5,figsize=(17,12),layout='constrained');entries=[];row_index=0
    for sensor,count,keys in [('S2',10,['B11','B12']),('L8',7,['SR_B6','SR_B7'])]:
        candidates=[j for j in stream.primary if stream.val.rows[j]['sensor']==sensor and stream.val.rows[j]['band_count']==count]
        index=min(candidates,key=lambda j:(sum(np.prod(b['shape']) for b in stream.val.rows[j]['bands']),stream.val.rows[j]['id']))
        meta,image=stream.image(index,'VAL');task=spectral_task(image,keys);run.heartbeat(step)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
            representation=model.encode(task.visible)
            for target in task.targets:
                prediction,coverage=model.decoder(representation,target.grid);q=target.valid&coverage;assert q.equal(target.valid)
                truth=target.values.float();error=(prediction.float()-truth).abs();visible=task.visible.bands[0]
                arrays=dict(visible=visible.values.cpu().numpy(),visible_valid=visible.keep().cpu().numpy(),target=truth.cpu().numpy(),prediction=prediction.float().cpu().numpy(),error=error.cpu().numpy(),valid=q.cpu().numpy())
                name=f'group_{sensor}_{target.band_key}.npz';np.savez_compressed(destination/name,**arrays)
                low,high=np.quantile(arrays['target'][arrays['valid']],[.01,.99]);high=max(high,low+1e-6)
                panels=[('visible','Visible '+visible.band_key),('target','True '+target.band_key),('prediction',f'G5 step{step} prediction'),('error','Absolute reflectance error'),('valid','Valid evaluation support')]
                for col,(key,title) in enumerate(panels):
                    mask=arrays['visible_valid'] if key=='visible' else arrays['valid'];limits=dict(vmin=low,vmax=high) if key in ['target','prediction'] else (dict(vmin=0,vmax=max(float(np.quantile(arrays['error'][mask],.99)),1e-6)) if key=='error' else (dict(vmin=0,vmax=1) if key=='valid' else {}))
                    ax=axes[row_index,col];im=ax.imshow(np.ma.masked_where(~mask,arrays[key]) if key!='valid' else arrays[key],cmap='magma' if key=='error' else ('gray' if key=='valid' else 'viridis'),**limits);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
                    if key=='error':fig.colorbar(im,ax=ax,shrink=.6)
                axes[row_index,0].set_ylabel(sensor+' hidden '+','.join(keys));row_index+=1
                entries.append(dict(member=meta['id'],sensor=sensor,hidden=keys,target=target.band_key,shape=list(truth.shape),mae=float(error[q].mean()),rmse=float((prediction[q].float()-truth[q]).square().mean().sqrt()),valid_pixels=int(q.sum()),input_sha256=input_hash(task),prediction_sha256=tensor_hash(prediction),arrays=name,arrays_sha256=digest(destination/name)))
    fig.savefig(destination/'grouped_predictions.png',dpi=140);plt.close(fig)
    target=run.dir/f'artifacts/grouped_predictions_{step:05d}.png';shutil.copy2(destination/'grouped_predictions.png',target);run.artifact('artifacts/'+target.name,kind='image',tag='unseen-SWIR-group',phase='diagnostic',step=step)
    assert state_hash(model)==before
    save(destination/'grouped_index.json',dict(checkpoint_sha256=digest(checkpoint),step=step,model_sha256=before,optimizer_updates=0,entries=entries,caveat='Conditional within-satellite grouped SWIR recovery; NOT cross-satellite generation. Fixed smallest native VAL cases, not aggregate generalization evidence.'))
    run.finish('completed',f'Step{step} read-only figures; no Encoder/optimizer updates')
    print('VISUALIZATION_PASS',destination,flush=True)


if __name__=='__main__':main()
