"""Fixed untrained predictions for the G3 task preview; CPU, no updates."""
from datetime import datetime,timezone
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .encoder_v2 import ObservationEncoderV2
from .encoder_v2_runtime import BASE,G3,LOG_ROOT,load_members
from .encoder_v2_tasks import spatial_task,band_task,seed_for,run_task
from .expansion_data import OUT,read,save,digest


def main():
    start=time.perf_counter();torch.set_num_threads(4)
    model=ObservationEncoderV2().eval();model.load_state_dict(torch.load(BASE/'G2_arch/initialization_untrained.pt',map_location='cpu',weights_only=True)['model'])
    data=load_members(read(G3/'members.json'));stats=read(OUT/'statistics.json')['bands'];cases=[]
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(4,5,figsize=(15,12),layout='constrained')
    for sensor in ['S2','L8']:
        meta,image=next((m,im) for m,im in data if m['role']=='TRAIN' and m['sensor']==sensor)
        for task in [spatial_task(image,seed_for(meta['id'],'A',0)),band_task(image,'B11' if sensor=='S2' else 'SR_B5')]:
            with torch.no_grad():rows,exposure,predictions=run_task(model,task,meta,stats,True)
            result=predictions[0];index=len(cases);y=result['prediction'].numpy();truth=result['target'].numpy();q=result['valid'].numpy()
            visible=next((b for b in task.visible.bands if b.band_key==result['band']),task.visible.bands[0])
            x=visible.values.numpy();v=visible.keep().numpy();lo,hi=np.quantile(truth[q],[.01,.99]);hi=max(hi,lo+1e-6)
            axes[index,0].imshow(np.ma.masked_where(~v,x),cmap='viridis');axes[index,0].set_title(f'{sensor} {task.kind} 可见 '+visible.band_key)
            axes[index,1].imshow(np.ma.masked_where(~q,truth),cmap='viridis',vmin=lo,vmax=hi);axes[index,1].set_title('真实目标 '+result['band'])
            axes[index,2].imshow(np.ma.masked_where(~q,y),cmap='viridis',vmin=lo,vmax=hi);axes[index,2].set_title('随机初始化预测（同色标）')
            error=np.abs(y-truth);axes[index,3].imshow(np.ma.masked_where(~q,error),cmap='magma');axes[index,3].set_title(f'绝对误差 MAE={error[q].mean():.4f}')
            axes[index,4].imshow(q,cmap='gray',vmin=0,vmax=1);axes[index,4].set_title('实际评价/监督区')
            path=G3/'cases'/f'{sensor}_{task.kind}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,visible=x,visible_valid=v,target=truth,prediction=y,valid=q)
            cases.append(dict(member=meta['id'],sensor=sensor,task=task.kind,band=result['band'],source=exposure,
                arrays=str(path),sha256=digest(path),output_hash=result['output_hash'],device='CPU',optimizer_updates=0))
    fig.suptitle('G3固定案例：随机初始化、0更新；仅用于核对输入/目标/预测/有效区，不是学习效果')
    path=G3/'preview/dry_run_predictions.png';fig.savefig(path,dpi=130);plt.close(fig)
    run=LOG_ROOT/'encoder_v2_G3_zero_update';target=run/'artifacts/dry_run_predictions.png';target.write_bytes(path.read_bytes())
    events=[json.loads(line) for line in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    event=dict(type='artifact',timestamp=datetime.now(timezone.utc).isoformat(),session_id=events[-1]['session_id'],
        path='artifacts/dry_run_predictions.png',artifact_type='image',tag='fixed-untrained-cases',phase='diagnostic',global_step=0,
        role='comparison',metadata=dict(generated_after_dry_run=True,device='CPU',optimizer_updates=0))
    with (run/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    save(G3/'cases/index.json',dict(cases=cases,wall_seconds=time.perf_counter()-start,optimizer_updates=0))
    print('G3_FIXED_UNTRAINED_CASES_PUBLISHED')


if __name__=='__main__':main()
