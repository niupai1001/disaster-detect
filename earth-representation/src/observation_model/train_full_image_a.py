"""User-authorized A: complete original files,50 TRAIN per sensor, bounded2000 updates."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch
if __name__=='__main__':torch.cuda.init()
from .data import RELEASE,ROOT
from .full_image_data import read_full
from .model import BandRaster
from .token_prototype import PatchModel,digest
from src.ssl.tvlp_writer import TvlpRun

OUT=ROOT/'observation_model/outputs/token_A_full'


def prepare():
    rows=[json.loads(x) for x in (RELEASE/'single_masked.jsonl').read_text().splitlines()]
    data={'train':[],'val':[]};manifest=[]
    for split in ['train','val']:
        for sensor in ['S2','L8']:
            pool=[r for r in rows if r['split']==split and r['sensor']==sensor]
            pool.sort(key=lambda r:hashlib.sha256(('0:'+r['observation_id']).encode()).hexdigest())
            groups=set();acq=set();chosen=[]
            for row in pool:
                if split=='train' and (row['group'] in groups or row['acquisition_id'] in acq):continue
                arrays,record=read_full(row)
                if split=='train' and min(b['valid_fraction'] for b in record['full_bands'])<.5:continue
                target=len(chosen)%len(arrays);n=record['full_footprint_m']//60
                locations=sorted(((y,x) for y in range(1,n-1) for x in range(1,n-1)),key=lambda p:((p[0]+.5-n/2)**2+(p[1]+.5-n/2)**2,p))
                location=None
                for y,x in locations:
                    if all(float(v[y*(60//b['native_gsd_m']):(y+1)*(60//b['native_gsd_m']),x*(60//b['native_gsd_m']):(x+1)*(60//b['native_gsd_m'])].float().mean())>=.8 for (_,v),b in zip(arrays,record['full_bands'])):
                        location=[y,x];break
                if location is None:raise ValueError('no valid local target in original file: '+row['observation_id'])
                record.update(target_index=target,target_band=record['full_bands'][target]['name'],mask_patch_yx=location)
                data[split].append(dict(arrays=arrays,record=record));chosen.append(row);manifest.append(record);groups.add(row['group']);acq.add(row['acquisition_id'])
                if split=='train' and len(chosen)==50:break
            if split=='train' and len(chosen)!=50:raise ValueError(f'Only{len(chosen)} legitimate whole-file TRAIN {sensor}, need50')
    train=[r for r in manifest if r['split']=='train'];val=[r for r in manifest if r['split']=='val']
    assert not ({r['group'] for r in train}&{r['group'] for r in val})
    assert not ({r['acquisition_id'] for r in train}&{r['acquisition_id'] for r in val})
    for a in train:
        for b in val:
            x=a['full_geographic_bbox'];y=b['full_geographic_bbox']
            if max(x[0],y[0])<min(x[2],y[2]) and max(x[1],y[1])<min(x[3],y[3]):raise ValueError('complete raw extents overlap across roles')
    return data,manifest


def inputs(obs,stats):
    r=obs['record'];items=[]
    for (value,valid),band in zip(obs['arrays'],r['full_bands']):
        s=stats[band['name']];items.append(BandRaster(band['name'],((value-s['mean'])/s['std'])[None].cuda(),valid[None].cuda(),band['wavelength_nm'],band['bandwidth_nm'],band['native_gsd_m'],torch.tensor([band['phase_xy_m']],device='cuda')))
    n=r['full_footprint_m']//60;h=torch.zeros(1,len(items),2*n,2*n,device='cuda',dtype=torch.bool);y,x=r['mask_patch_yx'];h[:,r['target_index'],2*y:2*y+2,2*x:2*x+2]=True
    return items,h


def evaluate(model,observations,stats,backward=False):
    counts={}
    for o in observations:
        b=o['record']['target_band'];counts[b]=counts.get(b,0)+1
    c={s:sum(b.startswith(s+':') for b in counts) for s in ['S2','L8']};total=0.;rows=[]
    for o in observations:
        r=o['record'];items,h=inputs(o,stats);j=r['target_index'];b=items[j];pred=model(items,h)[j];p=60//b.native_gsd_m;y,x=r['mask_patch_yx'];ys=slice(y*p,(y+1)*p);xs=slice(x*p,(x+1)*p)
        valid=b.valid[:,ys,xs];truth=b.value[:,ys,xs];prediction=pred[:,ys,xs];error=(prediction-truth)[valid]
        mse=error.square().mean();loss=mse/(2*c[r['sensor']]*counts[b.name])
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite original-file A loss')
        if backward:loss.backward()
        total+=float(loss.detach());rows.append(dict(id=r['observation_id'],sensor=r['sensor'],band=b.name,mse=float(mse.detach()),rmse=float(mse.detach().sqrt())*stats[b.name]['std'],bias=float(error.detach().mean())*stats[b.name]['std'],target_count=int(valid.sum()),prediction_variance=float(prediction[valid].detach().var(unbiased=False))))
    return total,rows


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    policy=json.loads((ROOT/'observation_model/execution_policy.json').read_text())
    if policy.get('authorized_A_full_max_updates',0)!=2000:raise RuntimeError('full-image A requires current user stage authorization')
    if (OUT/'result.json').exists():raise FileExistsError('preserve completed full-image A')
    torch.manual_seed(0);torch.set_num_threads(4)
    data,members=prepare();(OUT/'members.json').write_text(json.dumps(members,indent=2))
    state=torch.load(ROOT/'observation_model/outputs/token_v1/complete_input.pt',map_location='cpu',weights_only=False);stats=state['statistics']
    model=PatchModel().cuda();model.load_state_dict(state['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=2,threshold=.001,threshold_mode='rel',min_lr=1e-6)
    step=0;history=[];best=float('inf');anchor=float('inf');stale=0;reason='update_limit'
    config=dict(version='full_original_A',train_per_sensor=50,val='all frozen held-out originals',source_extent='all original native pixels; invalid padding only',max_updates=2000,min_early_stop=500,val_interval=100,early_stop_patience=8,relative_improvement=.001,lr=.001,weight_decay=0,effective_batch=100,microbatch=1,seed=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),statistics_sha256=hashlib.sha256(json.dumps(stats,sort_keys=True).encode()).hexdigest())
    last=OUT/'last.pt';resume=last.exists()
    if resume:
        s=torch.load(last,map_location='cuda',weights_only=False);assert s['config']==config
        assert s['member_sha256']==hashlib.sha256((OUT/'members.json').read_bytes()).hexdigest(),'resume members or input hashes changed'
        model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);step=s['step'];history=s['history'];best=s['best'];anchor=s['anchor'];stale=s['stale'];torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda_rng']])
    tv=TvlpRun(OUT/'tvlp',project='conditional-observation',name='A完整原始影像｜S2和Landsat各50训练｜最多2000步',run_id='A_full_native50_v2',resume=resume,seed=0,config=config,metric_definitions={'train/loss':{'direction':'min'},'val/score':{'direction':'min'},'train/lr':{'direction':'none'}})
    def save(name):
        s=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,history=history,best=best,anchor=anchor,stale=stale,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),statistics=stats,config=config,member_sha256=hashlib.sha256((OUT/'members.json').read_bytes()).hexdigest())
        tmp=OUT/(name+'.tmp');torch.save(s,tmp);tmp.replace(OUT/name)
    start=time.perf_counter()
    try:
        from .full_a_controls import fit,evaluate as evaluate_controls
        with torch.no_grad():
            fits=fit(data['train'],stats)
            controls={role:evaluate_controls(obs,stats,fits) for role,obs in data.items()}
        torch.save(fits,OUT/'spectral_fits.pt');(OUT/'controls.json').write_text(json.dumps(controls,indent=2));(OUT/'config.json').write_text(json.dumps(config,indent=2))
        while step<2000:
            model.train();optimizer.zero_grad(set_to_none=True);lr=optimizer.param_groups[0]['lr'];loss,rows=evaluate(model,data['train'],stats,True)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True);optimizer.step();step+=1
            tv.log('train',dict(loss=loss,gradient_norm=float(grad),lr=lr,source_exposure=100*step,epoch_equivalent=step),global_step=step,wall_time_sec=time.perf_counter()-start);tv.heartbeat(step)
            if step==1 or step%10==0:print(json.dumps(dict(step=step,loss=loss,lr=lr,seconds=time.perf_counter()-start)),flush=True);save('last.pt')
            if step%100==0:
                model.eval()
                with torch.no_grad():score,vr=evaluate(model,data['val'],stats)
                improved=score<best;best=min(best,score)
                if score<anchor*.999:anchor=score;stale=0
                else:stale+=1
                scheduler.step(score);history.append(dict(step=step,score=score,lr=lr,stale=stale));tv.log('val',dict(score=score,stale=stale),global_step=step)
                (OUT/f'validation_{step}.json').write_text(json.dumps(dict(score=score,rows=vr),indent=2));save('last.pt')
                if improved:save('best.pt')
                print(json.dumps(dict(validation=history[-1])),flush=True)
                if step>=500 and stale>=8:reason='fixed_geographic_validation_plateau';break
        save('last.pt');model.eval()
        with torch.no_grad():last_val,last_rows=evaluate(model,data['val'],stats)
        best_state=torch.load(OUT/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(best_state['model'])
        with torch.no_grad():best_train,bt_rows=evaluate(model,data['train'],stats);best_val,bv_rows=evaluate(model,data['val'],stats)
        tv.log('final_val',dict(best_score=best_val,last_score=last_val,spatial=controls['val']['spatial']['score'],spectral=controls['val']['spectral']['score']),global_step=step);tv.finish()
        (OUT/'result.json').write_text(json.dumps(dict(status='A_FULL_COMPLETE',step=step,best=best,stop_reason=reason,history=history,config=config,seconds=time.perf_counter()-start,best_step=best_state['step'],best_model=dict(train=best_train,val=best_val,train_rows=bt_rows,val_rows=bv_rows),last_model=dict(val=last_val,val_rows=last_rows)),indent=2))
    except BaseException as exc:tv.finish('failed',str(exc));raise


if __name__=='__main__':main()
