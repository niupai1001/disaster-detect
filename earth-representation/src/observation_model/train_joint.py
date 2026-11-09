"""Joint hidden-band supervision; isolated from the frozen v2 trainers."""

from .execution_gate import require_training_authorization
if __name__ == "__main__": require_training_authorization()
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import sys

import numpy as np
import torch
from torch import nn

if __name__=='__main__' and not ('--device' in sys.argv and sys.argv[sys.argv.index('--device')+1]=='cpu'):
    torch.cuda.init()

from .data import NativeSet, RELEASE, BalancedSampler
from .model import make_model
from .check_precision import tile_values
from src.ssl.tvlp_writer import TvlpRun


class PhysicalDecoder(nn.Module):
    """Learn standardized targets, expose physical reflectance to all consumers."""
    def __init__(self, base, statistics):
        super().__init__(); self.base=base; self.statistics=statistics

    def forward(self, representation, query, tiles=None):
        z, valid=self.base(representation, query, tiles)
        s=self.statistics[query.band]
        return z.float()*s['std']+s['mean'], valid


def joint_model(statistics):
    model=make_model(coordinate_mode='explicit_global')
    model.decoder=PhysicalDecoder(model.decoder, statistics)
    return model


def balanced_masks(c, count, reserved, rng):
    """Balance hidden cardinality and band exposure without looking at values."""
    result=np.zeros((count,c),bool); exposure=np.zeros(c,int)
    sizes=np.resize(rng.permutation(np.arange(1,c//2+1)),count); rng.shuffle(sizes)
    for i,k in enumerate(sizes):
        choices=[x for x in itertools.combinations(range(c),int(k)) if x not in reserved]
        cost=np.array([sum(exposure[list(x)]) for x in choices])
        candidates=np.flatnonzero(cost==cost.min())
        chosen=choices[int(rng.choice(candidates))]
        result[i,list(chosen)]=True; exposure[list(chosen)]+=1
    if not (result.sum(0)>0).all():
        raise ValueError('batch must cover every target band')
    return result


def masked_loss(model,data,stats,plans,backward=False):
    total=0.; logs={}; target_exposure=0; pixels=0
    for sensor,(ids,masks,tiles) in plans.items():
        c=len(data.metadata[sensor]); counts=masks.sum(0)
        for i,index in enumerate(ids):
            row=torch.tensor([index],device=data.device)
            hidden=torch.as_tensor(masks[i:i+1],device=data.device)
            inputs=data.bands(sensor,row,stats,hidden)
            representation=model.encode(inputs)
            losses=[]
            for b in np.flatnonzero(masks[i]):
                query=data.query(sensor,int(b),row); name=query.band
                truth=data.value[sensor][name][row]; valid=data.valid[sensor][name][row]
                selected=tile_values(valid,tiles[i],query.grid_gsd_m)
                prediction,coverage=model.decoder(representation,query,tiles[i])
                keep=selected&coverage
                if not valid.any():raise ValueError('no real target pixels')
                if not torch.equal(keep,selected):raise ValueError('target coverage differs from actual QA')
                error=((prediction-tile_values(truth,tiles[i],query.grid_gsd_m))/stats[name]['std']).square()
                value=(error*keep).sum()*(400/len(tiles[i]))/valid.sum()
                losses.append(value/(2*c*int(counts[b])))
                logs[name]=logs.get(name,0.)+float(value.detach())/int(counts[b])
                target_exposure+=1; pixels+=int(keep.sum())
            loss=sum(losses)
            if not torch.isfinite(loss):raise FloatingPointError('joint loss is not finite')
            if backward:loss.backward()
            total+=float(loss.detach())
    return total,logs,target_exposure,pixels


def schedule(step,maximum,warmup):
    if step<warmup:return 1e-4*(step+1)/warmup
    end=max(warmup+1,int(.8*maximum))
    t=min(1.,max(0.,(step-warmup)/(end-warmup)))
    return 1e-6+(1e-4-1e-6)*.5*(1+math.cos(math.pi*t))


def cross_loss(model,pairs,stats,positions,backward=False):
    """Both directions, every real target band; one encode per source view."""
    from .train_cross import cross_query
    total=0.; logs={}
    for source,target in [('S2','L8'),('L8','S2')]:
        bands=pairs.data.metadata[target]; losses=[]
        ids=pairs.indices[source][positions]
        representation=model.encode(pairs.data.bands(source,ids,stats))
        for b,band in enumerate(bands):
            query=cross_query(pairs.data,target,b,pairs.phase[positions])
            prediction,coverage=model.decoder(representation,query)
            valid=pairs.valid[target][positions,b]
            if not torch.equal(valid&coverage,valid):raise ValueError('cross prediction misses real QA pixels')
            error=((prediction-pairs.target[target][positions,b])/stats[band['name']]['std']).square()
            value=((error*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)).mean()
            losses.append(value/(2*len(bands)))
            logs[source+'_to_'+target+'/'+band['name']]=float(value.detach())
        loss=sum(losses)
        if not torch.isfinite(loss):raise FloatingPointError('cross loss is not finite')
        if backward:loss.backward()
        total+=float(loss.detach())
    return total,logs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--stop-after',type=int,default=0)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--fit',action='store_true')
    parser.add_argument('--fixed-lr',type=float)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--pilot',action='store_true',help='fixed geographic VAL every250 updates; no TEST')
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
    stats=json.loads((RELEASE/'statistics.json').read_text())['bands']
    contract=json.loads((RELEASE/'masking.json').read_text())
    data=NativeSet('train',device=args.device); model=joint_model(stats).to(args.device)
    decay=[]; no_decay=[]
    for n,p in model.named_parameters():
        (no_decay if p.ndim<=1 or n.endswith('.bias') else decay).append(p)
    optimizer=torch.optim.AdamW([dict(params=decay,weight_decay=.04),dict(params=no_decay,weight_decay=0.)],lr=1e-4)
    rng=np.random.default_rng(0); generator=torch.Generator(device=args.device).manual_seed(0)
    sampler=BalancedSampler({s:len(r) for s,r in data.records.items()})
    step=0; history=[]; exposures=0; initial=None; validation_history=[]; best=float('inf')
    config=dict(version='joint_v3_pilot' if args.pilot else 'joint_v3_engineering',steps=args.steps,fit=args.fit,precision='FP32',effective_batch=6 if args.fit else 32,
        statistics_sha256=hashlib.sha256((RELEASE/'statistics.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        objective='equal sensor, band, observation; TRAIN-standardized MSE; all hidden targets',fixed_lr=args.fixed_lr,device=args.device,
        pilot=args.pilot,augmentation='none; candidate strategy comparison, not single-factor attribution')
    if args.resume:
        state=torch.load(args.out/'last.pt',map_location='cpu',weights_only=False)
        assert state['config']==config
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        for group in optimizer.state.values():
            for k,v in group.items():
                if isinstance(v,torch.Tensor) and k!='step':group[k]=v.to(args.device)
        sampler.load_state_dict(state['sampler']); rng.bit_generator.state=state['rng']; generator.set_state(state['tile_rng'])
        torch.set_rng_state(state['torch_rng'])
        if args.device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        step=state['step'];history=state['history'];exposures=state['target_exposure'];initial=state['initial_fixed']
        validation_history=state.get('validation_history',[]);best=state.get('best',float('inf'))
    elif (args.out/'last.pt').exists():raise FileExistsError('explicit resume required')
    def plans(fixed=False):
        result={}
        for sensor in ['S2','L8']:
            c=len(data.metadata[sensor]); n=16
            if fixed:
                ids=list(range(3)); mask=np.zeros((3,c),bool)
                for j in range(3):mask[j,j::3]=True
                tiles=torch.arange(16,device=args.device).expand(3,-1)
            else:
                ids=sampler.take(sensor,n)
                reserved={tuple(x) for x in contract[sensor]['unseen_combinations']}
                mask=balanced_masks(c,n,reserved,rng)
                tiles=torch.stack([torch.randperm(400,device=args.device,generator=generator)[:16] for _ in ids])
            result[sensor]=(ids,mask,tiles)
        return result
    fixed=plans(True)
    validation=None;validation_plans={}
    if args.pilot:
        if args.fit or args.device!='cuda':raise ValueError('pilot requires the full TRAIN pool and GPU')
        validation=NativeSet('val');vrng=np.random.default_rng(1907)
        for sensor in ['S2','L8']:
            c=len(validation.metadata[sensor]);n=len(validation.records[sensor])
            reserved={tuple(x) for x in contract[sensor]['unseen_combinations']}
            validation_plans[sensor]=(list(range(n)),balanced_masks(c,n,reserved,vrng),torch.arange(400,device=args.device).expand(n,-1))
    tv=TvlpRun(args.out.parent/'tvlp',project='conditional-observation',name=args.out.name,run_id=args.out.name,resume=args.resume,seed=0,config=config,
        metric_definitions={'train/loss':{'direction':'min'},'fixed/score':{'direction':'min'}})
    def save(name='last.pt'):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),rng=rng.bit_generator.state,
            tile_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else [],step=step,history=history,
            target_exposure=exposures,initial_fixed=initial,config=config,statistics=stats,validation_history=validation_history,best=best)
        tmp=args.out/(name+'.tmp');torch.save(state,tmp);tmp.replace(args.out/name)
    start=time.perf_counter()
    try:
        if initial is None:
            with torch.no_grad():initial=masked_loss(model,data,stats,fixed)[0]
        while step<args.steps:
            model.train();optimizer.zero_grad(set_to_none=True)
            lr=args.fixed_lr or schedule(step,args.steps,500 if args.pilot else min(500,max(1,args.steps//10)))
            for group in optimizer.param_groups:group['lr']=lr
            loss,bands,exposure,pixels=masked_loss(model,data,stats,fixed if args.fit else plans(),True)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True)
            optimizer.step();step+=1;exposures+=exposure
            tv.log('train',dict(loss=loss,lr=lr,gradient_norm=float(norm),target_exposure=exposures,valid_pixels=pixels,source_exposure=step*config['effective_batch']),global_step=step);tv.heartbeat(step)
            if step%10==0 or step==1 or step==args.steps:
                model.eval()
                with torch.no_grad():score,per_band,_,_=masked_loss(model,data,stats,fixed)
                history.append(dict(step=step,loss=loss,fixed=score,bands=per_band));tv.log('fixed',dict(score=score,**per_band),global_step=step)
                print(json.dumps(history[-1]),flush=True);save()
            if args.pilot and (step%250==0 or step==args.steps):
                from .train_masked import evaluation
                model.eval()
                with torch.no_grad():
                    single=evaluation(model,validation,stats,'float32')
                    model.eval();multi,_,_,_=masked_loss(model,validation,stats,validation_plans)
                singleton_mse=sum(sum(b['normalized_rmse']**2 for b in single['bands'].values() if b['sensor']==s)/len(validation.metadata[s]) for s in ['S2','L8'])/2
                score=.5*(singleton_mse+multi);improved=score<best;best=min(best,score)
                record=dict(step=step,score=score,singleton_mse=singleton_mse,multiple_mse=multi,singleton_nrmse=single['score'])
                validation_history.append(record)
                (args.out/f'validation_{step}.json').write_text(json.dumps(dict(**record,singleton=single),indent=2))
                tv.log('val',record,global_step=step);print(json.dumps(dict(validation=record)),flush=True);save()
                if improved:save('best.pt')
            if args.stop_after and step>=args.stop_after:save();tv.finish('cancelled','engineering disk resume');return
        save();tv.finish()
        result=dict(status='pilot_complete_pending_comparison' if args.pilot else 'engineering_complete',step=step,initial_fixed=initial,final_fixed=history[-1]['fixed'],history=history,seconds=time.perf_counter()-start,
            target_exposure=exposures,config=config,validation_history=validation_history,best=best if args.pilot else None,boundary='no TEST; pilot cap is not confirmed convergence')
        (args.out/'result.json').write_text(json.dumps(result,indent=2))
    except BaseException as exc:tv.finish('failed',str(exc));raise


if __name__=='__main__':main()
