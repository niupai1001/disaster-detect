"""A only: fixed single-band local masks, bounded fitting gate, then held-out scenes."""
import hashlib
import json
from pathlib import Path
import time
import torch

if __name__=='__main__':torch.cuda.init()

from .data import NativeSet, RELEASE
from .model import BandRaster
from .token_prototype import PatchModel, SIZE, digest
from .operators import resample_band
from src.ssl.tvlp_writer import TvlpRun

OUT=Path(__file__).parent/'outputs/token_A'


def select(data,stats,n):
    batches={};members=[]
    for sensor in ['S2','L8']:
        eligible=[]
        for i,r in enumerate(data.records[sensor]):
            good=True
            for b in data.metadata[sensor]:
                g=b['native_gsd_m'];a=480//g;p=60//g;v=data.valid[sensor][b['name']][i,a:a+SIZE//g,a:a+SIZE//g]
                if float(v.float().mean())<.8 or float(v[p:2*p,p:2*p].float().mean())<.8:good=False;break
            if good:eligible.append(i)
        # First sample geographically distinct groups/acquisitions, independently of band values.
        eligible.sort(key=lambda i:hashlib.sha256(('0:'+data.records[sensor][i]['observation_id']).encode()).hexdigest())
        chosen=[];groups=set();acquisitions=set()
        for i in eligible:
            r=data.records[sensor][i]
            if r['group'] not in groups and r['acquisition_id'] not in acquisitions:
                chosen.append(i);groups.add(r['group']);acquisitions.add(r['acquisition_id'])
            if len(chosen)==n:break
        if len(chosen)<n:raise ValueError(f'{sensor}: only{len(chosen)} independent eligible groups, need{n}')
        ids=torch.tensor(chosen,device=data.device);items=[]
        for b in data.bands(sensor,ids,stats):
            a=480//b.native_gsd_m;z=a+SIZE//b.native_gsd_m
            items.append(BandRaster(b.name,b.value[:,a:z,a:z],b.valid[:,a:z,a:z],b.wavelength_nm,b.bandwidth_nm,b.native_gsd_m,b.phase_xy_m))
        targets=torch.arange(n,device=data.device)%len(items)
        batches[sensor]=(items,targets)
        for j,i in enumerate(chosen):members.append(dict(split=data.split,sensor=sensor,observation=data.records[sensor][i],target=items[int(targets[j])].name,mask_patch_yx=[1,1],crop_start_m=480,crop_size_m=SIZE))
    return batches,members


def slice_items(items,ids):
    return [BandRaster(b.name,b.value[ids],b.valid[ids],b.wavelength_nm,b.bandwidth_nm,b.native_gsd_m,b.phase_xy_m[ids]) for b in items]


def masks(items,targets):
    h=torch.zeros(len(targets),len(items),8,8,device=targets.device,dtype=torch.bool)
    h[torch.arange(len(targets),device=h.device),targets,2:4,2:4]=True
    return h


def network(model,batches,stats,backward=False,save_cases=False):
    total=0.;rows=[];cases={}
    for sensor,(items,targets) in batches.items():
        c=len(items);counts=torch.bincount(targets,minlength=c)
        for begin in range(0,len(targets),4):
            ix=torch.arange(begin,min(begin+4,len(targets)),device=targets.device);bs=slice_items(items,ix);ts=targets[ix]
            h=masks(bs,ts);pred=model(bs,h);loss=0.
            for j,b in enumerate(bs):
                g=b.native_gsd_m;p=60//g
                for k in (ts==j).nonzero().flatten().tolist():
                    v=b.valid[k,p:2*p,p:2*p];truth=b.value[k,p:2*p,p:2*p];y=pred[j][k,p:2*p,p:2*p];e=(y-truth)[v]
                    value=e.square().mean();loss=loss+value/(2*c*int(counts[j]))
                    rows.append(dict(sensor=sensor,band=b.name,index=int(ix[k]),mse=float(value.detach()),rmse=float(value.detach().sqrt())*stats[b.name]['std'],bias=float(e.detach().mean())*stats[b.name]['std'],prediction_variance=float(y[v].detach().var(unbiased=False)),target_variance=float(truth[v].var(unbiased=False))))
                    if save_cases:
                        raw_input=b.value[k].detach().clone();raw_input[p:2*p,p:2*p]=0
                        cases[f'{sensor}_{int(ix[k])}']=dict(band=b.name,gsd=g,input=raw_input.cpu(),target=b.value[k].detach().cpu(),prediction=pred[j][k].detach().cpu(),valid=b.valid[k].cpu(),input_sha256=digest(raw_input),output_sha256=digest(pred[j][k]))
            if not torch.isfinite(loss):raise FloatingPointError('A nonfinite loss')
            if backward:loss.backward()
            total+=float(loss.detach())
    return total,rows,cases


def features(items,target,row):
    b=items[target];g=b.native_gsd_m;p=60//g;values=[];valid=[]
    for j,other in enumerate(items):
        if j==target:continue
        x,v=resample_band(other.value[row:row+1],other.valid[row:row+1],other.native_gsd_m,other.phase_xy_m[row:row+1],g,b.phase_xy_m[row:row+1],footprint=SIZE)
        values.append(x[0,p:2*p,p:2*p]);valid.append(v[0,p:2*p,p:2*p]>=.8)
    return torch.stack(values,-1),torch.stack(valid,-1)


def fit_linear(batches):
    fits={}
    for sensor,(items,targets) in batches.items():
        for j,b in enumerate(items):
            xs=[];ys=[];p=60//b.native_gsd_m
            for i in (targets==j).nonzero().flatten().tolist():
                x,v=features(items,j,i);ok=v.all(-1)&b.valid[i,p:2*p,p:2*p]
                xs.append(x[ok].double());ys.append(b.value[i,p:2*p,p:2*p][ok].double())
            x=torch.cat(xs);y=torch.cat(ys)
            if len(x)<2:raise ValueError('insufficient legitimate training regression pixels')
            mean=x.mean(0);ym=y.mean();xc=x-mean
            w=torch.linalg.solve(xc.T@xc/len(x)+.01*torch.eye(x.shape[1],device=x.device,dtype=torch.float64),xc.T@(y-ym)/len(x))
            fits[b.name]=dict(mean=mean,ym=ym,w=w,train_pixels=len(x),lambda_fixed=.01)
    return fits


def controls(batches,fits,stats):
    rows={'spatial':[],'spectral':[]}
    for sensor,(items,targets) in batches.items():
        for i,j in enumerate(targets.tolist()):
            b=items[j];g=b.native_gsd_m;p=60//g;truth=b.value[i,p:2*p,p:2*p];ok=b.valid[i,p:2*p,p:2*p]
            yy,xx=torch.meshgrid(torch.arange(b.value.shape[-2],device=ok.device),torch.arange(b.value.shape[-1],device=ok.device),indexing='ij');xy=torch.stack([yy,xx],-1).float()
            visible=b.valid[i].clone();visible[p:2*p,p:2*p]=False
            source=xy[visible];target=xy[p:2*p,p:2*p].reshape(-1,2)
            distance,index=torch.cdist(target,source).topk(min(4,len(source)),largest=False)
            weight=distance.clamp_min(1e-6).pow(-2);values=b.value[i][visible][index]
            spatial=((values*weight).sum(1)/weight.sum(1)).reshape(p,p)
            x,v=features(items,j,i);fit=fits[b.name];x=torch.where(v,x,fit['mean'].float())
            spectral=((x.double()-fit['mean'])@fit['w']+fit['ym']).float()
            for name,pred in [('spatial',spatial),('spectral',spectral)]:
                e=(pred-truth)[ok];rows[name].append(dict(sensor=sensor,band=b.name,index=i,mse=float(e.square().mean()),rmse=float(e.square().mean().sqrt())*stats[b.name]['std'],bias=float(e.mean())*stats[b.name]['std']))
    return rows


def summary(rows):
    bands={}
    for r in rows:bands.setdefault(r['band'],[]).append(r)
    means={b:sum(r['mse'] for r in rs)/len(rs) for b,rs in bands.items()}
    score=sum(sum(v for b,v in means.items() if b.startswith(s+':'))/sum(b.startswith(s+':') for b in means) for s in ['S2','L8'])/2
    return dict(score=score,bands=means,rows=rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'result.json').exists() or (OUT/'model.pt').exists():raise FileExistsError('preserve A results')
    torch.manual_seed(0);torch.set_num_threads(4);stats=json.loads((RELEASE/'statistics.json').read_text())['bands']
    train,tm=select(NativeSet('train'),stats,24);val,vm=select(NativeSet('val'),stats,16)
    assert not ({r['observation']['group'] for r in tm}&{r['observation']['group'] for r in vm})
    assert not ({r['observation']['acquisition_id'] for r in tm}&{r['observation']['acquisition_id'] for r in vm})
    (OUT/'members.json').write_text(json.dumps(tm+vm,indent=2))
    initial=Path(__file__).parent/'outputs/token_v1/complete_input.pt';state=torch.load(initial,map_location='cuda',weights_only=False)
    model=PatchModel().cuda();model.load_state_dict(state['model'],strict=True);model.eval()
    # A's target-only mask must not remove same-location observations from other bands.
    bs=slice_items(train['S2'][0],torch.tensor([0],device='cuda'));ts=train['S2'][1][:1];h=masks(bs,ts);j=int(ts[0]);p=60//bs[j].native_gsd_m
    with torch.no_grad():
        before=model(bs,h);changed=slice_items(bs,torch.tensor([0],device='cuda'));changed[j].value=changed[j].value.clone();changed[j].value[:,p:2*p,p:2*p]+=10000;after=model(changed,h)
    leak=max(float((a-b).abs().max()) for a,b in zip(before,after));assert leak==0
    fits=fit_linear(train);train_controls={k:summary(v) for k,v in controls(train,fits,stats).items()}
    with torch.no_grad():start_loss,_,_=network(model,train,stats)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0);history=[];start=time.perf_counter()
    tv=TvlpRun(OUT/'tvlp_live',project='conditional-observation',name='A单波段局部遮盖｜实时训练',run_id='token_A_live',seed=0,config=dict(updates=200,train_observations=48,val_observations=32,mask='single-band60m',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),metric_definitions={'train/loss':{'direction':'min'},'train/gradient_norm':{'direction':'none'},'train/lr':{'direction':'none'},'val/score':{'direction':'min'}})
    try:
        for step in range(1,201):
            model.train();optimizer.zero_grad(set_to_none=True);loss,_,_=network(model,train,stats,True);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True);optimizer.step()
            tv.log('train',dict(loss=loss,gradient_norm=float(grad),lr=.001,source_exposure=step*48,epoch_equivalent=step),global_step=step,wall_time_sec=time.perf_counter()-start);tv.heartbeat(step)
            if step==1 or step%20==0:history.append(dict(step=step,loss=loss,gradient=float(grad),lr=.001));print(json.dumps(history[-1]),flush=True)
        model.eval()
        with torch.no_grad():final,rows,cases=network(model,train,stats,save_cases=True)
        tsumm=summary(rows);linear=train_controls['spectral']
        passed=final<=max(.01,1.25*linear['score']) and all(v<=max(.05,1.5*linear['bands'][b]) for b,v in tsumm['bands'].items())
        result=dict(status='A_FIT_PASSED' if passed else 'A_FIT_FAILED_STOPPED',train_model=tsumm,train_controls=train_controls,initial_train=start_loss,history=history,fit_gate=passed,
            gate_rule='mean<=max(.01,1.25*linear_train); every_band<=max(.05,1.5*linear_band_train)',hidden_value_intervention=leak,seconds=time.perf_counter()-start,
            settings=dict(train_observations=48,val_observations=32,seed=0,updates=200,lr=.001,initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),initialization='same complete-input checkpoint as previous B',architecture='unchanged tokenizer/Transformer/linear head',scope='A only; no B/C/cross-source continuation'))
        torch.save(dict(model=model.state_dict(),statistics=stats),OUT/'model.pt');torch.save(dict(train=cases),OUT/'cases.pt')
        if passed:
            with torch.no_grad():vl,vr,vc=network(model,val,stats,save_cases=True)
            result['val_model']=summary(vr);result['val_controls']={k:summary(v) for k,v in controls(val,fits,stats).items()};result['status']='A_COMPLETE_NO_AUTOMATIC_EXTENSION'
            torch.save(dict(train=cases,val=vc),OUT/'cases.pt')
        else:
            # Zero-update path audit only; no additional fitting when the gate fails.
            model.zero_grad(set_to_none=True);network(model,{'S2':train['S2'],'L8':train['L8']},stats,True)
            result['path_audit']={part:sum(float(p.grad.abs().sum()) for p in getattr(model,part).parameters() if p.grad is not None) for part in ['tokenizer','encoder','heads']}
            result['failure']='fixed TRAIN spectral fit failed vs declared gate; inspect spectral path; held-out evaluation and C/cross training stopped'
        (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],train_model=final,train_linear=linear['score'],fit_gate=passed)),flush=True)
        tv.log('final_train',dict(score=tsumm['score'],fit_gate=float(passed)),global_step=200)
        if passed:tv.log('val',dict(score=result['val_model']['score'],spatial_interpolation=result['val_controls']['spatial']['score'],spectral_regression=result['val_controls']['spectral']['score']),global_step=200)
        tv.finish('completed','A bounded trial ended; fitting result in fit_gate; no automatic continuation')
    except BaseException as exc:
        tv.finish('failed',str(exc));raise



if __name__=='__main__':main()
