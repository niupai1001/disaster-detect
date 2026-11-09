"""Bounded, gated native-token reconstruction. Never launches downstream training."""
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F

if __name__=='__main__':torch.cuda.init()

from .model import NativeBandTokenizer, SharedObservationTransformer, BandRaster, band_descriptor
from .data import NativeSet, RELEASE

OUT=Path(__file__).parent/'outputs/token_v1'
SIZE=240


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def patches(x,g):
    p=60//g;n=x.shape[-1]//p
    return x.reshape(len(x),n,p,n,p).permute(0,1,3,2,4).reshape(len(x),n,n,p,p)


def unpatch(x,g):
    b,n,_,p,_=x.shape
    return x.permute(0,1,3,2,4).reshape(b,n*p,n*p)


class IndependentTokenizer(NativeBandTokenizer):
    def forward(self,band,footprint):
        g=band.native_gsd_m;p=60//g;n=footprint//60
        values=patches(band.value,g);valid=patches(band.valid,g);split=p//2
        tokens=[];keeps=[];positions=[];regions=[]
        for ry,rx in [(0,0),(0,1),(1,0),(1,1)]:
            ya,yb=(0,split) if ry==0 else (split,p)
            xa,xb=(0,split) if rx==0 else (split,p)
            region=torch.zeros(p,p,device=values.device,dtype=torch.bool);region[ya:yb,xa:xb]=True;regions.append(region)
            keep=valid&region;v=torch.where(keep,values,0)
            projected=self.projections[str(g)](torch.cat([v.flatten(-2),keep.float().flatten(-2)],-1))
            q=2*ry+rx;tokens.append(projected[...,q*self.width:(q+1)*self.width]);keeps.append(keep.any((-1,-2)))
            positions.append(((xa+xb-1)*g/2,(ya+yb-1)*g/2))
        assert torch.stack(regions).sum(0).eq(1).all()
        t=torch.stack(tokens,3).reshape(len(values),n,n,2,2,self.width).permute(0,1,3,2,4,5).reshape(len(values),2*n,2*n,self.width)
        k=torch.stack(keeps,3).reshape(len(values),n,n,2,2).permute(0,1,3,2,4).reshape(len(values),2*n,2*n)
        xy=torch.stack(torch.meshgrid(torch.arange(n,device=t.device),torch.arange(n,device=t.device),indexing='ij'),-1).flip(-1)*60
        phase=torch.as_tensor(band.phase_xy_m,device=t.device,dtype=t.dtype)
        if phase.ndim==1:phase=phase.expand(len(t),2)
        centers=torch.tensor(positions,device=t.device,dtype=t.dtype)
        xy=(xy[None,:,:,None]+centers[None,None,None]+phase[:,None,None,None])/footprint
        xy=xy.reshape(len(t),n,n,2,2,2).permute(0,1,3,2,4,5).reshape(len(t),2*n,2*n,2)
        pos=torch.cat([xy,xy.sin(),(phase/60)[:,None,None].expand(-1,2*n,2*n,-1)],-1)
        meta=self.position(pos)+self.descriptor(band_descriptor(band.wavelength_nm,band.bandwidth_nm,g,t.device))
        return (t+meta)*k[...,None],k,meta


class PatchModel(nn.Module):
    def __init__(self):
        super().__init__();self.tokenizer=IndependentTokenizer();self.encoder=SharedObservationTransformer()
        self.mask_token=nn.Parameter(torch.zeros(128))
        self.heads=nn.ModuleDict({str(g):nn.Linear(4*128,(60//g)**2) for g in [10,20,30]})

    def forward(self,items,hidden=None):
        footprint=items[0].value.shape[-1]*items[0].native_gsd_m
        if footprint%60 or any(b.value.shape[-2:]!=(footprint//b.native_gsd_m,)*2 for b in items):
            raise ValueError('native arrays must share a declared padded support divisible by60m')
        tokens=[];valid=[];n=footprint//60
        if hidden is not None:
            if hidden.ndim==3:hidden=hidden[:,None].expand(-1,len(items),-1,-1)
            grouped=hidden.reshape(len(hidden),len(items),n,2,n,2)
            if not torch.equal(grouped,grouped[:,:,:,0:1,:,0:1].expand_as(grouped)):
                raise ValueError('Prototype masks whole60m groups of4 tokens; arbitrary30m masks are not aligned on native20m pixels')
        for i,band in enumerate(items):
            if hidden is not None:
                local_hidden=hidden[:,i]
                pixel_mask=F.interpolate(local_hidden[:,None].float(),size=band.value.shape[-2:],mode='nearest')[:,0].bool()
                band=BandRaster(band.name,torch.where(pixel_mask,0,band.value),band.valid&~pixel_mask,band.wavelength_nm,band.bandwidth_nm,band.native_gsd_m,band.phase_xy_m)
            t,k,meta=self.tokenizer(band,footprint)
            if hidden is not None:t=torch.where(local_hidden[...,None],meta+self.mask_token,t);k=k|local_hidden
            tokens.append(t);valid.append(k)
        x=self.encoder(torch.stack(tokens,1),torch.stack(valid,1));out=[]
        for i,band in enumerate(items):
            p=60//band.native_gsd_m
            tile=x[:,i].reshape(len(x),n,2,n,2,128).permute(0,1,3,2,4,5).reshape(len(x),n,n,512)
            out.append(unpatch(self.heads[str(band.native_gsd_m)](tile).reshape(len(x),n,n,p,p),band.native_gsd_m))
        return out


def data_rows(data,stats,count):
    result=[];members=[]
    for sensor in ['S2','L8']:
        chosen=[]
        for index,row in enumerate(data.records[sensor]):
            good=True
            for band in data.metadata[sensor]:
                g=band['native_gsd_m'];a=480//g;b=a+SIZE//g
                if float(data.valid[sensor][band['name']][index,a:b,a:b].float().mean())<.8:good=False;break
            if good:chosen.append(index)
            if len(chosen)==count:break
        if len(chosen)!=count:raise ValueError('not enough predeclared high-valid fixed windows')
        ids=torch.tensor(chosen,device=data.device);items=[]
        for band in data.bands(sensor,ids,stats):
            a=480//band.native_gsd_m;b=a+SIZE//band.native_gsd_m
            items.append(BandRaster(band.name,band.value[:,a:b,a:b],band.valid[:,a:b,a:b],band.wavelength_nm,band.bandwidth_nm,band.native_gsd_m,band.phase_xy_m))
        result.append(items);members.extend(dict(split=data.split,sensor=sensor,observation=data.records[sensor][i],crop_start_m=480,crop_size_m=SIZE) for i in chosen)
    return result,members


def metrics(model,batches,stats,mask):
    total=0.;rows={};outputs={}
    for items in batches:
        h=None if mask is None else mask.expand(len(items[0].value),-1,-1)
        pred=model(items,h);losses=[]
        for band,y in zip(items,pred):
            valid=band.valid
            if h is not None:valid=valid&F.interpolate(h[:,None].float(),size=valid.shape[-2:],mode='nearest')[:,0].bool()
            e=(y-band.value)[valid]
            if not len(e):raise ValueError('empty reconstruction target')
            loss=e.square().mean();losses.append(loss)
            rows[band.name]=dict(mse=float(loss.detach()),rmse_physical=float(loss.detach().sqrt())*stats[band.name]['std'],
                bias_physical=float(e.detach().mean())*stats[band.name]['std'],prediction_variance=float(y[valid].detach().var(unbiased=False)),
                target_variance=float(band.value[valid].var(unbiased=False)),input_sha256=digest(band.value),output_sha256=digest(y))
            outputs[band.name]=dict(input=band.value.detach().cpu(),target=band.value.detach().cpu(),prediction=y.detach().cpu(),valid=band.valid.cpu(),evaluation_valid=valid.cpu())
        loss=sum(losses)/(2*len(items));total=total+loss
    return total,rows,outputs


def structural_checks(model,items):
    result={}
    for g in [10,20,30]:
        v=torch.arange((SIZE//g)**2,device='cuda').reshape(1,SIZE//g,SIZE//g).float()
        assert torch.equal(unpatch(patches(v,g),g),v)
        band=BandRaster('test',v,torch.ones_like(v,dtype=torch.bool),800,30,g,(g/2,g/2))
        a=model.tokenizer(band,SIZE)[0];band.value=band.value.clone();band.value[0,0,0]+=1
        b=model.tokenizer(band,SIZE)[0];changed=(a-b).abs().amax(-1)>1e-6;assert int(changed.sum())==1
        result[str(g)]=dict(changed_tokens=int(changed.sum()),roundtrip_exact=True,subregion_native_pixels=[(60//g)//2,(60//g)-(60//g)//2])
    h=torch.zeros(1,8,8,device='cuda',dtype=torch.bool);h[:,2:4,2:4]=True
    with torch.no_grad():
        a=model(items,h);altered=[]
        for band in items:
            region=F.interpolate(h[:,None].float(),size=band.value.shape[-2:],mode='nearest')[:,0].bool()
            altered.append(BandRaster(band.name,torch.where(region,band.value+10000,band.value),band.valid,band.wavelength_nm,band.bandwidth_nm,band.native_gsd_m,band.phase_xy_m))
        b=model(altered,h)
    diff=max(float((x-y).abs().max()) for x,y in zip(a,b));assert diff==0
    result['hidden_value_intervention_max']=diff
    return result,h


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'result.json').exists():raise FileExistsError('preserve completed prototype; do not silently rerun')
    torch.manual_seed(0);torch.set_num_threads(4)
    stats=json.loads((RELEASE/'statistics.json').read_text())['bands'];train,members=data_rows(NativeSet('train'),stats,2);val,vm=data_rows(NativeSet('val'),stats,1)
    (OUT/'members.json').write_text(json.dumps(members+vm,indent=2));model=PatchModel().cuda();model.eval();checks,mask=structural_checks(model,[BandRaster(b.name,b.value[:1],b.valid[:1],b.wavelength_nm,b.bandwidth_nm,b.native_gsd_m,b.phase_xy_m[:1]) for b in train[0]])
    (OUT/'structural_checks.json').write_text(json.dumps(checks,indent=2));stages={};reason=''
    for name,hidden in [('complete_input',None),('fixed_mask',mask)]:
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0)
        with torch.no_grad():initial,_,_=metrics(model,train,stats,hidden)
        history=[];start=time.perf_counter()
        for step in range(1,201):
            model.train();optimizer.zero_grad(set_to_none=True);loss,_,_=metrics(model,train,stats,hidden)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True);optimizer.step()
            if step==1 or step%20==0:
                history.append(dict(step=step,loss=float(loss.detach()),gradient=float(grad),lr=1e-3));print(json.dumps(dict(stage=name,**history[-1])),flush=True)
        model.eval()
        with torch.no_grad():loss,bands,outputs=metrics(model,train,stats,hidden);vl,vbands,voutputs=metrics(model,val,stats,hidden)
        passed=float(loss)<.01 and max(x['mse'] for x in bands.values())<.05 if hidden is None else float(loss)<float(initial)*.5 and all(math_row['prediction_variance']>1e-6 for math_row in bands.values())
        stages[name]=dict(passed=passed,initial=float(initial),final=float(loss),validation=float(vl),bands=bands,validation_bands=vbands,history=history,seconds=time.perf_counter()-start)
        torch.save(dict(model=model.state_dict(),statistics=stats),OUT/(name+'.pt'));torch.save(dict(train=outputs,val=voutputs),OUT/(name+'_cases.pt'))
        (OUT/(name+'.json')).write_text(json.dumps(stages[name],indent=2))
        if not passed:reason=name+' failed its fixed learning gate; later stages stopped';break
    result=dict(status='PROTOTYPE_PASSED' if not reason else 'STOPPED_AT_FAILED_GATE',reason=reason,stages=stages,checks=checks,
        settings=dict(seed=0,footprint_m=SIZE,train_observations=4,val_observations=2,max_updates_per_gate=200,mask_patch_fraction=1/16,mask_tokens_all_bands=True),
        copy_baseline='External measured-value copy has zero error by definition; model output never splices input values; no hidden truth in encoder',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],reason=reason)),flush=True)


if __name__=='__main__':main()
