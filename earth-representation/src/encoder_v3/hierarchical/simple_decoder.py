"""Fixed local interpolation plus a small, band-conditioned bilinear residual head."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from scipy.spatial import cKDTree


def nearest_weights(source,query):
    distance,index=cKDTree(source).query(query,k=min(8,len(source)),workers=1)
    if distance.ndim==1:distance,index=distance[:,None],index[:,None]
    weight=1/np.maximum(distance,1e-12)**2
    return index,weight/weight.sum(1,keepdims=True)


def native_stencil(grid,keep,target,device):
    source=np.flatnonzero(keep);query=np.flatnonzero(target)
    tensor=lambda a,dtype:torch.as_tensor(a,dtype=dtype,device=device)
    if not len(source):return tensor(query,torch.long),None,None
    a,_,_,_,e,_=grid.affine;w=grid.shape[1]
    def points(ids):return np.column_stack(((ids%w+.5)*a,(ids//w+.5)*e))
    index,weight=nearest_weights(points(source),points(query))
    return tensor(query,torch.long),tensor(source[index],torch.long),tensor(weight,torch.float32)


def visible_condition(memory,numeric,truth,keep,empty=False):
    count=keep.sum().clamp_min(1)
    mean=torch.where(keep[None],truth,0.).flatten(1).sum(1)/count
    centered=torch.where(keep[None],truth-mean[:,None,None],0.)
    variance=centered.square().flatten(1).sum(1)/count
    scale=torch.where(variance>0,variance.clamp_min(1e-12).sqrt(),0.)
    if empty:scale=torch.ones_like(scale)
    return torch.cat((memory,numeric,mean[:,None],torch.log1p(scale)[:,None]),-1),scale


class NativeResidualDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.ModuleList(nn.Linear(w,16) for w in (64,96,128))
        self.norm=nn.LayerNorm(16)
        self.condition=nn.Sequential(nn.Linear(46,32),nn.GELU(),nn.Linear(32,32))
        self.output=nn.Linear(16,4)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def context(self,encoded,valids,layout):
        fine_to_middle=layout[4][0][2];middle_to_coarse=layout[4][1][2]
        value=self.projection[0](encoded[0])+self.projection[1](encoded[1])[fine_to_middle]
        value=value+self.projection[2](encoded[2])[middle_to_coarse[fine_to_middle]]
        value=value.float()
        # Hidden cells read actual visible fine features directly, without learned attention.
        visible=valids[0].detach().cpu().numpy();source=np.flatnonzero(visible);query=np.flatnonzero(~visible)
        if not len(source):raise ValueError('NO_VISIBLE_FINE_CONTEXT')
        if len(query):
            index,weights=nearest_weights(layout[1][0].centers[source],layout[1][0].centers[query])
            ids=torch.as_tensor(source[index],device=value.device);weights=value.new_tensor(weights)
            pieces=[]
            for start in range(0,len(query),8192):
                def gather(x,i,w):return (x[i]*w[...,None]).sum(1)
                args=(value,ids[start:start+8192],weights[start:start+8192])
                pieces.append(checkpoint(gather,*args,use_reentrant=False) if torch.is_grad_enabled() else gather(*args))
            value=value.index_copy(0,torch.as_tensor(query,device=value.device),torch.cat(pieces))
        return self.norm(value)

    def coefficients(self,context,condition,needed):
        gamma,beta=self.condition(condition).chunk(2,dim=-1)
        def head(x,g,b):return self.output(F.gelu(x[None]*(1+g[:,None])+b[:,None]))
        pieces=[]
        for ids in needed.split(8192):
            args=(context[ids],gamma,beta)
            pieces.append(checkpoint(head,*args,use_reentrant=False) if torch.is_grad_enabled() else head(*args))
        result=torch.cat(pieces,1)
        return result.new_zeros(len(condition),len(context),4).index_copy(1,needed,result)

    @staticmethod
    def integrate(coefficients,mapping,grid,geo,part,target):
        """Exact area mean of each region's bilinear function on rectangular overlaps."""
        coefficients=coefficients.float()  # Geometry and native measurement integration stay FP32 under autocast.
        single=coefficients.ndim==2
        if single:coefficients=coefficients[None]
        w=grid.shape[1];a,_,ox,_,e,oy=grid.affine
        x=coefficients.new_tensor(part.x);y=coefficients.new_tensor(part.y);nx=len(part.x)-1
        selected=target.flatten()[mapping.source]
        source=mapping.source[selected];destination=mapping.destination[selected];area=mapping.area[selected]
        result=coefficients.new_zeros(len(coefficients),mapping.pixels)
        for begin in range(0,len(source),65536):
            s=source[begin:begin+65536];d=destination[begin:begin+65536];ix=d%nx;iy=d//nx
            left=(ox-geo.origin[0])+(s%w)*a;right=left+a
            top=(oy-geo.origin[1])+(s//w)*e;bottom=top+e
            mx=(torch.maximum(left,x[ix])+torch.minimum(right,x[ix+1]))*.5
            my=(torch.maximum(bottom,y[iy])+torch.minimum(top,y[iy+1]))*.5
            dx=(mx-(x[ix]+x[ix+1])*.5)/(x[ix+1]-x[ix])
            dy=(my-(y[iy]+y[iy+1])*.5)/(y[iy+1]-y[iy])
            basis=torch.stack((torch.ones_like(dx),dx,dy,dx*dy),-1)
            value=(coefficients[:,d]*basis[None]).sum(-1)*area[begin:begin+65536]/mapping.pixel_area
            result.index_add_(1,s,value)
        return result[0].reshape(grid.shape) if single else result.reshape(len(coefficients),*grid.shape)

    def forward(self,sample,encoded,valids,conditions,layout,keeps,targets,return_predictions):
        context=self.context(encoded,valids,layout)
        numeric=context.new_tensor([c.physical.encoded()[0] for c in sample.channels])
        # Batch the mask transfer per native grid; reuse exact stencils across equal band masks.
        groups={}
        for i,c in enumerate(sample.channels):groups.setdefault(c.grid,[]).append(i)
        predictions=[None]*len(keeps);losses=[];baseline_losses=[];fallback=0
        for grid,indices in groups.items():
            masks=torch.stack([torch.stack((keeps[i],targets[i])) for i in indices]).cpu().numpy()
            grouped={}
            for j,i in enumerate(indices):
                keep,target=masks[j]
                if not target.any():
                    if return_predictions:
                        predictions[i]=torch.where(keeps[i],sample.channels[i].values,0.)
                    continue
                key=np.packbits(masks[j]).tobytes()
                grouped.setdefault(key,[]).append(i)
            for members in grouped.values():
                first=members[0];keep=keeps[first];target=targets[first];mapping=layout[2][first][0]
                query,source,weight=native_stencil(grid,keep.cpu().numpy(),target.cpu().numpy(),context.device)
                needed=torch.unique(mapping.destination[target.flatten()[mapping.source]])
                capacity=max(1,1048576//max(len(needed)*16,len(query)*8))
                for begin in range(0,len(members),capacity):
                    ids=members[begin:begin+capacity];truth=torch.stack([sample.channels[i].values for i in ids])
                    baseline=torch.where(keep[None],truth,0.).flatten(1)
                    values=baseline.new_zeros(len(ids),len(query)) if source is None else (baseline[:,source]*weight[None]).sum(-1)
                    baseline=baseline.index_copy(1,query,values).reshape(len(ids),*grid.shape)
                    condition,scale=visible_condition(conditions['output'][ids],numeric[ids],truth,keep,source is None)
                    coefficients=self.coefficients(context,condition,needed)
                    correction=self.integrate(coefficients,mapping,grid,layout[0],layout[1][0],target)
                    # Attenuate corrections in near-constant scenes without a sensor-specific threshold.
                    gain=scale if source is None else scale.square()/(1+scale)
                    correction=correction*gain[:,None,None]
                    prediction=baseline+correction
                    expected=torch.where(target[None],truth,0.)
                    losses.extend(((torch.where(target[None],prediction,0.)-expected).square().flatten(1).sum(1)/target.sum()).unbind())
                    baseline_losses.extend(((torch.where(target[None],baseline,0.)-expected).square().flatten(1).sum(1)/target.sum()).unbind())
                    if return_predictions:
                        for i,value in zip(ids,prediction.unbind()):predictions[i]=value
                    if source is None:fallback+=len(ids)
        return predictions,losses,baseline_losses,fallback
