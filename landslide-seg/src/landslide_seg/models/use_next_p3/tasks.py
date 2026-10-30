import math
import copy
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from landslide_seg.models.use_next_p2.tasks import ScaleTask,masks,patch_targets
from landslide_seg.models.use_next_p2.encoder import geometry


def band_view(asset,seed):
    count=len(asset.image)
    if asset.modality not in ('s2','landsat') or count<3:return asset
    rng=np.random.default_rng(seed)
    if rng.random()>=.5:return asset
    drop=int(round(count*rng.uniform(.2,.4)))
    drop=max(math.ceil(.2*count),min(math.floor(.4*count),drop))
    ids=sorted(rng.choice(count,size=count-drop,replace=False).tolist());result=copy.copy(asset)
    for key in ('image','pixel_valid','channel_metadata','metadata_known'):setattr(result,key,getattr(asset,key)[ids])
    result.channel_names=tuple(asset.channel_names[i] for i in ids)
    for key in ('native_resolution','effective_resolution'):
        if getattr(asset,key) is not None:setattr(result,key,tuple(getattr(asset,key)[i] for i in ids))
    return result


class LatentTask(ScaleTask):
    def __init__(self,root,mode='physical',objective='joint'):
        super().__init__(root,mode)
        if mode!='physical' or objective not in ('scale','latent','joint'):raise ValueError('Invalid P3 objective')
        self.objective=objective
        torch.manual_seed(1235)
        self.latent_predictor=nn.Sequential(nn.Linear(194,192),nn.GELU(),nn.Linear(192,192))

    def task_parameters(self):return list(self.predictor.parameters())+list(self.latent_predictor.parameters())

    @torch.no_grad()
    def after_optimizer(self,step,total):
        momentum=1-(1-.996)*(.5*(1+math.cos(math.pi*(step+1)/total)))
        self.use.ema(momentum)

    def loss(self,asset,spatial_seed,band_seed,calibration):
        scale=None;target_scale=None;scale_reason=None
        hidden,grid=masks(asset,spatial_seed)
        if self.objective in ('scale','joint'):
            target_scale,scale_reason=patch_targets(asset,grid,calibration)
        if self.objective=='scale' and target_scale is None:return None,dict(reason=scale_reason,modality=asset.modality)
        # Frozen native levels are not consumed by these local SSL losses; no unused native forward.
        view=band_view(asset,band_seed)
        student=self.use.local(view,hidden_cells=hidden,grid=grid)
        context=dict(hidden_cells=int(hidden[::4,::4].sum()),visible_bands=list(view.channel_names),
                     true_valid_channel_pixels=int(asset.pixel_valid.sum()))
        if student.global_embedding is None:return None,dict(reason='NO_VISIBLE_CONTEXT',modality=asset.modality)
        if target_scale is not None:
            values,support=target_scale;c,ph,pw=values.shape
            yy,xx=torch.meshgrid(torch.arange(ph,device=values.device),torch.arange(pw,device=values.device),indexing='ij')
            query=torch.stack((xx+.5,yy+.5),-1).float()
            metadata=torch.cat((torch.where(asset.metadata_known,asset.channel_metadata,0),asset.metadata_known.float()),-1)
            decoder=torch.cat((student.global_embedding[None,None,None].expand(c,ph,pw,-1),
                               query[None].expand(c,-1,-1,-1),metadata[:,None,None].expand(-1,ph,pw,-1)),-1)
            prediction=self.predictor(decoder).squeeze(-1)
            weights=support*hidden[::4,::4].to(support.device)[None]
            if weights.sum()>0:scale=((prediction-values.detach()).square()*weights).sum()/weights.sum()
        if self.objective=='scale':
            return scale,dict(modality=asset.modality,scale_applied=scale is not None,latent_applied=False,
                              loss=float(scale.detach()) if scale is not None else None,high_frequency='NOT_AVAILABLE_EFFECTIVE_SUPPORT_UNKNOWN',**context)
        with torch.no_grad():teacher=self.use.teacher_local(asset,grid=grid)
        target=teacher.metric_patch.tensor.permute(1,2,0).detach();ph,pw,_=target.shape
        matrix,shape,_,_=geometry(asset,grid);matrix=matrix.to(target.device)
        area=torch.sparse.mm(matrix,asset.pixel_valid.any(0).float().flatten()[:,None])[:,0]
        area=area.reshape(ph,4,pw,4).sum((1,3))
        weights=area*teacher.metric_patch.valid_mask*hidden[::4,::4].to(target.device)
        if weights.sum()==0:return None,dict(reason='NO_LATENT_TARGET_SUPPORT',modality=asset.modality)
        supported=target[teacher.metric_patch.valid_mask]
        variance=float((supported-supported.mean(0)).square().mean())
        if not torch.isfinite(target).all() or variance<1e-8:raise ValueError('Degenerate or nonfinite EMA spatial target')
        yy,xx=torch.meshgrid(torch.arange(ph,device=target.device),torch.arange(pw,device=target.device),indexing='ij')
        query=torch.stack((xx+.5,yy+.5),-1).float()
        prediction=self.latent_predictor(torch.cat((student.global_embedding[None,None].expand(ph,pw,-1),query),-1))
        latent=((F.layer_norm(prediction,(192,))-F.layer_norm(target,(192,))).square().mean(-1)*weights).sum()/weights.sum()
        total=latent+(scale if scale is not None else 0)
        return total,dict(modality=asset.modality,scale_applied=scale is not None,latent_applied=True,
                          scale_loss=float(scale.detach()) if scale is not None else None,latent_loss=float(latent.detach()),
                          target_spatial_variance=variance,teacher_detached=not target.requires_grad,loss=float(total.detach()),
                          high_frequency='NOT_AVAILABLE_EFFECTIVE_SUPPORT_UNKNOWN',latent_target_area_m2=float(weights.sum()),**context)
