"""One corrective scale experiment: explicit grids and input-derived weights."""
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from src.augment.spatial_degradation import lowpass


@contextmanager
def auxiliary_batch_statistics(model):
    """Use batch statistics without changing BN running buffers on extra forwards."""
    layers=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    previous=[m.track_running_stats for m in layers]
    try:
        for m in layers:m.track_running_stats=False
        yield
    finally:
        for m,value in zip(layers,previous):m.track_running_stats=value


def coarse_view(image,valid):
    # Same footprint, 10m -> 20m sampling; do not upsample back before the encoder.
    weights=lowpass(valid.float(),1.)
    smooth=lowpass(image*valid,1.)/weights.clamp_min(1e-6)
    coarse=F.avg_pool2d(smooth,2,2)
    good=F.avg_pool2d(valid.float(),2,2)>=.999
    return coarse*good,good


def input_support_weights(image,coarse,valid,shape,mode,generator):
    """Choose half the common valid blocks; random keeps identical weight mass."""
    with torch.no_grad():
        fraction=F.adaptive_avg_pool2d(valid.float(),shape)
        restored=F.interpolate(coarse,size=image.shape[-2:],mode="bilinear",align_corners=False)
        error=F.adaptive_avg_pool2d((image-restored).square().mean(1,keepdim=True),shape)
        energy=F.adaptive_avg_pool2d(image.square().mean(1,keepdim=True),shape).clamp_min(1e-6)
        smooth=1/(1+error/energy)
        weights=torch.zeros_like(fraction)
        for i in range(len(image)):
            eligible=(fraction[i].flatten()>=.8).nonzero().flatten()
            if not len(eligible):continue
            number=max(1,(len(eligible)+1)//2)
            scores=smooth[i].flatten()[eligible]
            selected=eligible[torch.argsort(scores,descending=True)[:number]]
            mass=(fraction[i].flatten()[selected]*smooth[i].flatten()[selected])
            if mode=="selective":
                weights[i].flatten()[selected]=mass
            elif mode=="random":
                ids=eligible[torch.randperm(len(eligible),device=image.device,generator=generator)[:number]]
                weights[i].flatten()[ids]=mass
            elif mode in ("uniform","fixed"):
                v=fraction[i].flatten()[eligible]
                weights[i].flatten()[eligible]=v*(mass.sum()/v.sum())
            else:raise ValueError(mode)
        return weights.detach()


def common_scale_loss(high,low,image,coarse,valid,mode,generator):
    # s16 high cells span 160m, low cells span 320m. Aggregate only the high side.
    aligned=F.adaptive_avg_pool2d(high,low.shape[-2:])
    similarity=F.cosine_similarity(aligned,low,dim=1).unsqueeze(1)
    weights=input_support_weights(image,coarse,valid,low.shape[-2:],mode,generator)
    loss=(weights*(1-similarity)).sum()/weights.sum().clamp_min(1e-6)
    return loss,dict(weight_sum=weights.sum(),active_blocks=(weights>0).sum())
