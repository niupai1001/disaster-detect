"""GPU sampling with declared native pixel centers and physical area support."""
import torch
from torch.nn import functional as F


def area_weights(input_count,output_count,input_gsd,output_gsd,input_phase,output_phase):
    source=torch.arange(input_count,device=input_phase.device)*input_gsd+input_phase[:,None]
    target=torch.arange(output_count,device=input_phase.device)*output_gsd+output_phase[:,None]
    left=torch.maximum(source[:,None]-input_gsd/2,target[:,:,None]-output_gsd/2)
    right=torch.minimum(source[:,None]+input_gsd/2,target[:,:,None]+output_gsd/2)
    return (right-left).clamp_min(0)/output_gsd


def resample_band(value,valid,source_gsd,source_phase,target_gsd,target_phase,footprint=1200):
    """Returns conditional input estimate and coverage, never invented target truth."""
    count=footprint//target_gsd
    if target_gsd>=source_gsd:
        wx=area_weights(value.shape[-1],count,source_gsd,target_gsd,source_phase[:,0],target_phase[:,0])
        wy=area_weights(value.shape[-2],count,source_gsd,target_gsd,source_phase[:,1],target_phase[:,1])
        numerator=torch.bmm(torch.bmm(wy,value*valid),wx.transpose(1,2))
        coverage=torch.bmm(torch.bmm(wy,valid.float()),wx.transpose(1,2))
    else:
        xy=torch.arange(count,device=value.device)*target_gsd
        xx=(xy[None]+target_phase[:,0,None]-source_phase[:,0,None])/source_gsd
        yy=(xy[None]+target_phase[:,1,None]-source_phase[:,1,None])/source_gsd
        gx=(2*(xx+.5)/value.shape[-1]-1)[:,None].expand(-1,count,-1)
        gy=(2*(yy+.5)/value.shape[-2]-1)[:,:,None].expand(-1,-1,count)
        grid=torch.stack([gx,gy],-1)
        numerator=F.grid_sample((value*valid)[:,None],grid,align_corners=False)[:,0]
        coverage=F.grid_sample(valid.float()[:,None],grid,align_corners=False)[:,0]
    return numerator/coverage.clamp_min(1e-6),coverage


def observable_10_to_30(value,valid,source_phase,target_phase):
    coords=torch.arange(-3,4,device=value.device,dtype=torch.float32)
    kernel=torch.exp(-.5*coords.square());kernel=kernel/kernel.sum()
    kernel=(kernel[:,None]*kernel[None])[None,None]
    numerator=F.conv2d((value*valid)[:,None],kernel,padding=3)[:,0]
    coverage=F.conv2d(valid.float()[:,None],kernel,padding=3)[:,0]
    blurred=numerator/coverage.clamp_min(1e-6)
    return resample_band(blurred,coverage,10,source_phase,30,target_phase,footprint=value.shape[-1]*10)
