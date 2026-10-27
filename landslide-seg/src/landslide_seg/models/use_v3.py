"""USE v3: shared physical spectral frontend, bidirectional spatial features."""
import torch
from torch import nn
from torch.nn import functional as F
from .use_v2 import USEv2, attend


class SpatialBlock(nn.Module):
    def __init__(self,width,shifted=False,global_attention=False):
        super().__init__()
        self.shifted=shifted
        self.global_attention=global_attention
        self.norm=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width)
        self.proj=nn.Linear(width,width)
        self.ffn=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
        self.relative_bias=nn.Parameter(torch.zeros(4,49))
        coordinates=torch.stack(torch.meshgrid(torch.arange(4),torch.arange(4),indexing='ij')).flatten(1)
        delta=coordinates[:,:,None]-coordinates[:,None,:]+3
        self.register_buffer('relative_index',delta[0]*7+delta[1])

    def forward(self,x,valid):
        b,d,h,w=x.shape
        offset=2 if self.shifted and not self.global_attention else 0
        if not self.global_attention:
            x=F.pad(x,(offset,(-w-offset)%4,offset,(-h-offset)%4))
            valid=F.pad(valid,(offset,(-w-offset)%4,offset,(-h-offset)%4))
        hp,wp=x.shape[-2:]
        if self.global_attention:
            t=x.flatten(2).transpose(1,2); v=valid.flatten(1)
        else:
            t=x.permute(0,2,3,1).reshape(b,hp//4,4,wp//4,4,d).permute(0,1,3,2,4,5).reshape(-1,16,d)
            v=valid.reshape(b,hp//4,4,wp//4,4).permute(0,1,3,2,4).reshape(-1,16)
        q,k,value=self.qkv(self.norm(t)).reshape(len(t),t.shape[1],3,4,d//4).permute(2,0,3,1,4).unbind(0)
        safe=v.clone(); safe[~v.any(-1),0]=True
        bias=t.new_zeros(len(t),4,t.shape[1],t.shape[1])
        if not self.global_attention:
            bias=bias+self.relative_bias[:,self.relative_index][None]
        bias=bias.masked_fill(~safe[:,None,None,:],float('-inf'))
        z=F.scaled_dot_product_attention(q,k,value,attn_mask=bias)
        t=t+self.proj(z.transpose(1,2).reshape_as(t))
        t=(t+self.ffn(t)).masked_fill(~v[...,None],0)
        if self.global_attention:
            return t.transpose(1,2).reshape(b,d,h,w)
        return t.reshape(b,hp//4,wp//4,4,4,d).permute(0,5,1,3,2,4).reshape(b,d,hp,wp)[:,:,offset:offset+h,offset:offset+w]


class USEv3(USEv2):
    def __init__(self):
        super().__init__(attention=True)
        self.stages=nn.ModuleList([nn.ModuleList([SpatialBlock(d,i%2==1,stage==2 and i==depth-1)
                                  for i in range(depth)])
                                  for stage,(d,depth) in enumerate(zip((32,64,128,256),(2,2,4,2)))])
        self.inject=nn.Conv2d(32,32,1)
        self.inject_gate=nn.Parameter(torch.tensor(.01))
        self.projections=nn.ModuleList([nn.Conv2d(d,32,1) for d in (32,64,128,256)])
        self.feedback_gate=nn.Parameter(torch.tensor(.01))
        self.local=nn.Sequential(nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,32,1))

    def forward(self, observation, return_pyramid=False):
        allowed = {'x','pixel_valid','band_valid','band_metadata','physical','known_mask','band_ids','spatial_metadata'}
        if set(observation)-allowed:
            raise ValueError('Unexpected input; labels cannot enter USE')
        x = observation['x']
        if x.ndim != 4 or not x.is_floating_point() or not 1 <= x.shape[1] <= 16:
            raise ValueError('Expected floating B,C,H,W with 1..16 channels')
        b,c,h,w = x.shape
        valid = observation['pixel_valid']
        band = observation.get('band_valid',torch.ones(b,c,dtype=torch.bool,device=x.device))
        if valid.shape != x.shape or valid.dtype != torch.bool or band.shape != (b,c) or band.dtype != torch.bool:
            raise ValueError('Invalid quality masks')
        valid = valid & band[:,:,None,None]
        if not valid.flatten(1).any(-1).all() or not torch.isfinite(x[valid]).all():
            raise ValueError('Empty observation or nonfinite valid values')
        x = torch.where(valid,x,0)
        physical = observation.get('physical',observation.get('band_metadata'))
        known = observation.get('known_mask')
        if physical is None:
            physical = x.new_zeros(b,c,3)
            known = torch.zeros_like(physical,dtype=torch.bool)
        if known is None or physical.shape != (b,c,3) or known.shape != physical.shape or known.dtype != torch.bool:
            raise ValueError('Physical metadata requires B,C,3 known mask')
        if not torch.isfinite(physical).all() or (physical[~known]!=0).any() or (physical[known]<=0).any():
            raise ValueError('Invalid physical units/placeholders')
        wave = physical[...,0][known[...,0]]
        if ((wave<.1)|(wave>3)).any():
            raise ValueError('Wavelength must be micrometres in .1..3')
        condition = self.physical(torch.cat((torch.log1p(physical),known.float()),-1))
        condition = condition * known.any(-1,keepdim=True)
        xp = F.pad(x,(0,w%2,0,h%2)); vp = F.pad(valid,(0,w%2,0,h%2))
        hp,wp = xp.shape[-2]//2,xp.shape[-1]//2
        def patches(t):
            return t.unfold(2,2,2).unfold(3,2,2).permute(0,2,3,1,4,5).reshape(b,hp*wp,c,4)
        values, masks = patches(xp), patches(vp)
        support = masks.any(-1).reshape(-1,c)
        t = self.tokenizer(torch.cat((values,masks.float()),-1)) + condition[:,None]
        t = t.reshape(-1,c,32).masked_fill(~support[...,None],0)
        if self.spectral is not None:
            z = self.spectral_norm(t)
            t = t + attend(self.spectral,z,z,support)
        t = (t+self.ffn(t)).masked_fill(~support[...,None],0)
        q = self.queries[None].expand(len(t),-1,-1)
        output = attend(self.readout,q,t,support).reshape(b,hp*wp,5,32)
        r = self.residual(torch.stack((values,masks.float()),-1))
        r = (r*masks[...,None]).sum(2)/masks.sum(2).clamp_min(1)[...,None]
        r = self.residual_out(r)*masks.any(2)[...,None]
        fine = output[:,:,:4] + r
        fine = fine.reshape(b,hp,wp,2,2,32).permute(0,5,1,3,2,4).reshape(b,32,hp*2,wp*2)[:,:,:h,:w]
        fine_valid = valid.any(1)
        fine = fine.masked_fill(~fine_valid[:,None],0)
        context = output[:,:,4].transpose(1,2).reshape(b,32,hp,wp)
        cv = masks.any((2,3)).reshape(b,hp,wp)
        count = F.avg_pool2d(F.pad(fine_valid[:,None].float(),(0,w%2,0,h%2)),2)
        pooled = F.avg_pool2d(F.pad(fine,(0,w%2,0,h%2)),2)/count.clamp_min(1e-8)
        context = context + self.inject_gate*self.inject(pooled)
        features, supports = [],[]
        for merge,stage in zip(self.merges,self.stages):
            context,cv = merge(context,cv)
            for block in stage:
                context = block(context,cv)
            features.append(context); supports.append(cv)
        main = self.projections[-1](features[-1])
        for i in (2,1,0):
            main = self.projections[i](features[i]) + F.interpolate(main,size=features[i].shape[-2:],mode='bilinear',align_corners=False)
            main = main.masked_fill(~supports[i][:,None],0)
        fine = fine + self.feedback_gate*F.interpolate(main,size=(h,w),mode='bilinear',align_corners=False)
        fine = (fine + self.local(fine)).masked_fill(~fine_valid[:,None],0)
        result = dict(main=main,fine=fine,valid=dict(main=supports[0],fine=fine_valid),
                      stride=dict(main=4,fine=1),spatial_metadata=observation.get('spatial_metadata'),
                      metadata_used=known & valid.flatten(2).any(-1)[...,None])
        if return_pyramid:
            result['context']=features
            result['valid']['context']=supports
            result['stride']['context']=[4,8,16,32]
        return result

