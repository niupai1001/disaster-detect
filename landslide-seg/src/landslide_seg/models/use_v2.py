"""Variable-band patch encoder. Quality and artificial masking precede every path."""
import torch
from torch import nn
from torch.nn import functional as F


def attend(module, query, tokens, valid, mask=None):
    # A zero placeholder prevents NaN for empty windows; its output is discarded.
    empty = ~valid.any(-1)
    safe = valid.clone()
    safe[empty, 0] = True
    tokens = torch.where(valid[..., None], tokens, 0)
    output = module(query, tokens, tokens, key_padding_mask=~safe,
                    attn_mask=mask, need_weights=False)[0]
    return output.masked_fill(empty[:, None, None], 0)


class WindowBlock(nn.Module):
    def __init__(self, width, shifted=False):
        super().__init__()
        self.shifted = shifted
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*2),
                                 nn.GELU(), nn.Linear(width*2, width))

    def forward(self, x, valid):
        b, d, h, w = x.shape
        # Offset padding implements shifted windows without cyclic wraparound.
        offset = 2 if self.shifted else 0
        x = F.pad(x, (offset, (-w-offset)%4, offset, (-h-offset)%4))
        valid = F.pad(valid, (offset, (-w-offset)%4, offset, (-h-offset)%4))
        hp, wp = x.shape[-2:]
        t = x.permute(0,2,3,1).reshape(b,hp//4,4,wp//4,4,d)
        t = t.permute(0,1,3,2,4,5).reshape(-1,16,d)
        v = valid.reshape(b,hp//4,4,wp//4,4).permute(0,1,3,2,4).reshape(-1,16)
        z = self.norm(t)
        t = t + attend(self.attn, z, z, v)
        t = (t + self.ffn(t)).masked_fill(~v[...,None], 0)
        x = t.reshape(b,hp//4,wp//4,4,4,d).permute(0,1,3,2,4,5)
        return x.reshape(b,hp,wp,d).permute(0,3,1,2)[:,:,offset:offset+h,offset:offset+w]


class Merge(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.proj = nn.Conv2d(incoming, outgoing, 2, stride=2)

    def forward(self, x, valid):
        pad = (0,x.shape[-1]%2,0,x.shape[-2]%2)
        x = self.proj(F.pad(x.masked_fill(~valid[:,None],0),pad))
        valid = F.max_pool2d(F.pad(valid[:,None].float(),pad),2,2)[:,0].bool()
        return x.masked_fill(~valid[:,None],0), valid


class USEv2(nn.Module):
    def __init__(self, attention=True):
        super().__init__()
        self.tokenizer = nn.Sequential(nn.Linear(8,32),nn.LayerNorm(32))
        self.physical = nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,32))
        self.spectral_norm = nn.LayerNorm(32)
        self.spectral = nn.MultiheadAttention(32,4,batch_first=True) if attention else None
        self.ffn = nn.Sequential(nn.LayerNorm(32),nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))
        self.queries = nn.Parameter(torch.randn(5,32)*.02)
        self.readout = nn.MultiheadAttention(32,4,batch_first=True)
        self.residual = nn.Sequential(nn.Linear(2,8),nn.GELU())
        self.residual_out = nn.Linear(8,32)
        self.merges = nn.ModuleList([Merge(32,32),Merge(32,64),Merge(64,128),Merge(128,256)])
        self.stages = nn.ModuleList([nn.ModuleList([WindowBlock(d, i%2==1) for i in range(depth)])
                                    for d,depth in zip((32,64,128,256),(1,1,2,1))])

    def forward(self, observation):
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
        features, supports = [],[]
        for merge,stage in zip(self.merges,self.stages):
            context,cv = merge(context,cv)
            for block in stage:
                context = block(context,cv)
            features.append(context); supports.append(cv)
        return {'fine':fine,'context':features,'valid':{'fine':fine_valid,'context':supports},
                'metadata_used':known & valid.flatten(2).any(-1)[...,None]}


def initialize_pair(seed=1234):
    torch.manual_seed(seed)
    reference = USEv2(True)
    torch.manual_seed(seed)
    control = USEv2(False)
    control.load_state_dict({k:v for k,v in reference.state_dict().items() if k in control.state_dict()},strict=True)
    return control,reference


class FrozenReadout(nn.Module):
    """Same small spatial head for frozen reconstruction and segmentation."""
    def __init__(self, encoder, fine=True, outputs=1):
        super().__init__()
        from .frozen_use import FrozenUSE
        self.use = FrozenUSE(encoder)
        self.projections = nn.ModuleList([nn.Conv2d(32,32,1) for _ in range(4)])
        self.output = nn.Conv2d(32,outputs,1)
        self.fine = nn.Conv2d(32,32,1) if fine else None

    def forward(self, observation):
        features = self.use(observation)
        size = observation['x'].shape[-2:]
        quarter=features['context'][0].shape[-2:]
        x = sum(F.interpolate(p(canonical_context(t)),size=quarter,mode='bilinear',align_corners=False)
                for p,t in zip(self.projections,features['context']))
        x=F.interpolate(x,size=size,mode='bilinear',align_corners=False)
        if self.fine is not None:
            x = x+self.fine(features['fine'])
        return self.output(x)


def canonical_context(tensor):
    """Deterministic channel grouping gives all encoders identical probe capacity."""
    b,d,h,w=tensor.shape
    return tensor.reshape(b,32,d//32,h,w).mean(2)


class LegacyPyramid(nn.Module):
    """Old selected USE, with explicitly deterministic pooled context scales."""
    def __init__(self,encoder):
        super().__init__()
        self.legacy=encoder

    def forward(self,observation):
        output=self.legacy(observation)
        fine=output['features']; valid=output['valid']
        context=[]; masks=[]
        for scale in (4,8,16,32):
            count=F.avg_pool2d(valid[:,None].float(),scale,scale,ceil_mode=True)
            feature=F.avg_pool2d(fine,scale,scale,ceil_mode=True)/count.clamp_min(1e-8)
            context.append(feature.masked_fill(count==0,0)); masks.append(count[:,0]>0)
        return dict(fine=fine,context=context,valid=dict(fine=valid,context=masks),metadata_used=observation['known_mask'])
