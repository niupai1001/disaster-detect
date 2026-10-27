"""Native-scale reconstruction and identical frozen consumers for USE v2/v3."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .frozen_use import FrozenUSE as LegacyFrozenUSE, state_sha256


def packed_state_sha256(module):
    tensors=sorted([('parameter.'+n,t) for n,t in module.named_parameters()]+
                   [('buffer.'+n,t) for n,t in module.named_buffers()])
    if all(t.device.type=='cpu' for _,t in tensors): return state_sha256(module)
    groups={}; locations={}
    for name,tensor in tensors:
        value=tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        groups.setdefault(tensor.device,[]).append((name,value))
    buffers={}
    for device,items in groups.items():
        # One blocking transfer per device, preserving every original tensor byte.
        buffers[device]=memoryview(torch.cat([v for _,v in items]).cpu().numpy())
        offset=0
        for name,value in items:
            locations[name]=(device,offset,offset+value.numel()); offset+=value.numel()
    digest=hashlib.sha256()
    for name,tensor in tensors:
        digest.update(str((name,str(tensor.dtype),tuple(tensor.shape))).encode())
        device,start,end=locations[name]
        digest.update(buffers[device][start:end])
    return digest.hexdigest()


class FrozenUSE(LegacyFrozenUSE):
    """Same boundary and SHA256 byte stream, with batched device transfers."""
    def verify(self,optimizer=None):
        if any(m.training for m in self.encoder.modules()) or any(p.requires_grad or p.grad is not None
                                        for p in self.encoder.parameters()):
            raise RuntimeError('Frozen USE training/gradient boundary violated')
        if optimizer is not None:
            identities={id(p) for p in self.encoder.parameters()}
            if any(id(p) in identities for group in optimizer.param_groups for p in group['params']):
                raise RuntimeError('Optimizer contains frozen USE parameters')
        if packed_state_sha256(self.encoder)!=self.reference_sha256:
            raise RuntimeError('Frozen USE parameter/buffer SHA256 changed')
        return self.reference_sha256


def pyramid(encoder,observation):
    from .use_v3 import USEv3
    return encoder(observation,return_pyramid=True) if isinstance(encoder,USEv3) else encoder(observation)


class NativeProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([nn.Conv2d(d,32,1) for d in (32,64,128,256)])

    def forward(self,features,size):
        return sum(F.interpolate(p(x),size=size,mode='bilinear',align_corners=False)
                   for p,x in zip(self.layers,features))/4


class Reconstruction(nn.Module):
    def __init__(self,encoder,task=None):
        super().__init__()
        self.use=FrozenUSE(encoder) if task else encoder
        self.task=task
        self.projection=NativeProjection() if task=='context' else None
        self.decode=nn.Sequential(nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,1))
        self.condition=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,33))

    def features(self,observation):
        if self.task:
            # FrozenUSE preserves its hash and eval boundary; v3 defaults omit pyramid.
            self.use.verify()
            with torch.no_grad(): result=pyramid(self.use.encoder,observation)
            self.use.verify()
            return result
        return pyramid(self.use,observation)

    def predict(self,features,metadata,known,summary):
        size=features['fine'].shape[-2:]
        if self.task=='fine': x=features['fine']
        elif self.task=='context': x=self.projection(features['context'],size)
        else: x=features['fine']+F.interpolate(features['main'],size=size,mode='bilinear',align_corners=False)
        x=self.decode(x)
        weights=self.condition(torch.cat((torch.log1p(metadata),known.float(),summary),-1))
        return torch.einsum('bcd,bdhw->bchw',weights[...,:32],x)+weights[...,32,None,None]

    def forward(self,observation,metadata,known,summary):
        return self.predict(self.features(observation),metadata,known,summary)


class FullDecoder(nn.Module):
    def __init__(self,encoder,kind,pretrained):
        super().__init__()
        from .use_downstream import build_visual
        self.use=FrozenUSE(encoder)
        torch.manual_seed(1235)
        self.adapter=NativeProjection()
        self.visual,_=build_visual('unet' if kind=='unet' else 'patch32',str(pretrained))

    def train(self,mode=True):
        super().train(mode)
        for m in self.visual.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm): m.eval()
        return self

    def forward(self,observation):
        self.use.verify()
        with torch.no_grad(): f=pyramid(self.use.encoder,observation)
        self.use.verify()
        return self.visual(f['fine']+self.adapter(f['context'],f['fine'].shape[-2:]))['logits']
