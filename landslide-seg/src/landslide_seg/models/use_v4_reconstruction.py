"""Task modules, kept outside the transferable USE encoder."""
import torch
from torch import nn
from torch.nn import functional as F
from .use_v3_reconstruction import Reconstruction as V3Reconstruction, FrozenUSE
from .use_v4_mit import load_visual
from .spectral_stem import ResidualFusionHead


class NativeProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(d,32,1) for d in (32,64,160,256))

    def forward(self,features,size):
        return sum(F.interpolate(p(x),size=size,mode='bilinear',align_corners=False)
                   for p,x in zip(self.layers,features))/4


class Reconstruction(V3Reconstruction):
    def __init__(self,encoder,task=None):
        super().__init__(encoder,task)
        if task=='context': self.projection=NativeProjection()
        self.fusion=nn.ModuleList(nn.Conv2d(d,32,1) for d in (32,64,160,256)) if task is None else None

    def features(self,observation):
        if self.task:
            with torch.no_grad(): result=self.use.encoder(observation)
        else: result=self.use(observation)
        if self.fusion is not None:
            main=self.fusion[-1](result['context'][-1])
            for i in (2,1,0):
                main=self.fusion[i](result['context'][i])+F.interpolate(main,size=result['context'][i].shape[-2:],mode='bilinear',align_corners=False)
                main=main.masked_fill(~result['valid']['context'][i][:,None],0)
            result['main']=main
            result['valid']['main']=result['valid']['context'][0]
        return result


class Segmenter(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.use=FrozenUSE(encoder)
        self.decoder=load_visual().decode_head
        self.decoder.classifier=nn.Conv2d(256,1,1)
        self.residual=ResidualFusionHead()
        self.train()

    def train(self,mode=True):
        super().train(mode)
        self.use.eval()
        for m in self.decoder.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm): m.eval()
        return self

    def forward(self,observation):
        with torch.no_grad(): f=self.use.encoder(observation)
        features=[]
        for value,proj in zip(f['context'],self.decoder.linear_c):
            b,_,h,w=value.shape
            features.append(F.interpolate(proj(value).transpose(1,2).reshape(b,256,h,w),size=f['context'][0].shape[-2:],mode='bilinear',align_corners=False))
        d=self.decoder
        decoded=d.dropout(d.activation(d.batch_norm(d.linear_fuse(torch.cat(features[::-1],1)))))
        size=f['fine'].shape[-2:]
        return F.interpolate(d.classifier(decoded),size=size,mode='bilinear',align_corners=False)+self.residual(decoded,f['fine'],size)
