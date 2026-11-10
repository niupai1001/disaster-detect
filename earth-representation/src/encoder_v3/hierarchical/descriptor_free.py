"""Shared ordinary convolution and explicit learned identities; no physical descriptors."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import Encoder


class LearnedBands(nn.Module):
    def __init__(self,keys):
        super().__init__()
        if not keys or any(len(k)!=2 or any(not isinstance(v,str) or not v for v in k) for k in keys):
            raise ValueError('Explicit instrument/band keys required')
        self.keys=tuple(tuple(k) for k in keys)
        if len(set(self.keys))!=len(self.keys):raise ValueError('Duplicate band identity')
        self.lookup={key:i for i,key in enumerate(self.keys)}
        self.fusion=nn.Embedding(len(keys),32)
        self.output=nn.Embedding(len(keys),32)

    def forward(self,sample):
        try:ids=[self.lookup[(sample.instrument,c.name)] for c in sample.channels]
        except (KeyError,AttributeError) as exc:raise ValueError('Untrained instrument/band identity') from exc
        ids=torch.tensor(ids,device=self.output.weight.device)
        fusion=self.fusion(ids)
        return {'conv':fusion,'fusion':fusion,'output':self.output(ids)}


class SharedConvolution(nn.Conv2d):
    def __init__(self):super().__init__(1,32,3,padding=1)

    def forward(self,condition):
        return self.weight[None].expand(len(condition),-1,-1,-1,-1),self.bias[None].expand(len(condition),-1)


class DescriptorFreeEncoder(Encoder):
    def __init__(self,depth,keys):
        super().__init__(depth)
        self.memory=LearnedBands(keys)
        self.convolution=SharedConvolution()

    def conditions(self,sample):return self.memory(sample)

    def convolve(self,clean,kernel):
        # Treat bands on one native grid as a batch for the same ordinary kernel.
        return F.conv2d(clean[:,None],self.convolution.weight,padding=1)
