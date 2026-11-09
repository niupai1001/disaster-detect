"""Candidate execution-only optimization; not enabled in the pending G5 handover."""
from types import MethodType
import torch
from .encoder_v2_fast import FastObservationEncoderV2
from .model import AttentionBlock,CrossBlock


def safe_mask(valid):
    mask=~valid
    return mask.index_copy(1,mask.new_tensor([0],dtype=torch.long),mask[:,:1]&~mask.all(1,keepdim=True))


def cross_forward(self,q,k,valid):
    mask=safe_mask(valid);normalized=self.knorm(k)
    q=q+self.attn(self.qnorm(q),normalized,normalized,key_padding_mask=mask,need_weights=False)[0]
    return q+self.ffn(self.norm(q))


def attention_forward(self,x,valid):
    mask=safe_mask(valid);q=self.norm(x)
    x=x+self.attn(q,q,q,key_padding_mask=mask,need_weights=False)[0]
    return (x+self.ffn(self.norm2(x)))*valid[...,None]


class AttentionFastObservationEncoderV2(FastObservationEncoderV2):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for block in self.modules():
            if isinstance(block,CrossBlock):block.forward=MethodType(cross_forward,block)
            elif isinstance(block,AttentionBlock):block.forward=MethodType(attention_forward,block)
