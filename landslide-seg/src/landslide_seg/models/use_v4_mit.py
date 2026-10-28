"""v3 spectral readout injected before the frozen MiT Stage1 patch norm."""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .use_v2 import USEv2, attend

ROOT = Path(__file__).resolve().parents[3]
V3 = ROOT/'outputs/use_v3/runs/pretrain/best_unlabeled.pt'
MIT = ROOT/'weights/segformer/segformer-b0-finetuned-ade-512-512'
FRONTEND = ('tokenizer','physical','spectral_norm','spectral','ffn','queries','readout','residual','residual_out')


def load_visual():
    from transformers import SegformerForSemanticSegmentation
    model, info = SegformerForSemanticSegmentation.from_pretrained(
        str(MIT), local_files_only=True, use_safetensors=True, output_loading_info=True)
    if any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError(f'Incomplete pretrained loading: {info}')
    return model


def propagate(support, conv):
    return F.max_pool2d(support[:,None].float(), conv.kernel_size, conv.stride,
                        conv.padding, conv.dilation)[:,0].bool()


def masked_block(layer, tokens, support, height, width):
    """Original Q/K/V, SR, output and FFN; only unsupported attention keys are masked."""
    valid = support.flatten(1)
    z = layer.layer_norm_1(tokens).masked_fill(~valid[...,None], 0)
    attention = layer.attention.self
    b, _, d = z.shape
    def heads(x):
        return x.reshape(b,-1,attention.num_attention_heads,attention.attention_head_size).transpose(1,2)
    q = heads(attention.query(z))
    if attention.sr_ratio > 1:
        reduced = attention.sr(z.transpose(1,2).reshape(b,d,height,width))
        keys = propagate(support, attention.sr).flatten(1)
        z = attention.layer_norm(reduced.flatten(2).transpose(1,2))
    else:
        keys = valid
    safe = keys.clone(); empty = ~keys.any(1); safe[empty,0] = True
    k, v = heads(attention.key(z)), heads(attention.value(z))
    scores = (q @ k.transpose(-1,-2)) / math.sqrt(attention.attention_head_size)
    probs = attention.dropout(scores.masked_fill(~safe[:,None,None,:],float('-inf')).softmax(-1))
    context = (probs @ v).transpose(1,2).reshape(b,-1,d).masked_fill(empty[:,None,None],0)
    out = layer.attention.output(context, z)
    out = out.masked_fill(empty[:,None,None],0)
    tokens = (tokens + layer.drop_path(out)).masked_fill(~valid[...,None],0)
    z = layer.layer_norm_2(tokens).masked_fill(~valid[...,None],0)
    return (tokens + layer.drop_path(layer.mlp(z,height,width))).masked_fill(~valid[...,None],0)


def staged(encoder, x, support=None, injected=False):
    features, supports = [], []
    if support is None:
        support = torch.ones(x.shape[0],*x.shape[-2:],dtype=torch.bool,device=x.device)
    for index, (patch, blocks, norm) in enumerate(zip(encoder.patch_embeddings,encoder.block,encoder.layer_norm)):
        if index == 0 and injected:
            h,w = x.shape[-2:]
            tokens = patch.layer_norm(x.flatten(2).transpose(1,2))
        else:
            support = propagate(support,patch.proj)
            tokens,h,w = patch(x)
        tokens = tokens.masked_fill(~support.flatten(1)[...,None],0)
        for block in blocks:
            tokens = masked_block(block,tokens,support,h,w)
        tokens = norm(tokens).masked_fill(~support.flatten(1)[...,None],0)
        x = tokens.transpose(1,2).reshape(x.shape[0],-1,h,w).contiguous()
        features.append(x); supports.append(support)
    return features,supports


class USEv4(nn.Module):
    def __init__(self, source=V3):
        super().__init__()
        front = USEv2(attention=True)
        state = torch.load(source,map_location='cpu',weights_only=False)['use']
        for name in FRONTEND:
            value = getattr(front,name)
            if isinstance(value,nn.Parameter):
                value = nn.Parameter(state[name].clone())
            else:
                prefix = name+'.'
                value.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
            setattr(self,name,value)
        self.adapter = nn.Conv2d(32,32,3,stride=2,padding=1)
        self.mit = load_visual().segformer.encoder.requires_grad_(False).eval()

    def train(self,mode=True):
        super().train(mode)
        self.mit.eval()
        return self

    def forward(self,observation,return_pyramid=True,diagnostics=False):
        fine, query, support, fine_valid, known_used, queries = self.frontend(observation)
        support = propagate(support,self.adapter)
        injected = self.adapter(query).masked_fill(~support[:,None],0)
        features,supports = staged(self.mit,injected,support,injected=True)
        result = dict(fine=fine,context=features,valid=dict(fine=fine_valid,context=supports),
                      stride=dict(fine=1,context=[4,8,16,32]),
                      spatial_metadata=observation.get('spatial_metadata'),metadata_used=known_used)
        if diagnostics: result.update(queries=queries,mit_input=injected)
        return result

    def frontend(self, observation):
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
        return fine, context, cv, fine_valid, known & valid.flatten(2).any(-1)[...,None], output

