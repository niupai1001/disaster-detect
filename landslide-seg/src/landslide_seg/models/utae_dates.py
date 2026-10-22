"""U-TAE adapter enabling the vendored LTAE's existing positional encoder.

The vendored forward commented out positional encoding; its source remains untouched.
Below follows that spatial/temporal forward with the original PE addition enabled.
Variable lengths run in length groups, preventing padding from altering GroupNorm.
"""
import torch
from .utae_seg import UTAESeg

class UTAERealDates(UTAESeg):
    def _temporal(self,x,dates):
        m=self.net.temporal_encoder
        b,t,c,h,w=x.shape
        z=x.permute(0,3,4,1,2).contiguous().view(b*h*w,t,c)
        z=m.in_norm(z.permute(0,2,1)).permute(0,2,1)
        if m.inconv is not None:z=m.inconv(z.permute(0,2,1)).permute(0,2,1)
        positions=dates[:,None,None,:].expand(b,h,w,t).reshape(b*h*w,t)
        z=z+m.positional_encoder(positions)
        z,att=m.attention_heads(z,pad_mask=None)
        z=z.permute(1,0,2).contiguous().view(b*h*w,-1)
        z=m.dropout(m.mlp(z))
        if m.out_norm is not None:z=m.out_norm(z)
        z=z.view(b,h,w,-1).permute(0,3,1,2)
        att=att.view(m.n_head,b,h,w,t).permute(0,1,4,2,3)
        return z,att
    def _dense(self,x,dates):
        n=self.net
        features=[n.in_conv.smart_forward(x)]
        for block in n.down_blocks:features.append(block.smart_forward(features[-1]))
        out,att=self._temporal(features[-1],dates)
        for i,block in enumerate(n.up_blocks):
            skip=n.temporal_aggregator(features[-i-2],pad_mask=None,attn_mask=att)
            out=block(out,skip)
        return n.out_conv(out)
    def forward(self,x,dates,keep=None,anchor=None):
        if keep is None or bool(keep.all()):return {'logits':self._dense(x,dates)}
        lengths=keep.sum(1)
        if bool((lengths==0).any()):raise ValueError('Empty observation sequence')
        assert torch.equal(keep,torch.arange(x.shape[1],device=x.device)[None]<lengths[:,None])
        output=[];indices=[]
        for length in lengths.unique(sorted=True).tolist():
            ix=(lengths==length).nonzero().flatten()
            output.append(self._dense(x[ix,:length],dates[ix,:length]));indices.append(ix)
        ix=torch.cat(indices);return {'logits':torch.cat(output)[ix.argsort()]}
