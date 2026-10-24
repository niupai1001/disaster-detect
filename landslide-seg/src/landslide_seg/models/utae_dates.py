"""U-TAE adapter enabling the vendored LTAE's existing positional encoder.

The vendored forward commented out positional encoding; its source remains untouched.
Below follows that spatial/temporal forward with the original PE addition enabled.
Variable lengths run in length groups, preventing padding from altering GroupNorm.

时间机制的同架构控制（与日期组共用CNN、解码器、skip连接）：

- ``positional="dates"``（默认）：按真实观测日期加位置编码，与已有日期组一致；
- ``positional="none"``：去掉位置编码模块，只保留内容注意力；
- ``aggregation="mean"``：用时间均值代替 L-TAE 注意力，并让解码器 skip 也用均值。
被关掉的模块会从模型里移除，参数统计因此只含实际参与前向的权重。
"""
import torch
from .utae_seg import UTAESeg

class UTAERealDates(UTAESeg):
    def __init__(self, in_channels: int = 11, num_classes: int = 1,
                 positional: str = "dates", aggregation: str = "attention"):
        if positional not in ("dates", "none"):
            raise ValueError(f"positional 必须是 dates/none，得到 {positional!r}")
        if aggregation not in ("attention", "mean"):
            raise ValueError(f"aggregation 必须是 attention/mean，得到 {aggregation!r}")
        super().__init__(in_channels, num_classes,
                         agg_mode="att_group" if aggregation == "attention" else "mean")
        self.positional, self.aggregation = positional, aggregation
        if positional == "none":
            self.net.temporal_encoder.positional_encoder = None
        if aggregation == "mean":
            self.net.temporal_encoder.attention_heads = None
        self.adapter_report.update(positional=positional, aggregation=aggregation)
        self._init_digest = self.digest()

    def _temporal(self,x,dates):
        m=self.net.temporal_encoder
        b,t,c,h,w=x.shape
        z=x.permute(0,3,4,1,2).contiguous().view(b*h*w,t,c)
        z=m.in_norm(z.permute(0,2,1)).permute(0,2,1)
        if m.inconv is not None:z=m.inconv(z.permute(0,2,1)).permute(0,2,1)
        if self.aggregation=="mean":
            z=m.dropout(m.mlp(z.mean(dim=1)))
            if m.out_norm is not None:z=m.out_norm(z)
            return z.view(b,h,w,-1).permute(0,3,1,2), None
        if m.positional_encoder is not None:
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
