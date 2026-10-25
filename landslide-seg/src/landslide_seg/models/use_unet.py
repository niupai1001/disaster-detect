"""A trainable U-Net consuming frozen USE's 32-channel spatial features."""
import torch
from torch import nn
from torch.nn import functional as F


def _block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(8, outputs), nn.ReLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(8, outputs), nn.ReLU(inplace=True))


class UNetFeatureDecoder(nn.Module):
    """Four down/up levels, skip concatenation; all task-side weights start random."""

    def __init__(self):
        super().__init__()
        widths=(32,64,128,256,512)
        self.encoder=nn.ModuleList(_block(32 if i==0 else widths[i-1], width) for i,width in enumerate(widths))
        self.pool=nn.MaxPool2d(2)
        self.upsample=nn.ModuleList(nn.ConvTranspose2d(widths[i],widths[i-1],2,stride=2) for i in range(4,0,-1))
        self.decoder=nn.ModuleList(_block(2*widths[i-1],widths[i-1]) for i in range(4,0,-1))
        self.head=nn.Conv2d(32,1,1)
        self.load_info={'initialization':'random; no pretrained U-Net weights'}
        self.adapter_report={'interface':'frozen USE32 directly to full trainable U-Net',
            'widths':list(widths),'skip':'concatenation','upsampling':'transposed convolution',
            'normalization':'GroupNorm(8); no BatchNorm running buffers','use_multiscale':False}

    def forward(self,features,**kwargs):
        if features.ndim!=4 or features.shape[1]!=32 or min(features.shape[-2:])<16:
            raise ValueError('USE U-Net requires [B,32,H,W] with H,W >= 16')
        skips=[];x=features
        for index,stage in enumerate(self.encoder):
            x=stage(x if index==0 else self.pool(x));skips.append(x)
        for up,stage,skip in zip(self.upsample,self.decoder,reversed(skips[:-1])):
            x=up(x)
            if x.shape[-2:]!=skip.shape[-2:]:
                x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=stage(torch.cat((skip,x),dim=1))
        return {'logits':self.head(x)}

    def param_groups(self):
        # All U-Net layers are new task-side parameters, matching the new FPN learning rate.
        return {'encoder':[],'decoder':list(self.parameters()),'new':[]}
