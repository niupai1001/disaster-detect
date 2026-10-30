import torch
from torch import nn

from landslide_seg.models.use_next_v2 import EOAsset
from landslide_seg.models.use_next_v2.anchors import S2_BANDS
from landslide_seg.models.use_next_v2.geometry import to_label_grid
from .encoder import ScaleUSE,grid_for


class ScaleReadout(nn.Module):
    def __init__(self,root,mode,stats,local_state):
        super().__init__();self.use=ScaleUSE(root,mode);self.stats=stats
        self.use.local.load_state_dict(local_state,strict=True)
        self.use.requires_grad_(False).eval()
        torch.manual_seed(1235)
        self.projections=nn.ModuleList([nn.Conv2d(c,32,1) for c in (192,192,192,768,1536)])
        self.head=nn.Sequential(nn.GroupNorm(4,32),nn.GELU(),nn.Conv2d(32,1,1))

    def train(self,mode=True):
        super().train(mode);self.use.eval();return self

    def forward(self,batch):
        outputs=[]
        for i,x in enumerate(batch['x']):
            meta=batch['spatial_metadata'][i];valid=batch['pixel_valid'][i]
            physical=(x*x.new_tensor(self.stats['std'])[:,None,None]+x.new_tensor(self.stats['mean'])[:,None,None])*.0001
            asset=EOAsset('s2',S2_BANDS,torch.where(valid,physical,0),batch['band_metadata'][i],batch['known_mask'][i],
                {k:meta[k] for k in ('crs','transform')},None,None,valid,meta['date'],'surface_reflectance',meta)
            with torch.no_grad():result=self.use(asset,grid_for(asset))
            levels=[result.native_detail,result.metric_local,result.metric_patch,result.semantic_levels.get('dofa'),result.semantic_levels.get('anysat')]
            maps=[];supports=[]
            for proj,level in zip(self.projections,levels):
                if level is None:continue
                feature,mask=to_label_grid(level,asset.sample_grid['transform'],asset.sample_grid['crs'],asset.image.shape[-2:])
                maps.append(proj(feature[None])*mask[None,None]);supports.append(mask.float())
            fused=torch.stack(maps).sum(0)/torch.stack(supports).sum(0).clamp_min(1)[None,None]
            outputs.append(self.head(fused))
        return torch.cat(outputs)
