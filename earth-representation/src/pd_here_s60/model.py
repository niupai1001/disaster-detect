"""60 m Dual-Axis ViT on the fixed native Sentinel-2 layout."""
from dataclasses import asdict, dataclass
import hashlib
import json

import torch
from torch import nn

from pd_here_geotoken_dualvit_r3.model import Block, grid_position, sincos_1d


@dataclass(frozen=True)
class Layout:
    band_keys: tuple = ('B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12')
    band_gsd: tuple = (10,10,10,20,20,20,10,20,20,20)
    native_shapes: tuple = ((6,6),(6,6),(6,6),(3,3),(3,3),(3,3),(6,6),(3,3),(3,3),(3,3))
    offsets: tuple = (0,36,72,108,117,126,135,171,180,189,198)
    patch_size_m: int = 60
    window_shape_m: tuple = (480,480)
    grid_shape: tuple = (8,8)
    qa_rule: str = 'audited QA and finite value; padding invalid'
    patch_order: str = 'row-major; native row-major inside each band'

    @property
    def layout_hash(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


def retile(x, old_grid, new_grid):
    """Reindex an existing native raster; preserve the leading batch axis."""
    b,_,h,w = x.shape
    raster = x.reshape(b,old_grid,old_grid,h,w).transpose(2,3).reshape(b,old_grid*h,old_grid*w)
    nh,nw = raster.shape[-2]//new_grid,raster.shape[-1]//new_grid
    return raster.reshape(b,new_grid,nh,new_grid,nw).transpose(2,3).reshape(b,new_grid**2,nh,nw)


def inputs60(values, qa, keep):
    return (tuple(retile(x,4,8) for x in values),
            tuple(retile(q,4,8) for q in qa),
            retile(keep[...,None,None].expand(-1,-1,2,2),4,8).flatten(1))


class DualAxisViT(nn.Module):
    def __init__(self, means=None, *, spatial_only=False):
        super().__init__()
        self.layout = Layout()
        self.spatial_only = spatial_only
        self.spatial_values = nn.Linear(198,128)
        self.spatial_qa = nn.Linear(198,128,bias=False)
        self.availability = nn.Linear(10,128,bias=False)
        if not spatial_only:
            self.spectral_values = nn.ModuleDict({str(k):nn.Linear(k,128) for k in (9,36)})
            self.spectral_qa = nn.ModuleDict({str(k):nn.Linear(k,128,bias=False) for k in (9,36)})
        self.spatial_block = Block(128)
        if not spatial_only:
            self.spectral_block = Block(128)
            self.fusion_block = Block(128)
            self.types = nn.Parameter(torch.empty(2,128))
            nn.init.normal_(self.types,std=.02)
        self.encoder_norm = nn.LayerNorm(128,eps=1e-6)
        self.decoder_embed = nn.Linear(128,128)
        self.mask_token = nn.Parameter(torch.empty(1,1,128))
        nn.init.normal_(self.mask_token,std=.02)
        self.decoder_availability = nn.Linear(10,128,bias=False)
        self.decoder_blocks = nn.ModuleList([Block(128),Block(128)])
        self.decoder_norm = nn.LayerNorm(128,eps=1e-6)
        self.prediction = nn.Linear(128,198)
        self.register_buffer('position',grid_position((8,8),128))
        if not spatial_only:
            self.register_buffer('band_position',sincos_1d(torch.arange(10),128))
        if means is not None:
            with torch.no_grad():
                self.prediction.weight.zero_()
                self.prediction.bias.copy_(torch.repeat_interleave(
                    torch.as_tensor(means,dtype=torch.float32),torch.tensor([h*w for h,w in self.layout.native_shapes])))

    def encode(self, values, qa, keep, bands):
        support = torch.stack([q.flatten(2).any(-1) for q in qa],-1)
        evidence = keep & (support & bands[:,None]).any(-1)
        counts = evidence.sum(-1)
        band_counts = bands.sum(-1)
        if (counts==0).any() or not (band_counts==band_counts[0]).all():
            raise ValueError('The view needs visible evidence and a common visible band count')
        b,p = keep.shape
        ids = torch.arange(p,device=keep.device).expand(b,p).masked_fill(~evidence,p).sort(-1).values[:,:int(counts.max())]
        valid = ids<p
        ids = ids.clamp_max(p-1)
        parent,slot = valid.nonzero(as_tuple=True)
        geo = ids[parent,slot]
        q = [mask[parent,geo].flatten(1) & bands[parent,j,None] for j,mask in enumerate(qa)]
        x = [torch.where(mask,v[parent,geo].flatten(1),0.) for v,mask in zip(values,q,strict=True)]
        spatial = self.spatial_values(torch.cat(x,-1))+self.spatial_qa(torch.cat(q,-1).float())
        spatial = spatial+self.availability(bands[parent].float())+self.position[geo]
        packed = spatial.new_zeros(*ids.shape,128)
        packed[parent,slot] = spatial
        spatial = self.spatial_block(packed,valid)[parent,slot]
        if not self.spatial_only:
            n = int(band_counts[0])
            visible_ids = bands.nonzero()[:,1].reshape(b,n)[parent]
            spatial = self.fuse_spectral(spatial,visible_ids,x,q)
        packed = torch.zeros_like(packed)
        packed[parent,slot] = self.encoder_norm(spatial)
        return dict(values=packed,ids_keep=ids,valid=valid,grid_shape=self.layout.grid_shape,
                    layout_hash=self.layout.layout_hash,band_visible=bands)

    def fuse_spectral(self, spatial, visible_ids, x, q):
        groups,n = visible_ids.shape
        spectral = spatial.new_zeros(groups,n,128)
        spectral_valid = torch.zeros_like(visible_ids,dtype=torch.bool)
        for pixels in (9,36):
            canonical = [j for j,shape in enumerate(self.layout.native_shapes) if shape[0]*shape[1]==pixels]
            source = torch.tensor(canonical,device=spatial.device)
            matches = visible_ids[...,None]==source
            group,col,index = matches.nonzero(as_tuple=True)
            if not len(group):
                continue
            q_group = torch.stack([q[j] for j in canonical],1)[group,index]
            v_group = torch.stack([x[j] for j in canonical],1)[group,index]
            present = q_group.any(-1)
            group,col,index = group[present],col[present],index[present]
            v_group,q_group = v_group[present],q_group[present]
            spectral[group,col] = (self.spectral_values[str(pixels)](v_group)
                                   +self.spectral_qa[str(pixels)](q_group.float())+self.band_position[source[index]])
            spectral_valid[group,col] = True
        spectral = self.spectral_block(spectral,spectral_valid)
        joint = torch.cat((spatial[:,None]+self.types[0],spectral+self.types[1]),1)
        fusion_valid = torch.cat((torch.ones_like(spectral_valid[:,:1]),spectral_valid),1)
        return self.fusion_block(joint,fusion_valid)[:,0]

    def decode(self, encoding, structural):
        valid = encoding['valid']
        parent,slot = valid.nonzero(as_tuple=True)
        full = self.mask_token.expand(len(structural),64,-1).clone()
        projected = self.decoder_embed(encoding['values'][parent,slot])
        full[parent,encoding['ids_keep'][parent,slot]] = projected.to(full.dtype)
        full = torch.where(structural[...,None],full,0.)
        x = full+self.position+self.decoder_availability(encoding['band_visible'].float())[:,None]
        for block in self.decoder_blocks:
            x = block(x,structural)
        return torch.where(structural[...,None],self.prediction(self.decoder_norm(x)),0.),full

    def forward(self, values, qa, keep, bands):
        if len(values)!=10 or len(qa)!=10 or bands.shape!=(len(keep),10) or keep.shape!=(len(keep),64):
            raise ValueError('Expected fixed S60 layout and view')
        if bands.dtype!=torch.bool or keep.dtype!=torch.bool or any(
            x.shape!=(len(keep),64,*s) or q.shape!=x.shape or q.dtype!=torch.bool
            for x,q,s in zip(values,qa,self.layout.native_shapes,strict=True)):
            raise ValueError('Native shapes or boolean masks differ from S60 layout')
        structural = torch.stack([q.flatten(2).any(-1) for q in qa],-1).any(-1)
        encoded = self.encode(values,qa,keep,bands)
        prediction,full = self.decode(encoded,structural)
        return dict(**encoded,prediction=prediction,pre_position_decoder=full,encoded=encoded['values'])
