"""Band-aligned dual branches with one global token per visible (geo patch, band)."""
import torch
from torch import nn
from .model import Block, Layout, grid_position, sincos_1d


class BandTokenDualViT(nn.Module):
    def __init__(self, layout: Layout, width=128, decoder_width=128, heads=4, global_depth=2):
        super().__init__()
        self.layout = layout
        self.width = width
        self.geographic_width = 2*width
        self.decoder_width = decoder_width
        n = len(layout.band_keys)
        self.spectral_values = nn.ModuleDict()
        self.spectral_qa = nn.ModuleDict()
        self.spatial_values = nn.ModuleDict()
        self.spatial_qa = nn.ModuleDict()
        self.prediction = nn.ModuleDict()
        for h,w in sorted(set(layout.native_shapes)):
            key = f'{h}x{w}'
            self.spectral_values[key] = nn.Linear(h*w,width)
            self.spectral_qa[key] = nn.Linear(h*w,width,bias=False)
            self.spatial_values[key] = nn.Linear(h*w,width)
            self.spatial_qa[key] = nn.Linear(h*w,width,bias=False)
            self.prediction[key] = nn.Linear(decoder_width,h*w)
        self.spectral_block = Block(width,heads)
        self.fuse = nn.Linear(2*width,2*width)
        self.availability = nn.Linear(n,2*width,bias=False)
        self.global_block = Block(2*width,heads)
        self.extra_global_blocks = nn.ModuleList([Block(2*width,heads) for _ in range(global_depth-1)])
        self.encoder_norm = nn.LayerNorm(2*width,eps=1e-6)
        self.decoder_embed = nn.Linear(2*width,decoder_width)
        self.mask_token = nn.Parameter(torch.zeros(1,1,decoder_width))
        nn.init.normal_(self.mask_token,std=.02)
        self.decoder_availability = nn.Linear(n,decoder_width,bias=False)
        self.decoder_blocks = nn.ModuleList([Block(decoder_width,heads) for _ in range(2)])
        self.decoder_norm = nn.LayerNorm(decoder_width,eps=1e-6)
        self.band_offset = nn.Parameter(torch.zeros(n))
        self.register_buffer('band_position',sincos_1d(torch.arange(n),width))
        global_position = grid_position(layout.patch_grid,2*width)[:,None]+sincos_1d(torch.arange(n),2*width)[None]
        decoder_position = grid_position(layout.patch_grid,decoder_width)[:,None]+sincos_1d(torch.arange(n),decoder_width)[None]
        self.register_buffer('global_position',global_position.flatten(0,1))
        self.register_buffer('decoder_position',decoder_position.flatten(0,1))

    def forward(self, values, qa, keep_geo, band_visible):
        n,p = len(self.layout.band_keys),self.layout.patches
        if len(values)!=n or len(qa)!=n:
            raise ValueError('Expected exact frozen canonical band order')
        batch=values[0].shape[0]
        if keep_geo.shape!=(batch,p) or band_visible.shape!=(batch,n):
            raise ValueError('Mask dimensions differ from Layout')
        if keep_geo.dtype!=torch.bool or band_visible.dtype!=torch.bool:
            raise TypeError('Visibility masks must be boolean')
        counts=keep_geo.sum(-1)
        band_counts=band_visible.sum(-1)
        if (counts==0).any() or (band_counts==0).any() or not (band_counts==band_counts[0]).all():
            raise ValueError('Each parent needs visible context and a common nonzero visible band count')
        spectral,spatial,supports=[],[],[]
        computed=0
        for j,(x,q,shape) in enumerate(zip(values,qa,self.layout.native_shapes,strict=True)):
            if x.shape!=(batch,p,*shape) or q.shape!=x.shape or q.dtype!=torch.bool:
                raise ValueError('Native band values and QA differ from Layout')
            selected=(keep_geo & band_visible[:,j,None]).nonzero()
            spec=x.new_zeros(batch,p,self.width)
            space=torch.zeros_like(spec)
            support=q.flatten(2).any(-1)
            if len(selected):
                bi,pi=selected.unbind(-1)
                valid=q[bi,pi]
                safe=torch.where(valid,x[bi,pi],0.).flatten(1)
                if not torch.isfinite(safe).all():
                    raise ValueError('Nonfinite visible native values')
                valid_vector=valid.flatten(1).to(safe.dtype)
                key=f'{shape[0]}x{shape[1]}'
                s=self.spectral_values[key](safe)+self.spectral_qa[key](valid_vector)
                v=self.spatial_values[key](safe)+self.spatial_qa[key](valid_vector)
                present=valid_vector.any(-1)
                spec[bi,pi]=torch.where(present[:,None],s,0.)
                space[bi,pi]=torch.where(present[:,None],v+self.band_position[j],0.)
                computed+=len(selected)
            spectral.append(spec)
            spatial.append(space)
            supports.append(support)
        support=torch.stack(supports,2)
        legal=support.any(-1)
        if (keep_geo & ~legal).any():
            raise ValueError('Kept geographic patch has no legal observations')
        pv=int(counts.max())
        nv=int(band_counts[0])
        geo_ids=torch.arange(p,device=keep_geo.device).expand(batch,p).masked_fill(~keep_geo,p).sort(-1).values[:,:pv]
        geo_kept=geo_ids<p
        geo_ids=geo_ids.clamp_max(p-1)
        band_ids=band_visible.nonzero()[:,1].reshape(batch,nv)
        bi=torch.arange(batch,device=keep_geo.device)[:,None,None]
        gi=geo_ids[:,:,None]
        bj=band_ids[:,None,:]
        spectral=torch.stack(spectral,2)[bi,gi,bj]
        spatial=torch.stack(spatial,2)[bi,gi,bj]
        pair_valid=support[bi,gi,bj] & geo_kept[:,:,None]
        if not (pair_valid.any(-1)|~geo_kept).all():
            raise ValueError('A visible geographic patch has no visible band evidence')
        spectral=spectral+self.band_position[band_ids][:,None]
        spectral=self.spectral_block(spectral.reshape(batch*pv,nv,self.width),
                                     pair_valid.reshape(batch*pv,nv)).reshape(batch,pv,nv,self.width)
        joint=self.fuse(torch.cat((spatial,spectral),-1)).reshape(batch,pv*nv,self.geographic_width)
        pair_ids=(gi*n+bj).reshape(batch,pv*nv)
        pair_valid=pair_valid.reshape(batch,pv*nv)
        encoded=joint+self.global_position[pair_ids]+self.availability(band_visible.float())[:,None]
        encoded=self.global_block(encoded,pair_valid)
        for block in self.extra_global_blocks:
            encoded=block(encoded,pair_valid)
        encoded=torch.where(pair_valid[...,None],self.encoder_norm(encoded),0.)
        full=self.mask_token.expand(batch,p*n,-1).clone()
        projected=self.decoder_embed(encoded)
        parent_ids=torch.arange(batch,device=keep_geo.device)[:,None].expand_as(pair_ids)
        full[parent_ids[pair_valid],pair_ids[pair_valid]]=projected[pair_valid]
        decoder_valid=support.flatten(1)
        full=torch.where(decoder_valid[...,None],full,0.)
        decoded=full+self.decoder_position[None]+self.decoder_availability(band_visible.float())[:,None]
        for block in self.decoder_blocks:
            decoded=block(decoded,decoder_valid)
        decoded=self.decoder_norm(decoded).reshape(batch,p,n,self.decoder_width)
        predictions=[]
        for j,(h,w) in enumerate(self.layout.native_shapes):
            band=self.prediction[f'{h}x{w}'](decoded[:,:,j])+self.band_offset[j]
            predictions.append(torch.where(support[:,:,j,None],band,0.))
        return dict(prediction=torch.cat(predictions,-1),encoded=encoded,values=encoded,
                    ids_keep=pair_ids,valid=pair_valid,patch_grid=self.layout.patch_grid,
                    layout_hash=self.layout.layout_hash,band_visible=band_visible,
                    pre_position_decoder=full,
                    trace=dict(geographic_patches_kept=pv,visible_bands=nv,
                               encoder_length=pv*nv,decoder_length=p*n,computed_band_patches=computed,
                               canonical_pair_index='geo_id * band_count + band_id'))
