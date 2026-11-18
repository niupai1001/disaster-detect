"""60 m Dual-Axis encoder for variable native layouts and scene extents."""
from collections import defaultdict

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from pd_here_geotoken_dualvit_r3.model import Block, grid_position, sincos_1d
from .whole_scene import WholeSceneViT


MODEL_ID = 'S60-WholeScene-DualAxisViT'


class SpectralProjection(nn.Module):
    def __init__(self, layout):
        super().__init__()
        shapes = dict.fromkeys((h,w) for _,h,w in layout)
        self.values = nn.ModuleDict({f'{h}x{w}':nn.Linear(h*w,128) for h,w in shapes})
        self.qa = nn.ModuleDict({f'{h}x{w}':nn.Linear(h*w,128,bias=False) for h,w in shapes})
        self.register_buffer('band_position',sincos_1d(torch.arange(len(layout)),128))

    def forward(self, values, qa, layout, visible_ids):
        widths = [h*w for _,h,w in layout]
        values = values.split(widths,-1)
        qa = qa.split(widths,-1)
        groups = defaultdict(list)
        for column,band in enumerate(visible_ids):
            _,h,w = layout[band]
            groups[f'{h}x{w}'].append((column,band))
        tokens = values[0].new_zeros(len(values[0]),len(visible_ids),128)
        valid = torch.zeros(tokens.shape[:2],dtype=torch.bool,device=tokens.device)
        for shape,items in groups.items():
            columns,bands = zip(*items)
            pixels = torch.stack([values[j] for j in bands],1)
            quality = torch.stack([qa[j] for j in bands],1)
            tokens[:,columns] = self.values[shape](pixels)+self.qa[shape](quality.float())+self.band_position[list(bands)]
            valid[:,columns] = quality.any(-1)
        return tokens,valid


class WholeSceneDualAxisViT(WholeSceneViT):
    def __init__(self, layouts):
        # This initialization order preserves every shared spatial-only initial tensor.
        super().__init__(layouts)
        self.spectral_projections = nn.ModuleDict({
            index:SpectralProjection(layout) for layout,index in self.layout_indices.items()
        })
        self.spectral_block = Block(128)
        self.fusion_block = Block(128)
        self.types = nn.Parameter(torch.empty(2,128))
        nn.init.normal_(self.types,std=.02)

    def forward(self, scene, keep, bands):
        index = self.layout_indices[scene.layout]
        projection = self.projections[index]
        structural = scene.qa.any(-1)
        position = grid_position(scene.grid,128).to(scene.values.device)[structural]
        evidence = scene.qa[structural] & bands[projection.band_slots]
        selected = keep[structural] & evidence.any(-1)
        quality = evidence[selected]
        safe = torch.where(quality,scene.values[structural][selected],0.)
        spatial = (projection.input_values(safe)+projection.input_qa(quality.float())
                   +projection.availability(bands.float())+position[selected])
        available = projection.decoder_availability(bands.float())
        if len(spatial):
            if spatial.is_cuda:
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    spatial = self.spatial_block(spatial[None])[0]
            else:
                spatial = self.spatial_block(spatial[None])[0]
            visible_ids = bands.nonzero().flatten().tolist()
            spectral,valid = self.spectral_projections[index](safe,quality,scene.layout,visible_ids)
            spectral = self.spectral_block(spectral,valid)
            fused = torch.cat((spatial[:,None]+self.types[0],spectral+self.types[1]),1)
            valid = torch.cat((torch.ones_like(valid[:,:1]),valid),1)
            spatial = self.encoder_norm(self.fusion_block(fused,valid)[:,0])
        if spatial.is_cuda:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                decoded = self.decode_latent(spatial,selected,position,available)
        else:
            decoded = self.decode_latent(spatial,selected,position,available)
        return projection.output(decoded)
