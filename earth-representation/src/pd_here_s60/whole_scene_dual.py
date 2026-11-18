"""Variable-grid adaptation of the 120 m R3 dual-branch ViT."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from pd_here_geotoken_dualvit_r3.model import Block, grid_position, sincos_1d, to_subpatch
from .whole_scene import LayoutProjection, WholeSceneViT


class DualProjection(LayoutProjection):
    def __init__(self, layout, means):
        super().__init__(layout,means,joint_input=False)
        self.subshapes = tuple(((h+1)//2,(w+1)//2) for _,h,w in layout)
        self.subcounts = tuple(h*w for h,w in self.subshapes)
        self.spectral_values = nn.ModuleDict({str(n):nn.Linear(n,128) for n in sorted(set(self.widths))})
        self.spectral_qa = nn.ModuleDict({str(n):nn.Linear(n,128,bias=False) for n in sorted(set(self.widths))})
        self.spectral_reduce = nn.Linear(len(layout)*128,128)
        self.register_buffer('band_position',sincos_1d(torch.arange(len(layout)),128))
        self.availability = nn.Linear(len(layout),256,bias=False)
        self.factorized = sum(self.subcounts)*128>131072
        if self.factorized:
            # Very fine imagery cannot use the legacy dense spatial reducer within the GPU budget.
            self.pool = nn.ParameterList([nn.Parameter(torch.empty(n,4)) for n in self.subcounts])
            for p in self.pool:
                nn.init.normal_(p,std=p.shape[0]**-.5)
            self.spatial_reduce = nn.Linear(len(layout)*4*128,128)
        else:
            self.spatial_reduce = nn.Linear(sum(self.subcounts)*128,128)
        for j,shape in enumerate(self.subshapes):
            self.register_buffer(f'local_{j}',grid_position(shape,128),persistent=False)


class WholeSceneDualViT(WholeSceneViT):
    projection_type = DualProjection

    def __init__(self, layouts, *, global_depth=2, spectral_only=False):
        if global_depth not in (1,2):
            raise ValueError('Expected one or two post-fusion global blocks')
        super().__init__(layouts)
        self.spatial_values = nn.Linear(4,128)
        self.spatial_qa = nn.Linear(4,128,bias=False)
        self.spectral_block = Block(128)
        self.fuse = nn.Linear(256,256)
        self.global_blocks = nn.ModuleList([Block(256),Block(256)])
        self.encoder_norm = nn.LayerNorm(256,eps=1e-6)
        self.decoder_embed = nn.Linear(256,128)
        self.spectral_only = spectral_only
        if global_depth==1:
            del self.global_blocks[1]
        if spectral_only:
            spectral_projection = nn.Linear(128,256)
            with torch.no_grad():
                spectral_projection.weight.copy_(self.fuse.weight[:,128:])
                spectral_projection.bias.copy_(self.fuse.bias)
            self.fuse = spectral_projection
            del self.spatial_values,self.spatial_qa,self.spatial_block
            for projection in self.projections.values():
                del projection.spatial_reduce
                if projection.factorized:
                    del projection.pool
                for j in range(len(projection.subshapes)):
                    delattr(projection,f'local_{j}')

    def spatial_band(self, values, qa, position, weight, pooling):
        v = self.spatial_values(values)+self.spatial_qa(qa.float())+position
        v = self.spatial_block(v,qa.any(-1))
        if pooling is not None:
            v = torch.einsum('nsd,sr->nrd',v,pooling)
        return F.linear(v.flatten(1),weight)

    def branch_tokens(self, a, scene, selected, visible_qa, bands, structural):
        x = scene.values[structural][selected]
        q = visible_qa[selected]
        spec = []
        valid_spec = []
        spectral_ids = bands.nonzero().flatten().tolist()
        start = 0
        if not self.spectral_only:
            spatial = a.spatial_reduce.bias.expand(len(x),-1)
        for j,((_,h,w),value,quality) in enumerate(zip(scene.layout,x.split(a.widths,-1),
                                                        q.split(a.widths,-1),strict=True)):
            n = a.widths[j]
            width = (4 if a.factorized else a.subcounts[j])*128
            if j not in spectral_ids:
                start += width
                continue
            safe = torch.where(quality,value,0.)
            spec.append(a.spectral_values[str(n)](safe)+a.spectral_qa[str(n)](quality.float())+a.band_position[j])
            present = quality.any(-1)
            valid_spec.append(present)
            if self.spectral_only:
                continue
            if not present.any():
                start += width
                continue
            r,c = a.subshapes[j]
            pad = (0,2*c-w,0,2*r-h)
            sub = to_subpatch(F.pad(safe[present].reshape(-1,h,w),pad))
            sq = to_subpatch(F.pad(quality[present].reshape(-1,h,w),pad,value=False))
            pos = getattr(a,f'local_{j}')+a.band_position[j]
            weight = a.spatial_reduce.weight[:,start:start+width]
            pool = a.pool[j] if a.factorized else None
            stride = max(1,48000//a.subcounts[j])
            parts=[]
            for xx,qq in zip(sub.split(stride),sq.split(stride),strict=True):
                if self.training and a.subcounts[j]*len(xx)>=24000:
                    projected=checkpoint(self.spatial_band,xx,qq,pos,weight,pool,use_reentrant=False)
                else:
                    projected=self.spatial_band(xx,qq,pos,weight,pool)
                parts.append(projected)
            contribution=torch.zeros_like(spatial)
            contribution[present]=torch.cat(parts).to(contribution.dtype)
            spatial = spatial+contribution
            start += width
        spectral = self.spectral_block(torch.stack(spec,1),torch.stack(valid_spec,1))
        fixed = spectral.new_zeros(len(x),len(bands),128)
        fixed[:,spectral_ids] = spectral
        spectral = a.spectral_reduce(fixed.flatten(1))
        joint = spectral if self.spectral_only else torch.cat((spatial,spectral),-1)
        return self.fuse(joint)+a.availability(bands.float())

    def forward(self, scene, keep, bands):
        a = self.projections[self.layout_indices[scene.layout]]
        structural = scene.qa.any(-1)
        qa = scene.qa[structural]
        visible = keep[:,None] if keep.ndim==1 else keep
        evidence = qa & bands[a.band_slots] & visible[structural]
        selected = evidence.any(-1)
        position = grid_position(scene.grid,128).to(qa.device)[structural]
        if selected.any():
            x = self.branch_tokens(a,scene,selected,evidence,bands,structural)
            x = x+grid_position(scene.grid,256).to(qa.device)[structural][selected]
        else:
            x = position.new_zeros(0,256)
        if position.is_cuda:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                decoded = self.decode(x,selected,position,a.decoder_availability(bands.float()))
        else:
            decoded = self.decode(x,selected,position,a.decoder_availability(bands.float()))
        return a.output(decoded)

    def decode(self, x, selected, position, availability):
        if len(x):
            x = x[None]
            for block in self.global_blocks:
                x = block(x)
            x = self.encoder_norm(x[0])
        return self.decode_latent(x,selected,position,availability)

