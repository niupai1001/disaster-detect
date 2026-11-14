"""V3-FR with the frozen backbone weights and frozen native supervision."""
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from encoder_v3.hierarchical_revised.tasks import task_masks
from encoder_v3.patch_latent.model import Decoder, Encoded, Encoder, Result, native_gradient_loss
from encoder_v3.shared_mae.geometry import metric_geometry

from .frontend import VisibleBand
from .fusion import PhaseAwareFusion
from .position import PhysicalPosition, positioned_block


@dataclass(frozen=True)
class Options:
    frontend: str
    blocks: int
    film: bool
    position_mode: str
    terminal_expander: str
    legacy_b1_detail: bool

    def __post_init__(self):
        if (self.frontend not in ("historical","ordered") or type(self.blocks) is not int
            or self.blocks not in (0,2) or type(self.film) is not bool
            or self.position_mode not in ("none","qk_physical")
            or self.terminal_expander not in ("legacy_continuous","learned_subpixel_residual")
            or type(self.legacy_b1_detail) is not bool or (self.film and self.blocks!=2)
            or (self.frontend=="historical" and self.blocks!=0)):
            raise ValueError("Invalid registered V3-FR option combination")


@dataclass(frozen=True)
class TargetSpec:
    """Output grid, boolean native selection and 32-dimensional descriptor, no values."""
    grid: object
    selection: Tensor
    descriptor: Tensor


def split_field(fine, support, r):
    _,_,h,w = support.shape
    slots = fine.reshape(1,-1,h,r,w,r).transpose(3,4)
    return torch.where(support[...,None,None],slots,0.).flatten(2)


class TerminalDecoder(Decoder):
    def __init__(self, old, terminal):
        nn.Module.__init__(self)
        self.resolution = old.resolution
        self.continuous = old.continuous
        self.project,self.upsample,self.fuse = old.project,old.upsample,old.fuse
        self.subpixel,self.refine = old.subpixel,old.refine
        self.readout,self.area = old.readout,old.area
        self.terminal = terminal
        if terminal=="learned_subpixel_residual":
            self.expander = nn.Conv2d(32,8*self.resolution**2,1)
            self.fine_refine = nn.Sequential(nn.Conv2d(8,16,3,padding=1),nn.GELU(),
                                             nn.Conv2d(16,8,3,padding=1))
            del self.subpixel,self.refine

    def parent_field(self, encoded):
        pyramid = [torch.where(valid,project(feature),0.)
                   for project,feature,valid in zip(self.project,encoded.features,encoded.support,strict=True)]
        feature = pyramid[-1]
        for level in reversed(range(len(pyramid)-1)):
            skip = pyramid[level]
            h,w = skip.shape[-2:]
            resized = self.resize_supported(feature,encoded.support[level+1],2)[...,:h,:w]
            feature = torch.where(encoded.support[level],skip+F.gelu(self.upsample[level](resized)),0.)
        return feature+F.gelu(self.fuse(feature))

    def subcells(self, encoded):
        feature,support = self.parent_field(encoded),encoded.support[0]
        if self.terminal=="legacy_continuous":
            packed = torch.where(support,self.subpixel(feature),0.)
            fine = self.resize_supported(packed,support,self.resolution)
            fine = fine+self.refine(fine)
        else:
            fine = F.pixel_shuffle(torch.where(support,self.expander(feature),0.),self.resolution)
            fine = fine+self.fine_refine(fine)
        return split_field(fine,support,self.resolution)

    def predict(self, encoded, targets):
        if len(targets)!=len(encoded.descriptor):
            raise ValueError("Target list must preserve the frozen band ordinals")
        geometry = [s.grid for s in targets]
        if any(grid not in encoded.geometry.grids for grid in geometry):
            raise ValueError("Target geometry outside immutable scene")
        requested = Encoded(encoded.features,encoded.support,encoded.geometry,
                            torch.stack([spec.descriptor for spec in targets]),encoded.decoded_field)
        return self(requested,geometry,[s.selection for s in targets])


class LegacyB1Detail(nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.detail_gain = gain

    def forward(self,predictions,encoded,bands,targets):
        # The historical algorithm only reads legal visible values, explicitly cleaned at the wrapper.
        from types import SimpleNamespace
        channels = [SimpleNamespace(values=b.clean,grid=b.grid) for b in bands]
        return Decoder.add_native_detail(self,predictions,encoded,channels,
                                         [b.visible for b in bands],targets)


class FRPretraining(nn.Module):
    def __init__(self, config, options, *, seed):
        super().__init__()
        if (tuple(config.widths)!= (32,64,96) or tuple(config.heads)!=(2,4,4)
            or tuple(config.depths)!=(2,2,2) or tuple(config.reductions)!=(8,4,2)
            or tuple(config.spatial_scopes)!=("mit",)*3 or config.kv_reduction!="conv"
            or config.grid_metres!=40 or config.decoder_mode!="continuous_native"
            or config.decoder_width!=32 or config.decoder_subcell_width!=8 or config.decoder_subcells!=4
            or config.learned_band_names):
            raise ValueError("V3-FR requires the bound historical Patch Latent configuration")
        self.config,self.options = config,options
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = Encoder(config)
            old = Decoder(config)
            self.legacy = LegacyB1Detail(old.detail_gain)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+11003)
            self.position = (nn.ModuleList(PhysicalPosition(w) for w in config.widths)
                             if options.position_mode=="qk_physical" else None)
            torch.manual_seed(seed+12011)
            self.decoder = TerminalDecoder(old,options.terminal_expander)
            if options.frontend=="ordered":
                torch.manual_seed(seed+13001)
                self.encoder.spectral = PhaseAwareFusion(options.blocks,options.film)
        if not options.legacy_b1_detail:
            self.legacy.requires_grad_(False)

    def encode(self, bands, descriptor, geometry, metrics=None):
        spectral = self.encoder.spectral
        if self.options.frontend=="historical":
            from types import SimpleNamespace
            cleaned = [SimpleNamespace(values=b.clean,grid=b.grid) for b in bands]
            features,footprint = spectral(cleaned,[b.visible for b in bands],descriptor,geometry,metrics)
        else:
            slots = spectral(bands,descriptor,geometry,channel_chunk=self.config.channel_chunk,metrics=metrics)
            features,footprint = slots.backbone_input(),slots.structural.any(1)
        footprint = footprint.reshape(1,1,*geometry.shape)
        pyramid,supports = [],[]
        for level,blocks in enumerate(self.encoder.stages):
            if level:
                footprint = F.max_pool2d(footprint.float(),3,stride=2,padding=1)>0
                features = torch.where(footprint,self.encoder.merges[level-1](features),0.)
            position = self.position[level](geometry,footprint,level,blocks[0].reduction) if self.position else None
            for block in blocks:
                if position is None:
                    fn,args = block,(features,footprint)
                else:
                    def fn(x,mask,q,k,m,block=block):
                        return positioned_block(block,x,mask,q,k,m)
                    args = features,footprint,*position
                features = (checkpoint(fn,*args,use_reentrant=False)
                            if self.training and torch.is_grad_enabled() and features.numel()>1048576
                            else fn(*args))
            pyramid.append(features)
            supports.append(footprint)
        if metrics is not None:
            metrics["stage_shapes"] = [list(f.shape) for f in pyramid]
            metrics["stage_pairs"] = [sum(b.attention_pairs(*f.shape[-2:]) for b in blocks)
                                      for f,blocks in zip(pyramid,self.encoder.stages,strict=True)]
        return Encoded(tuple(pyramid),tuple(supports),geometry,descriptor)

    def views(self,sample,keeps,targets,metrics=None):
        geometry = metric_geometry(sample.channels,self.config.grid_metres)
        descriptor = self.encoder.descriptor([c.physical for c in sample.channels])
        bands = [VisibleBand.from_native(c.values,c.valid,keep,c.grid)
                 for c,keep in zip(sample.channels,keeps,strict=True)]
        encoded = self.encode(bands,descriptor,geometry,metrics)
        specs = [TargetSpec(c.grid,t,d) for c,t,d in zip(sample.channels,targets,descriptor,strict=True)]
        return bands,encoded,specs

    def forward(self,sample,*,kind,seed,epoch=0,metrics=None,main_path_only=False):
        keeps,targets = task_masks(sample,self.config,kind,seed,epoch)
        bands,encoded,specs = self.views(sample,keeps,targets,metrics)
        main = self.decoder.predict(encoded,specs)
        if not main:
            raise ValueError("No native supervision targets")
        total = (self.legacy(main,encoded,bands,targets)
                 if kind=="B1" and self.options.legacy_b1_detail and not main_path_only else main)
        ids = sorted(total)
        truth = {i:sample.channels[i].values[targets[i]].detach().float() for i in ids}
        prediction = torch.cat([total[i] for i in ids]).float()
        expected = torch.cat([truth[i] for i in ids])
        lengths = torch.tensor([total[i].numel() for i in ids],device=prediction.device)
        mse = (prediction-expected).square()
        delta = self.config.huber_delta
        pixels = mse if delta is None else F.huber_loss(prediction,expected,delta=delta,reduction="none")
        loss = torch.segment_reduce(pixels,"mean",lengths=lengths).mean()
        if kind=="B1" and self.config.gradient_loss_weight:
            grouped = defaultdict(list)
            for i in ids:
                grouped[sample.channels[i].grid].append(i)
            terms = [native_gradient_loss(torch.cat([total[i].float() for i in group]),
                         torch.cat([truth[i] for i in group]),torch.stack([targets[i] for i in group]))
                     for group in grouped.values()]
            loss = loss+self.config.gradient_loss_weight*torch.cat(terms).mean()
        result = Result(loss,dict(zip(ids,torch.segment_reduce(mse.detach() if delta else mse,
                        "mean",lengths=lengths).unbind())),prediction,tuple(targets),encoded)
        result.main_prediction = torch.cat([main[i] for i in ids]).float()
        result.detail_contribution = prediction-result.main_prediction
        return result
