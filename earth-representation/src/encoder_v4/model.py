"""Physical descriptors, dual native stems, Group-Swin, and native readout."""
from collections import defaultdict
from dataclasses import dataclass
from math import log

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import WorkGrid, footprint_map


@dataclass(frozen=True)
class Config:
    slots: int = 16
    stride: int = 2
    spectral: str = 'band_self_then_latent'
    readout: str = 'all'
    descriptor: str = 'correct'
    position_chunk: int = 128
    target_chunk: int = 16384

    def __post_init__(self):
        if (self.slots not in (4, 8, 16, 32) or self.stride not in (1, 2)
                or self.spectral not in ('band_self_then_latent', 'latent_only')
                or self.readout not in ('all', 'fine')
                or self.descriptor not in ('correct', 'missing', 'shuffled')
                or min(self.position_chunk, self.target_chunk) < 1):
            raise ValueError('Invalid PD-HERE model configuration')


@dataclass(frozen=True)
class Stage:
    values: torch.Tensor
    grid: WorkGrid
    support: torch.Tensor


@dataclass(frozen=True)
class EncoderPyramid:
    stages: tuple[Stage, Stage, Stage]


def mlp(width, hidden, output):
    return nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, output))


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, width * 2)
        self.out = nn.Linear(width, width)

    def forward(self, x, context=None, valid=None, bias=None):
        if context is None:
            context = x
        q = self.q(x).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k, v = (z.unflatten(-1, (self.heads, -1)).transpose(1, 2)
                for z in self.kv(context).chunk(2, dim=-1))
        if valid is not None:
            allowed = valid[:, None, None, :]
            bias = allowed if bias is None else bias.masked_fill(~allowed, -torch.inf)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        y = self.out(y.transpose(1, 2).flatten(2))
        if valid is not None:
            y = y.masked_fill(~valid.any(-1)[:, None, None], 0.)
        return y


class Descriptor(nn.Module):
    names = ('center', 'bandwidth', 'srf', 'gsd', 'psf', 'mtf')
    states = ('missing', 'observed', 'estimated', 'not_applicable')

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.fields = nn.ModuleList(mlp(8, 32, 32) for _ in self.names)
        self.curves = nn.ModuleDict({k: mlp(3, 32, 32) for k in ('srf', 'psf', 'mtf')})
        self.spec = mlp(96, 64, 64)
        self.spatial = mlp(96, 64, 64)
        self.spec_relation = Attention(64, 4)
        self.spatial_relation = Attention(64, 4)
        self.joint = mlp(128, 64, 64)
        self.scale = nn.Linear(64, 64)

    def encode(self, specs):
        like = self.scale.weight
        outputs = []
        for name, project in zip(self.names, self.fields):
            attrs = [getattr(physics, name) for physics in specs]
            if self.mode == 'missing':
                attrs = [type(attr)() for attr in attrs]
            flags = []
            for attr in attrs:
                summary = list(attr.summary)
                if name in ('gsd', 'bandwidth'):
                    summary = [log(value) for value in summary]
                flags.append((summary + [0., 0.])[:2] +
                             [float(attr.status == s) for s in self.states] +
                             [float(bool(attr.samples)), float(bool(attr.summary))])
            features = project(like.new_tensor(flags))
            if name in self.curves:
                lengths = [len(a.samples) for a in attrs]
                if sum(lengths):
                    points = like.new_tensor([p for a in attrs for p in a.samples])
                    values = self.curves[name](points[:, :3])
                    weights = points[:, 3:]
                    ids = torch.repeat_interleave(torch.arange(len(attrs), device=like.device),
                                                  torch.tensor(lengths, device=like.device), output_size=len(points))
                    total = features.new_zeros(features.shape).index_add_(0, ids, values * weights)
                    mass = weights.new_zeros((len(attrs), 1)).index_add_(0, ids, weights)
                    features = features + total / mass.clamp_min(1e-12)
            outputs.append(features)
        return self.spec(torch.cat(outputs[:3], -1)), self.spatial(torch.cat(outputs[3:], -1))

    def forward(self, specs):
        if self.mode == 'shuffled':
            specs = specs[1:] + specs[:1]
        spec, spatial = self.encode(specs)
        groups = {}
        route, indices = [], []
        for index, physics in enumerate(specs):
            key = tuple((a.status, a.unit, a.summary, a.samples) for a in (physics.gsd, physics.psf, physics.mtf))
            if key not in groups:
                groups[key] = len(indices)
                indices.append(index)
            route.append(groups[key])
        group = spatial[indices]
        group = group + self.spatial_relation(group[None])[0]
        spatial = group[route]
        spec = spec + self.spec_relation(spec[None])[0]
        joint = self.joint(torch.cat((spec, spatial), -1))
        return spec, spatial, joint.mean(0), self.scale(group.mean(0))

    def targets(self, requests):
        specs = [r.physics for r in requests]
        if self.mode == 'shuffled':
            specs = specs[1:] + specs[:1]
        spec, spatial = self.encode(specs)
        return self.joint(torch.cat((spec, spatial), -1))


class GroupBlock(nn.Module):
    def __init__(self, width, heads, shifted):
        super().__init__()
        self.slot = Attention(width, heads)
        self.window = Attention(width, heads)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        self.ffn = mlp(width, 4 * width, width)
        self.slot_q = nn.Linear(64, heads)
        self.slot_k = nn.Linear(64, heads)
        self.slot_condition = nn.Linear(64, heads)
        self.scale_condition = nn.Linear(64, heads)
        self.shifted = shifted

    def forward(self, x, valid, roles, joint, scale, spacing):
        h, w, slots, width = x.shape
        slot_bias = self.slot_q(roles).T[:, :, None] * self.slot_k(roles).T[:, None, :]
        slot_bias = slot_bias * torch.tanh(self.slot_condition(joint))[:, None, None]
        n = self.norms[0](x).reshape(h * w, slots, width)
        x = x + self.slot(n, bias=slot_bias[None]).reshape_as(x)
        n = self.norms[1](x).permute(2, 3, 0, 1)
        wh, ww = min(8, h), min(8, w)
        sy = wh // 2 if self.shifted and h > wh else 0
        sx = ww // 2 if self.shifted and w > ww else 0
        padding = (sx, (-(w + sx)) % ww, sy, (-(h + sy)) % wh)
        n = F.pad(n, padding)
        hp, wp = n.shape[-2:]
        nh, nw = hp // wh, wp // ww
        windows = n.reshape(slots, width, nh, wh, nw, ww).permute(2, 4, 0, 3, 5, 1)
        windows = windows.reshape(nh * nw * slots, wh * ww, width)
        window_valid = F.pad(valid, padding).reshape(nh, wh, nw, ww).permute(0, 2, 1, 3)
        window_valid = window_valid.reshape(nh * nw, wh * ww).repeat_interleave(slots, dim=0)
        rows, cols = torch.meshgrid(torch.arange(wh, device=x.device), torch.arange(ww, device=x.device), indexing='ij')
        dx, dy = spacing
        distance = (rows.flatten()[:, None] - rows.flatten()).abs() * dy
        distance = distance + (cols.flatten()[:, None] - cols.flatten()).abs() * dx
        bias = -F.softplus(self.scale_condition(scale))[:, None, None] * torch.log1p(distance / 10.)
        n = self.window(windows, valid=window_valid, bias=bias[None])
        n = n.reshape(nh, nw, slots, wh, ww, width).permute(0, 3, 1, 4, 2, 5)
        x = x + n.reshape(hp, wp, slots, width)[sy:sy + h, sx:sx + w]
        return (x + self.ffn(self.norms[2](x))).masked_fill(~valid[..., None, None], 0.)


def merge(x, projection):
    h, w, slots, width = x.shape
    padded = F.pad(x, (0, 0, 0, 0, 0, w % 2, 0, h % 2))
    neighbors = padded.reshape((h + 1) // 2, 2, (w + 1) // 2, 2, slots, width)
    return projection(neighbors.permute(0, 2, 4, 1, 3, 5).flatten(3))


class Encoder(nn.Module):
    widths = (64, 128, 256)

    def __init__(self, config, descriptor=None):
        super().__init__()
        self.config = config
        self.descriptor = descriptor if descriptor is not None else Descriptor(config.descriptor)
        self.stems = nn.ModuleList(nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.GELU()) for _ in range(2))
        self.modulation = nn.ModuleList(nn.Linear(64, 32) for _ in range(2))
        self.stem_gain = nn.Parameter(torch.full((2, 32), .001))
        self.spec_projection = nn.Linear(32, 64)
        self.space_projection = nn.Linear(32, 64)
        self.space_score = nn.Linear(32, 1)
        self.space_physics = nn.Linear(64, 1)
        self.band_relation = Attention(64, 4) if config.spectral == 'band_self_then_latent' else None
        self.band_bias_q = nn.Linear(64, 4)
        self.band_bias_k = nn.Linear(64, 4)
        self.query_relation = Attention(64, 4)
        self.query_bias = nn.Linear(64, 4)
        self.slots = nn.Parameter(torch.randn(config.slots + 1, 64) * .02)
        self.empty = nn.Parameter(torch.randn(config.slots + 1, 64) * .02)
        self.empty_condition = nn.Parameter(torch.zeros(2, 64))
        self.fusion_ffn = nn.Sequential(nn.LayerNorm(64), mlp(64, 128, 64))
        self.steps = nn.ModuleList(nn.ModuleList(GroupBlock(c, heads, i % 2 == 1) for i in range(depth))
                                   for c, heads, depth in zip(self.widths, (4, 4, 8), (2, 2, 4)))
        self.merges = nn.ModuleList(nn.Sequential(nn.LayerNorm(c * 4), nn.Linear(c * 4, c * 2))
                                    for c in self.widths[:-1])

    def spatial_features(self, bands, work, index, condition):
        grouped = defaultdict(list)
        for i, band in enumerate(bands):
            grouped[band.grid].append(i)
        output, support = [None] * len(bands), [None] * len(bands)
        for grid, ids in grouped.items():
            native = torch.stack([torch.stack((bands[i].values, bands[i].observed)) for i in ids])
            visible = native[:, 1].bool()
            features = self.stems[index](native)
            gain = torch.tanh(self.modulation[index](condition[ids])) * self.stem_gain[index]
            features = features * (1 + gain[..., None, None])
            mapping = footprint_map(grid, work.shape, work.origin, work.spacing, features.device)
            values, weights = mapping.aggregate(features.flatten(2).transpose(1, 2), visible)
            for local, original in enumerate(ids):
                output[original], support[original] = values[local], weights[local]
        return torch.stack(output, 1), torch.stack(support, 1)

    def fuse_spectral(self, x, valid, condition):
        band_bias = self.band_bias_q(condition).T[:, :, None] * self.band_bias_k(condition).T[:, None, :]
        bias = self.query_bias(condition).T[None, :, None, :]
        slots = self.slots[1:].expand(x.shape[0], -1, -1)
        if self.band_relation is not None:
            x = x + self.band_relation(x, valid=valid, bias=band_bias[None])
        slots = slots + self.query_relation(slots, x, valid, bias)
        return slots + self.fusion_ffn(slots)

    def forward(self, visible_observation):
        bands, work = visible_observation.bands, visible_observation.work
        h, w = work.shape
        if bands:
            spec, spatial, joint, scale = self.descriptor([b.physics for b in bands])
            spectral, support = self.spatial_features(bands, work, 0, spec)
            spatial_values, _ = self.spatial_features(bands, work, 1, spatial)
            valid = support > 0
            occupied = valid.any(-1)
            spectral = self.spec_projection(spectral)
            chunks = zip(spectral.split(self.config.position_chunk), valid.split(self.config.position_chunk))
            # Recompute band attention by chunk to bound retained B² scores for hyperspectral inputs.
            if self.training and torch.is_grad_enabled() and len(bands) > 32:
                pooled = torch.cat([checkpoint(self.fuse_spectral, x, mask, spec, use_reentrant=False)
                                    for x, mask in chunks])
            else:
                pooled = torch.cat([self.fuse_spectral(x, mask, spec) for x, mask in chunks])
            logits = self.space_score(spatial_values).squeeze(-1) + self.space_physics(spatial).T
            logits = logits + support.clamp_min(1e-12).log()
            logits = logits.masked_fill(~valid, -torch.inf).masked_fill(~occupied[:, None], 0.)
            summary = (torch.softmax(logits, 1)[..., None] * spatial_values).sum(1)
            x = torch.cat((self.space_projection(summary)[:, None, :], pooled), 1)
            x = torch.where(occupied[:, None, None], x, self.empty)
            support = support.amax(1).reshape(h, w)
        else:
            joint, scale = self.empty_condition.unbind()
            x = self.empty.expand(h * w, -1, -1)
            support = x.new_zeros((h, w))
        x = x.reshape(h, w, *self.empty.shape).masked_fill(~work.valid[..., None, None], 0.)
        stages = []
        for index, blocks in enumerate(self.steps):
            for block in blocks:
                x = block(x, work.valid, self.slots, joint, scale, work.spacing)
            stages.append(Stage(x, work, support))
            if index < len(self.merges):
                x = merge(x, self.merges[index])
                h, w = work.shape
                support = F.pad(support, (0, w % 2, 0, h % 2))
                support = support.reshape((h + 1) // 2, 2, (w + 1) // 2, 2).amax((1, 3))
                work = work.merge()
                x = x.masked_fill(~work.valid[..., None, None], 0.)
        return EncoderPyramid(tuple(stages))


def align_stage(field, grid, fine):
    if grid.shape == fine.shape and grid.spacing == fine.spacing:
        return field
    h, w = fine.shape
    hh, ww = grid.shape
    x = (torch.arange(w, device=field.device, dtype=field.dtype) + .5) * (2 * fine.spacing[0] / (ww * grid.spacing[0])) - 1
    y = (torch.arange(h, device=field.device, dtype=field.dtype) + .5) * (2 * fine.spacing[1] / (hh * grid.spacing[1])) - 1
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    locations = torch.stack((xx, yy), -1)[None]
    valid = grid.valid[None, None].to(field)
    values = field.flatten(2).permute(2, 0, 1)[None]
    mass = F.grid_sample(valid, locations, padding_mode='border', align_corners=False)
    values = F.grid_sample(values * valid, locations, padding_mode='border', align_corners=False)
    return (values / mass.clamp_min(1e-12))[0].permute(1, 2, 0).reshape(h, w, *field.shape[-2:])


class Decoder(nn.Module):
    def __init__(self, config, descriptor=None):
        super().__init__()
        self.config = config
        self.descriptor = descriptor if descriptor is not None else Descriptor(config.descriptor)
        self.projection = nn.ModuleList(nn.Linear(c, 64) for c in Encoder.widths)
        self.weight = nn.Parameter(torch.zeros(3))
        self.read = Attention(64, 4)
        self.pixel = mlp(132, 64, 1)

    def forward(self, pyramid, target_requests):
        stages = pyramid.stages if self.config.readout == 'all' else pyramid.stages[:1]
        fine = stages[0].grid
        weight = self.weight[:len(stages)].softmax(0)
        field = sum(weight[i] * align_stage(self.projection[i](s.values), s.grid, fine)
                    for i, s in enumerate(stages))
        field = field.flatten(0, 1)
        condition = self.descriptor.targets(target_requests)
        results = []
        for request, desc in zip(target_requests, condition, strict=True):
            cells, incidence = request.cells.unique(sorted=True, return_inverse=True)
            queries = desc.expand(cells.numel(), 1, -1)
            features = torch.cat([self.read(q, field[c])[:, 0] for q, c in
                                  zip(queries.split(self.config.position_chunk), cells.split(self.config.position_chunk))])
            dx, dy = fine.spacing
            gx, gy = request.grid.spacing
            _, _, left, _, _, top = request.grid.affine
            pixel = request.pixels[request.output_index]
            left = (left - fine.origin[0]) / dx + (pixel % request.grid.shape[1]) * (gx / dx)
            top = (top - fine.origin[1]) / dy - (pixel // request.grid.shape[1]) * (gy / dy)
            col = (request.cells % fine.shape[1]).to(features)
            row = (request.cells // fine.shape[1]).to(features)
            x = (torch.maximum(left, col) + torch.minimum(left + gx / dx, col + 1)) * .5
            y = (torch.maximum(top - gy / dy, row) + torch.minimum(top, row + 1)) * .5
            shift = torch.stack((x - col - .5, y - row - .5,
                                 torch.full_like(x, gx / dx), torch.full_like(y, gy / dy)), -1)
            output = features.new_zeros(request.pixels.numel())
            count = self.config.target_chunk
            for edges, indices, relative, area in zip(incidence.split(count), request.output_index.split(count),
                                                       shift.split(count), request.area.split(count)):
                inputs = torch.cat((features[edges], desc.expand(len(edges), -1), relative), -1)
                output = output.index_add(0, indices, self.pixel(inputs).flatten() * area)
            results.append(output)
        return tuple(results)


class Pretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        descriptor = Descriptor(config.descriptor)
        self.encoder = Encoder(config, descriptor)
        self.decoder = Decoder(config, descriptor)

    def forward(self, view):
        return self.decoder(self.encoder(view.visible), view.requests)


def objective(predictions, target_truth, task_metadata=None):
    losses = [F.huber_loss(p, t, delta=1.) for p, t in zip(predictions, target_truth, strict=True)]
    return torch.stack(losses).mean()
