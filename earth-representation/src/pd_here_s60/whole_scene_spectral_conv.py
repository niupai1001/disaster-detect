"""120 m spectral Transformer and native-grid residual spatial convolution."""
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache, partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from pd_here_geotoken_dualvit_r3.model import Block, grid_position
from .whole_scene import LayoutProjection, WholeSceneViT, native_axis, pack_scene


MODEL_ID = 'S120-SpectralTransformer-SpatialConv'
WIDE_MODEL_ID = MODEL_ID + '-512'
WIDE1024_ID = MODEL_ID + '-1024'
MODEL_NAME = '光谱Transformer+空间卷积'
MSI = dict(zip(
    ('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'),
    (.490, .560, .665, .705, .740, .783, .842, .865, 1.610, 2.190), strict=True))
OLI = {f'SR_B{i}': wavelength for i, wavelength in enumerate(
    (.443, .482, .561, .655, .865, 1.609, 2.201), 1)}
SPOT = dict(B0=.490, B1=.560, B2=.660, B3=.825)
VENUS = dict(B3=.4919, B4=.555, B7=.6662, B8=.702, B9=.7411, B10=.7822, B11=.8611)
REFERENCE_URLS = {
    'MSI nominal': 'https://s2.pages.eopf.copernicus.eu/msi/s2msi/1.3.0/context.html',
    'OLI nominal': 'https://pubs.usgs.gov/publication/ofr20261069/full',
    'SPOT6/7 nominal interval midpoint': 'https://step.esa.int/main/doc/list-of-supported-satellite-products-and-formats-v2/',
    'VENUS instrument equivalent': 'https://www.cesbio.cnrs.fr/multitemp/ven%C2%B5s-characteristics/',
}


def band_center(band, source):
    center = band.physics.center
    if center.summary:
        return center.summary[0], f'release {center.status}'
    if source == 'BigEarthNet-L' and band.key in OLI:
        return OLI[band.key], 'OLI nominal'
    if source == 'FLAIR-HUB SPOT' and band.key in SPOT:
        return SPOT[band.key], 'SPOT6/7 nominal interval midpoint'
    if source.startswith('SEN2VENUS') and '_fit_' in band.key:
        return VENUS[band.key.split('_fit_')[0]], 'VENUS instrument equivalent'
    if (source == 'FLAIR-HUB S2' or source.startswith('SEN2VENUS')) and band.key in MSI:
        return MSI[band.key], 'MSI nominal'
    raise ValueError(f'{source}/{band.key}: missing physical center wavelength')


def axis_basis(count, slots, size, spacing, shift, patch_m):
    cell, slot = native_axis(count, spacing, shift, slots, patch_m)
    centers = shift + (torch.arange(count, dtype=torch.float64) + .5) * spacing
    relative = ((centers - cell * patch_m) / patch_m).float().clamp(0, 1)
    weights = (1 - (relative[:, None] * 3 - torch.arange(4)).abs()).clamp_min(0)
    result = torch.zeros(size, slots, 4)
    result[cell, slot] = weights
    return result


@dataclass(frozen=True)
class NativeGroup:
    bands: tuple
    y: torch.Tensor
    x: torch.Tensor
    log_gsd: torch.Tensor


@dataclass(frozen=True)
class PatchGeometry:
    groups: tuple[NativeGroup, ...]
    wavelengths: torch.Tensor
    wavelength_sources: tuple[str, ...]

    def to(self, device):
        groups = tuple(replace(g, y=g.y.to(device), x=g.x.to(device), log_gsd=g.log_gsd.to(device))
                       for g in self.groups)
        return replace(self, groups=groups, wavelengths=self.wavelengths.to(device))

    def crop(self, r0, r1, c0, c1):
        return replace(self, groups=tuple(replace(g, y=g.y[r0:r1], x=g.x[c0:c1]) for g in self.groups))


def pack_spectral_conv_scene(observation, scale, source):
    scene = pack_scene(observation, scale, patch_m=120)
    groups = defaultdict(list)
    centers, provenance = zip(*(band_center(b, source) for b in observation.bands), strict=True)
    for i, b in enumerate(observation.bands):
        groups[(tuple(b.grid.shape), tuple(b.grid.affine), b.grid.crs)].append(i)
    left, top = scene.origin
    gh, gw = scene.grid
    geometry = []
    for ((h, w), (dx, _, x, _, dy, y), _), ids in groups.items():
        _, ph, pw = scene.layout[ids[0]]
        geometry.append(NativeGroup(
            tuple(ids), axis_basis(h, ph, gh, -dy, top-y, scene.patch_m),
            axis_basis(w, pw, gw, dx, x-left, scene.patch_m), torch.tensor([-dy, dx]).log2()))
    return replace(scene, metadata=PatchGeometry(tuple(geometry), torch.tensor(centers), provenance))


@lru_cache(maxsize=32)
def positions(grid, width, device):
    return grid_position(grid, width).to(device)


def moments(values, y, x):
    # A separable continuous projection of all native slots, without resampling the image.
    return (y.transpose(-2, -1)[:, None] @ (values @ x[:, None])).flatten(-2)


class SpectralConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavelength_embed = nn.Linear(13, 128)
        self.band_embed = nn.Linear(32, 128)
        self.spectral_transformer = Block(128)
        self.spectral_weights = nn.Linear(13, 8)
        self.spectral_projection = nn.Linear(1024, 1024)

        self.channel_kernel = nn.Linear(13, 64)
        self.spatial_conv = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                                          nn.Conv2d(32, 32, 3, padding=1))
        self.gsd_weights = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 16))
        self.spatial_projection = nn.Linear(32 * 16, 1024)

    def forward(self, values, evidence, geometry, rows, cols, widths):
        wavelength = geometry.wavelengths.log2()[:, None]
        phase = wavelength * torch.tensor([1., 2., 4., 8., 16., 32.], device=values.device)
        wave = torch.cat((wavelength, phase.sin(), phase.cos()), -1)
        value_bands, qa_bands = values.split(widths, -1), evidence.split(widths, -1)
        spectral, support, spatial, present = [], [], [], []
        band_order = []
        for group in geometry.groups:
            ids = list(group.bands)
            band_order.extend(ids)
            n, h, w = len(ids), group.y.size(1), group.x.size(1)
            qa = torch.stack([qa_bands[i] for i in ids], 1).reshape(-1, n, h, w)
            x = torch.stack([value_bands[i] for i in ids], 1).reshape_as(qa)
            features = torch.cat((x, qa.to(x.dtype)), 1)
            y_basis, x_basis = group.y[rows], group.x[cols]
            count = qa.sum((-1, -2))
            projected = moments(features, y_basis, x_basis).float()
            projected = projected.reshape(len(x), 2, n, 16).transpose(1, 2).flatten(-2)
            spectral.append(self.band_embed(projected / count.clamp_min(1)[..., None]))
            support.append(count > 0)

            # The wavelength-generated 1×1 kernel mixes all co-registered bands together.
            kernel = self.channel_kernel(wave[ids]).reshape(n, 2, 32).permute(2, 1, 0)
            kernel = kernel.reshape(32, 2*n, 1, 1)
            coverage = qa.sum(1, keepdim=True)
            native = F.conv2d(features, kernel) / coverage.clamp_min(1)
            good = coverage > 0
            native = native.contiguous(memory_format=torch.channels_last)
            residual = self.spatial_conv[1](self.spatial_conv[0](native))
            residual = torch.where(good, residual, 0.)
            native = torch.where(good, native + self.spatial_conv[2](residual), 0.)
            size = good.sum((-1, -2))
            pooled = moments(native, y_basis, x_basis).float() / size.clamp_min(1)[..., None]
            scale = 1 + self.gsd_weights(group.log_gsd)
            spatial.append((pooled * scale).flatten(1))
            present.append(size > 0)
        spatial = torch.stack(spatial).sum(0) / torch.stack(present).sum(0).clamp_min(1)
        valid = torch.cat(support, 1)
        ordered_wave = wave[band_order]
        tokens = torch.cat(spectral, 1) + self.wavelength_embed(ordered_wave)[None]
        tokens = self.spectral_transformer(tokens, valid)
        pooled = tokens.transpose(1, 2) @ self.spectral_weights(ordered_wave)
        pooled = pooled.float().flatten(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.spectral_projection(pooled), self.spatial_projection(spatial)


class WholeSceneSpectralConv(WholeSceneViT):
    projection_type = staticmethod(partial(LayoutProjection, joint_input=False))

    def __init__(self, layouts, *, fusion_width=256, decoder_width=128):
        if fusion_width not in (256,512,1024):
            raise ValueError('Fusion width must be 256, 512 or 1024')
        super().__init__(layouts, decoder_width=decoder_width)
        del self.spatial_block
        self.patch_encoder = SpectralConvEncoder()
        self.fusion = nn.Linear(2048, fusion_width)
        self.global_blocks = nn.Sequential(Block(fusion_width), Block(fusion_width))
        self.encoder_norm = nn.LayerNorm(fusion_width, eps=1e-6)
        self.decoder_embed = nn.Linear(fusion_width, decoder_width)

    def forward(self, scene, keep, bands):
        head = self.projections[self.layout_indices[scene.layout]]
        structural = scene.qa.any(-1)
        visible = keep[:, None] if keep.ndim == 1 else keep
        evidence = scene.qa & visible & bands[head.band_slots]
        ids = evidence.any(-1).nonzero().flatten()
        device = scene.values.device
        width = self.fusion.out_features
        if len(ids):
            qa = evidence[ids]
            x = torch.where(qa, scene.values[ids], 0.)
            row, col = ids // scene.grid[1], ids % scene.grid[1]
            branches = self.patch_encoder(x, qa, scene.metadata, row, col, head.widths)
            tokens = self.fusion(torch.cat(branches, -1)) + positions(scene.grid, width, device)[ids]
        else:
            tokens = scene.values.new_empty(0, width)
        position = positions(scene.grid, self.decoder_width, device)[structural]
        availability = head.decoder_availability(bands.float())
        selected = evidence.any(-1)[structural]
        if tokens.is_cuda:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                decoded = self.transform(tokens, selected, position, availability)
        else:
            decoded = self.transform(tokens, selected, position, availability)
        return head.output(decoded)

    def transform(self, tokens, selected, position, availability):
        encoded = self.encoder_norm(self.global_blocks(tokens[None])[0]) if len(tokens) else tokens
        return self.decode_latent(encoded, selected, position, availability)
