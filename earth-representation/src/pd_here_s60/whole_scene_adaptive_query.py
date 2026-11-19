"""120 m native-grid dual branch with query pooling and a shared native decoder."""
from dataclasses import replace
from math import isfinite, sqrt

import torch
from torch import nn
from torch.nn import functional as F

from pd_here_geotoken_dualvit_r3.model import Block

from .whole_scene import native_axis, tiles
from .whole_scene_runtime import band_slots
from .whole_scene_spectral_conv import moments, pack_spectral_conv_scene, positions


MODEL_IDS = {width: f'S120-AdaptiveQueryDualViT-{width}' for width in (256, 512, 1024)}
NATIVE_LIMIT = 512


def pack_adaptive_scene(observation, scale=1., source=None):
    if not observation.bands or not isfinite(scale) or scale <= 0:
        raise ValueError('Expected at least one native band and a positive radiometric scale')
    scene = pack_spectral_conv_scene(observation, scale, source or observation.sensor)
    wavelengths = scene.metadata.wavelengths
    if not (torch.isfinite(wavelengths) & (wavelengths > 0)).all():
        raise ValueError('Every band requires a finite positive centre wavelength in µm')
    return scene


def restore_native_bands(observation, scene, predictions):
    """Recover each input band's exact native raster shape and QA support."""
    all_patches = predictions.new_full(scene.values.shape, float('nan'))
    all_patches[scene.qa.any(dim=-1)] = predictions
    all_patches = torch.where(scene.qa, all_patches, float('nan'))
    widths, _ = band_slots(scene.layout)
    left, top = scene.origin
    rasters = []
    for band, (_, ph, pw), values in zip(observation.bands, scene.layout,
                                          all_patches.split(widths, -1), strict=True):
        height, width = band.grid.shape
        dx, _, x, _, dy, y = band.grid.affine
        iy, sy = native_axis(height, -dy, top-y, ph, scene.patch_m)
        ix, sx = native_axis(width, dx, x-left, pw, scene.patch_m)
        patch = (iy[:, None]*scene.grid[1] + ix[None]).to(values.device)
        slot = (sy[:, None]*pw + sx[None]).to(values.device)
        rasters.append(values[patch, slot])
    return tuple(rasters)


def adaptive_tiles(scene, keep):
    """Keep physical patches intact even when one native patch spans several output blocks."""
    max_h = max(h for _, h, _ in scene.layout)
    max_w = max(w for _, _, w in scene.layout)
    if max(max_h, max_w) <= NATIVE_LIMIT:
        yield from tiles(scene, keep)
        return
    height, width = scene.grid
    step_h, step_w = max(1, NATIVE_LIMIT//max_h), max(1, NATIVE_LIMIT//max_w)
    ids = torch.arange(height*width, device=keep.device).reshape(scene.grid)
    for r in range(0, height, step_h):
        r1 = min(r+step_h, height)
        for c in range(0, width, step_w):
            c1 = min(c+step_w, width)
            index = ids[r:r1, c:c1].flatten()
            origin = (scene.origin[0] + c*scene.patch_m, scene.origin[1] - r*scene.patch_m)
            yield replace(scene, grid=(r1-r, c1-c), values=scene.values[index], qa=scene.qa[index],
                          origin=origin, metadata=scene.metadata.crop(r, r1, c, c1)), keep[index], index


def native_task_mse(model, scene, view, task, backward_scale=None):
    widths, slots = band_slots(scene.layout)
    slots = slots.to(scene.values.device)
    keep, bands = view
    hidden = ~(keep[:, None] if keep.ndim == 1 else keep) if task == 'A' else ~bands[slots]
    targets = scene.qa & hidden
    counts = torch.stack([qa.sum() for qa in targets.split(widths, -1)])
    active = (counts > 0).sum()
    if not active:
        raise ValueError(f'{task}: no valid native targets')
    pixel_weights = (counts.clamp_min(1).float().reciprocal() / active)[slots]
    total = scene.values.new_zeros(())
    for tile, visible, index in adaptive_tiles(scene, keep):
        structural = tile.qa.any(dim=-1)
        support = targets[index][structural]
        if not support.any():
            continue
        with torch.autocast(scene.values.device.type, torch.bfloat16, enabled=scene.values.is_cuda):
            predicted = model(tile, visible, bands)
        errors = torch.where(support, predicted.float() - tile.values[structural], 0.)
        loss = (errors.square() * pixel_weights).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{task}: nonfinite native tile loss')
        if backward_scale is not None:
            (loss * backward_scale).backward()
        total += loss.detach()
    return total


def axis_coordinate(basis):
    # The existing linear basis reproduces the original fractional coordinate.
    return basis @ basis.new_tensor([0., 1 / 3, 2 / 3, 1.])


def wave_features(wavelengths):
    log_wave = wavelengths.log2()[:, None]
    phase = log_wave * wavelengths.new_tensor([1., 2., 4., 8., 16., 32.])
    return torch.cat((log_wave, phase.sin(), phase.cos()), -1)


class QueryPool(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.query, std=.02)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)

    def forward(self, values, valid):
        present = valid.any(dim=-1)
        safe = valid.clone()
        safe[:, 0] |= ~present
        keys = self.norm(values)
        query = self.query.expand(values.shape[0], -1, -1)
        readout = self.attention(query, keys, keys, key_padding_mask=~safe,
                                 need_weights=False)[0][:, 0]
        return torch.where(present[:, None], readout, 0.)


class AdaptivePatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_kernel = nn.Linear(13, 64)
        self.conv_in = nn.Conv2d(34, 32, 3, padding=1)
        self.conv_out = nn.Conv2d(32, 32, 3, padding=1)
        self.spatial_pixels = QueryPool(32)
        self.spatial_group = nn.Linear(32, 128)
        self.gsd_embed = nn.Linear(2, 128)
        self.spatial_groups = QueryPool(128)
        self.spatial_out = nn.Linear(128, 1024)

        self.band_embed = nn.Linear(32, 128)
        self.wavelength_embed = nn.Linear(13, 128)
        self.spectral_block = Block(128)
        self.spectral_projection = nn.Linear(128, 1024, bias=False)
        self.spectral_gate = nn.Linear(13, 1024)

    def forward(self, values, evidence, geometry, rows, cols, widths, wave):
        value_bands, qa_bands = values.split(widths, -1), evidence.split(widths, -1)
        spatial, present, spectral, support, order = [], [], [], [], []
        for group in geometry.groups:
            ids = list(group.bands)
            order.extend(ids)
            height, width = group.y.size(1), group.x.size(1)
            quality = torch.stack([qa_bands[i] for i in ids], 1).reshape(-1, len(ids), height, width)
            data = torch.stack([value_bands[i] for i in ids], 1).reshape_as(quality)
            features = torch.cat((data, quality.to(data.dtype)), 1)
            y_basis, x_basis = group.y[rows], group.x[cols]

            counts = quality.sum((-1, -2))
            summary = moments(features, y_basis, x_basis).float()
            summary = summary.reshape(len(data), 2, len(ids), 16).transpose(1, 2).flatten(-2)
            spectral.append(self.band_embed(summary / counts.clamp_min(1)[..., None]))
            support.append(counts > 0)

            kernel = self.channel_kernel(wave[ids]).reshape(len(ids), 2, 32)
            kernel = kernel.permute(2, 1, 0).reshape(32, 2 * len(ids), 1, 1)
            coverage = quality.sum(1, keepdim=True)
            good = coverage > 0
            base = F.conv2d(features, kernel) / coverage.clamp_min(1)
            y, x = torch.broadcast_tensors(axis_coordinate(y_basis)[:, :, None],
                                            axis_coordinate(x_basis)[:, None, :])
            coords = torch.stack((x, y), dim=1).to(base.dtype)
            coords = torch.where(good, coords, 0.)
            residual = self.conv_in(torch.cat((base, coords), 1))
            residual = torch.where(good, F.gelu(residual), 0.)
            native = torch.where(good, base + self.conv_out(residual), 0.)
            valid = good.flatten(1)
            group_features = self.spatial_pixels(native.flatten(2).transpose(1, 2), valid)
            spatial.append(self.spatial_group(group_features) + self.gsd_embed(group.log_gsd))
            present.append(valid.any(dim=1))

        spatial_token = self.spatial_out(self.spatial_groups(torch.stack(spatial, 1),
                                                             torch.stack(present, 1)))
        valid = torch.cat(support, 1)
        ordered_wave = wave[order]
        spectral_tokens = torch.cat(spectral, 1) + self.wavelength_embed(ordered_wave)
        spectral_tokens = self.spectral_block(spectral_tokens, valid)
        # Each block W(lambda) is a shared projection with a wavelength-conditioned output gate.
        projected = self.spectral_projection(spectral_tokens) * (1 + self.spectral_gate(ordered_wave))
        spectral_token = projected.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        return torch.stack((spatial_token, spectral_token), 1)


class AdaptiveQueryDualViT(nn.Module):
    def __init__(self, fusion_width=256):
        super().__init__()
        if fusion_width not in MODEL_IDS:
            raise ValueError('Fusion width must be 256, 512 or 1024')
        self.fusion_width = fusion_width
        self.patch_encoder = AdaptivePatchEncoder()
        self.patch_fusion = Block(1024)
        self.fusion = nn.Linear(2048, fusion_width)
        self.global_blocks = nn.Sequential(Block(fusion_width), Block(fusion_width))
        self.encoder_norm = nn.LayerNorm(fusion_width, eps=1e-6)

        self.decoder_embed = nn.Linear(fusion_width, 128)
        self.mask_token = nn.Parameter(torch.empty(1, 128))
        nn.init.normal_(self.mask_token, std=.02)
        self.decoder_availability = nn.Sequential(nn.Linear(16, 128), nn.GELU(), nn.Linear(128, 128))
        self.decoder_blocks = nn.Sequential(Block(128), Block(128))
        self.decoder_norm = nn.LayerNorm(128, eps=1e-6)

        self.output_latent = nn.Linear(128, 128)
        self.output_band = nn.Sequential(nn.Linear(15, 128), nn.GELU(), nn.Linear(128, 256))
        self.output_y = nn.Sequential(nn.Linear(1, 128), nn.GELU(), nn.Linear(128, 128))
        self.output_x = nn.Sequential(nn.Linear(1, 128), nn.GELU(), nn.Linear(128, 128))

    def fuse_patches(self, values, evidence, geometry, rows, cols, widths, wave):
        branches = self.patch_encoder(values, evidence, geometry, rows, cols, widths, wave)
        return self.fusion(self.patch_fusion(branches).flatten(1))

    def availability(self, geometry, wave, bands):
        band_gsd = wave.new_empty(len(wave), 2)
        for group in geometry.groups:
            band_gsd[list(group.bands)] = group.log_gsd
        physical = torch.cat((wave, band_gsd), -1).expand(*bands.shape, -1)
        return self.decoder_availability(torch.cat((physical, bands[..., None]), -1)).mean(-2)

    def native_output(self, decoded, groups, wave, rows, cols):
        latent = self.output_latent(decoded)
        outputs = [None] * len(wave)
        for group in groups:
            ids = list(group.bands)
            gsd = group.log_gsd.expand(len(ids), -1)
            gain, bias = self.output_band(torch.cat((wave[ids], gsd), -1)).chunk(2, dim=-1)
            coefficients = latent[:, None] * (1 + gain) + bias
            y = self.output_y(axis_coordinate(group.y[rows])[..., None])
            x = self.output_x(axis_coordinate(group.x[cols])[..., None])
            vertical = coefficients[:, :, None, :] * y[:, None, :, :]
            horizontal = x.transpose(1, 2)[:, None]
            if max(y.shape[1], x.shape[1]) <= NATIVE_LIMIT:
                pixels = vertical @ horizontal
            else:
                pixels = torch.cat([
                    torch.cat([v @ h for h in horizontal.split(NATIVE_LIMIT, dim=-1)], -1)
                    for v in vertical.split(NATIVE_LIMIT, dim=-2)
                ], -2)
            pixels = pixels / sqrt(128)
            for band_id, values in zip(ids, pixels.flatten(2).unbind(1), strict=True):
                outputs[band_id] = values
        return torch.cat(outputs, -1)

    def forward(self, scene, keep, bands):
        geometry = scene.metadata
        device = scene.values.device
        structural = scene.qa.any(dim=-1)
        output_ids = structural.nonzero().flatten()
        if not len(output_ids):
            return scene.values[:0]
        widths, slots = band_slots(scene.layout)
        visible = keep[:, None] if keep.ndim == 1 else keep
        evidence = scene.qa & visible & bands[slots.to(device)]
        has_evidence = evidence.any(dim=-1)
        ids = has_evidence.nonzero().flatten()
        wave = wave_features(geometry.wavelengths)
        if len(ids):
            qa = evidence[ids]
            pixels = torch.where(qa, scene.values[ids], 0.)
            fused = self.fuse_patches(pixels, qa, geometry, ids // scene.grid[1],
                                      ids % scene.grid[1], widths, wave)
            fused = fused + positions(scene.grid, self.fusion_width, device)[ids]
            encoded = self.encoder_norm(self.global_blocks(fused[None])[0])
        else:
            encoded = scene.values.new_empty(0, self.fusion_width)

        restore = self.mask_token.expand(len(output_ids), -1).clone()
        restore[has_evidence[structural]] = self.decoder_embed(encoded).to(restore.dtype)
        availability = self.availability(geometry, wave, bands)
        position = positions(scene.grid, 128, device)[output_ids]
        decoded = self.decoder_norm(self.decoder_blocks((restore + position + availability)[None])[0])
        return self.native_output(decoded, geometry.groups, wave, output_ids // scene.grid[1],
                                  output_ids % scene.grid[1])
