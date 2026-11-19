"""Execute compatible views together, with a separate attention sequence per view."""
from collections import defaultdict
from dataclasses import replace

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.rnn import pad_sequence

from .whole_scene import tile_bounds
from .whole_scene_runtime import band_slots, prepared_backward
from .whole_scene_spectral_conv import positions


def scene_signature(scene, *, max_patch_tokens=256):
    # Large inputs retain their original native tile execution.
    if len(tile_bounds(scene)) != 1 or scene.grid[0] * scene.grid[1] > max_patch_tokens:
        return None
    geometry = scene.metadata
    bucket = scene.grid if scene.grid[0]*scene.grid[1]>256 else None
    if geometry is None:
        return (scene.layout,bucket)
    return (scene.layout, bucket, tuple(geometry.wavelengths.tolist()),
            tuple((group.bands, tuple(group.log_gsd.tolist())) for group in geometry.groups))


def padded(rows):
    data = pad_sequence(rows, batch_first=True)
    length = data.shape[1]
    if length % 8:
        data = F.pad(data, (0, 0, 0, 8 - length % 8))
    counts = torch.tensor([len(row) for row in rows], device=data.device)
    return data, torch.arange(data.shape[1], device=data.device)[None] < counts[:, None]


def packed_forward(model, jobs):
    layout = jobs[0][1].layout
    device = jobs[0][2].values.device
    head = model.projections[model.layout_indices[layout]]
    spectral = jobs[0][2].metadata is not None
    width = model.fusion.out_features if spectral else 128
    widths, slots = band_slots(layout)
    inputs, input_qa, encoder_pos, decoder_pos = [], [], [], []
    restore_ids, targets, truth, weights, bands_in = [], [], [], [], []
    yy, xx, geometries = [], [], []
    y_offset = x_offset = native_offset = 0
    native_sources, source_offsets = [], {}
    for task, original, scene, keep, bands in jobs:
        qa = original.qa
        structural = qa.any(-1)
        visible = keep[:, None] if keep.ndim == 1 else keep
        evidence = qa & visible & bands[slots]
        present = evidence.any(-1)
        ids = present.nonzero().flatten()
        solid = structural.nonzero().flatten()
        if spectral:
            rows, cols = scene.grid
            yy.append(ids // cols + y_offset)
            xx.append(ids % cols + x_offset)
            y_offset += rows
            x_offset += cols
            geometries.append(scene.metadata)
        key = id(scene)
        if key not in source_offsets:
            source_offsets[key] = native_offset
            native_sources.append(scene.values)
            native_offset += len(scene.values)
        base = source_offsets[key]
        inputs.append(ids + base)
        input_qa.append(evidence[ids])
        encoder_pos.append(positions(scene.grid, width, 'cpu')[ids])
        decoder_pos.append(positions(scene.grid, model.decoder_width, 'cpu')[solid])
        restore_ids.append(present[structural].nonzero().flatten())
        bands_in.append(bands)
        support = qa[structural]
        if task is not None:
            hidden = ~visible if task == 0 else ~bands[slots][None]
            support = (qa & hidden)[structural]
        counts = torch.stack([slab.sum() for slab in support.split(widths, -1)])
        active = int((counts > 0).sum())
        if not active:
            raise ValueError('No native target in packed view')
        weights.append((counts.clamp_min(1).float().reciprocal() / active)[slots])
        truth.append(solid + base)
        targets.append(support)

    bands = torch.stack(bands_in).to(device).float()
    lengths = [len(idx) for idx in restore_ids]
    source = torch.cat(native_sources)
    q = torch.cat(input_qa).to(device)
    x = torch.where(q, source[torch.cat(inputs).to(device)], 0.)
    if not len(x):
        token = x.new_empty(0, width)
    elif spectral:
        first = geometries[0]
        geometry = replace(first, groups=tuple(
            replace(group, y=torch.cat([g.groups[i].y for g in geometries]),
                    x=torch.cat([g.groups[i].x for g in geometries]))
            for i, group in enumerate(first.groups)))
        branches = model.patch_encoder(x, q, geometry, torch.cat(yy).to(device),
                                       torch.cat(xx).to(device), widths)
        token = model.fusion(torch.cat(branches, -1))
    else:
        ix = torch.repeat_interleave(torch.arange(len(jobs)), torch.tensor(lengths)).to(device)
        token = head.input_values(x) + head.input_qa(q.float()) + head.availability(bands)[ix]
    token = token + torch.cat(encoder_pos).to(device)
    pos, decoder_mask = (value.to(device) for value in padded(decoder_pos))
    availability = head.decoder_availability(bands)[:, None]
    restored = model.mask_token.expand(len(jobs), pos.shape[1], -1).clone()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        if len(token):
            encoded, valid = padded(token.split(lengths))
            blocks = model.global_blocks if spectral else (model.spatial_block,)
            for block in blocks:
                encoded = block(encoded, valid)
            encoded = model.encoder_norm(encoded)
            index = torch.cat([idx + i * pos.shape[1] for i, idx in enumerate(restore_ids)]).to(device)
            restored.flatten(0, 1)[index] = model.decoder_embed(encoded[valid]).to(restored.dtype)
        decoded = restored + pos + availability
        for block in model.decoder_blocks:
            decoded = block(decoded, decoder_mask)
    predicted = head.output(model.decoder_norm(decoded))
    lengths = [len(p) for p in truth]
    owners = torch.repeat_interleave(torch.arange(len(jobs)), torch.tensor(lengths)).to(device)
    flat_weights = torch.stack(weights).to(device)[owners]
    residual = predicted[decoder_mask].float() - source[torch.cat(truth).to(device)]
    error = torch.where(torch.cat(targets).to(device), residual, 0.)
    counts = (error.square() * flat_weights).sum(-1)
    return torch.stack([part.sum() for part in counts.split(lengths)]), predicted, lengths


def batched_backward(model, group, cpu_cache, gpu_cache, signatures, views_for, epoch, parent_count=None):
    buckets = defaultdict(list)
    totals = []
    scale = .5 / (len(group) if parent_count is None else parent_count)
    for row in group:
        key = row['id']
        original, scene = cpu_cache[key], gpu_cache[key]
        views = views_for(original, key, epoch)
        signature = signatures[key]
        if signature is None:
            totals.append(prepared_backward(model, original, scene, views, scale))
        else:
            for task, (keep, bands) in enumerate(views):
                buckets[signature].append((task, original, scene, keep, bands))
    for jobs in buckets.values():
        rows,cols = jobs[0][1].grid
        limit = 8 if rows*cols>256 else len(jobs)
        for start in range(0,len(jobs),limit):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses, _, _ = packed_forward(model, jobs[start:start+limit])
            (losses.sum() * scale).backward()
            totals.append(losses.detach().reshape(-1, 2).sum(0))
    return torch.stack(totals).sum(0)
