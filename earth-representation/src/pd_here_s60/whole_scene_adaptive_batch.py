"""Pack compatible adaptive views while attending within each scene and task."""
from collections import defaultdict
from dataclasses import replace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .whole_scene_adaptive_query import NATIVE_LIMIT, native_task_mse, wave_features
from .whole_scene_batch import padded, scene_signature
from .whole_scene_runtime import band_slots
from .whole_scene_spectral_conv import positions


def adaptive_signature(scene):
    if any(max(h, w) > NATIVE_LIMIT for _, h, w in scene.layout):
        return None
    # Limit the padded attention batch and retain native tiling for very large scenes.
    return scene_signature(scene, max_patch_tokens=256)


def packed_loss(model, jobs):
    layout = jobs[0][0].layout
    device = jobs[0][1].values.device
    widths, slots = band_slots(layout)
    input_ids, input_qa, output_ids, restore_ids = [], [], [], []
    enc_pos, dec_pos, input_y, input_x, output_y, output_x = [], [], [], [], [], []
    supports, weights, band_masks = [], [], []
    y_offset = x_offset = native_offset = 0
    sources, offsets = [], {}
    for original, scene, task, (keep, bands) in jobs:
        if id(scene) not in offsets:
            offsets[id(scene)] = (native_offset, y_offset, x_offset)
            sources.append(scene)
            native_offset += len(scene.values)
            y_offset += scene.grid[0]
            x_offset += scene.grid[1]
        start, y0, x0 = offsets[id(scene)]
        visible = keep[:, None] if keep.ndim == 1 else keep
        evidence = original.qa & visible & bands[slots]
        active = evidence.any(-1)
        ids = active.nonzero().flatten()
        structural = original.qa.any(-1)
        solid = structural.nonzero().flatten()
        input_ids.append(ids + start)
        input_qa.append(evidence[ids])
        output_ids.append(solid + start)
        restore_ids.append(active[structural].nonzero().flatten())
        enc_pos.append(positions(scene.grid, model.fusion_width, 'cpu')[ids])
        dec_pos.append(positions(scene.grid, 128, 'cpu')[solid])
        input_y.append(ids // scene.grid[1] + y0)
        input_x.append(ids % scene.grid[1] + x0)
        output_y.append(solid // scene.grid[1] + y0)
        output_x.append(solid % scene.grid[1] + x0)
        support = original.qa[structural] & (~visible[structural] if task == 'A' else ~bands[slots])
        counts = torch.stack([item.sum() for item in support.split(widths, -1)])
        valid_bands = int((counts > 0).sum())
        if not valid_bands:
            raise ValueError(f'{task}: no valid native targets')
        weights.append((counts.clamp_min(1).float().reciprocal() / valid_bands)[slots])
        supports.append(support)
        band_masks.append(bands)

    def join(parts):
        return torch.cat(parts).to(device)

    values = torch.cat([scene.values for scene in sources])
    first = sources[0].metadata
    geometry = replace(first, groups=tuple(replace(
        group, y=torch.cat([scene.metadata.groups[i].y for scene in sources]),
        x=torch.cat([scene.metadata.groups[i].x for scene in sources]))
        for i, group in enumerate(first.groups)))
    wave = wave_features(geometry.wavelengths)
    position, valid = (part.to(device) for part in padded(dec_pos))
    batch = model.mask_token.expand(len(jobs), position.shape[1], -1).clone()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        if any(len(ids) for ids in input_ids):
            qa = join(input_qa)
            pixels = torch.where(qa, values[join(input_ids)], 0.)
            enc = model.fuse_patches(pixels, qa, geometry, join(input_y), join(input_x), widths, wave)
            enc = enc + join(enc_pos)
            enc, enc_valid = padded(enc.split([len(ids) for ids in input_ids]))
            for block in model.global_blocks:
                enc = block(enc, enc_valid)
            restored = join([ids + i * position.shape[1] for i, ids in enumerate(restore_ids)])
            batch.flatten(0, 1)[restored] = model.decoder_embed(model.encoder_norm(enc[enc_valid])).to(batch.dtype)
        availability = model.availability(geometry, wave, torch.stack(band_masks).to(device))
        batch = batch + position + availability[:, None]
        for block in model.decoder_blocks:
            batch = block(batch, valid)
    predicted = model.native_output(model.decoder_norm(batch[valid]), geometry.groups, wave,
                                    join(output_y), join(output_x))
    lengths = [len(ids) for ids in output_ids]
    parent_weights = torch.repeat_interleave(torch.stack(weights), torch.tensor(lengths), dim=0).to(device)
    errors = torch.where(join(supports), predicted.float() - values[join(output_ids)], 0.)
    per_patch = (errors.square() * parent_weights).sum(-1)
    return torch.stack([part.sum() for part in per_patch.split(lengths)])


def adaptive_batch_totals(model, rows, cpu, resident, signatures, views_for, exposure, *, backward):
    totals, buckets = [], defaultdict(list)
    scale = .5/len(rows) if backward else None
    device = next(model.parameters()).device
    with torch.autocast('cuda', torch.bfloat16, enabled=device.type == 'cuda'):
        for row in rows:
            key = row['id']
            original = cpu[key]
            scene = resident[key] if resident is not None else original.to(device)
            views = tuple(zip(('A','B1'), views_for(original, key, exposure), strict=True))
            signature = signatures[key] if signatures is not None else None
            if signature is None:
                totals.append(torch.stack([native_task_mse(model, scene, (keep.to(device), bands.to(device)),
                                                          task, scale)
                                           for task, (keep, bands) in views]))
            else:
                buckets[signature].extend((original, scene, task, view) for task, view in views)
        for jobs in buckets.values():
            # An even view budget keeps each A/B1 pair together.
            for at in range(0, len(jobs), 16):
                losses = packed_loss(model, jobs[at:at+16])
                if backward:
                    (losses.sum() * scale).backward()
                totals.append(losses.detach().reshape(-1, 2).sum(0))
    return torch.stack(totals).sum(0)
