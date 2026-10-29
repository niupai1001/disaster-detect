"""Frozen native MiT with a visible RGB residual or spectral token alignment."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .use_v4_mit import USEv4, V3, propagate, staged


def rgb_indices(ids):
    aliases = (('red', 'B04'), ('green', 'B03'), ('blue', 'B02'))
    result = []
    for names in aliases:
        found = [i for i, name in enumerate(ids) if name in names]
        if len(found) != 1:
            raise ValueError('Real RGB source mapping required')
        result.append(found[0])
    return result


def visible_rgb(observation):
    """Undo the declared per-band z-score, then fill missing physical RGB with 0."""
    images, supports = [], []
    for i, ids in enumerate(observation['band_ids']):
        indices = rgb_indices(ids)
        provenance = observation['spatial_metadata'][i]
        mean = torch.as_tensor(provenance['normalization_mean'], device=observation['x'].device)
        std = torch.as_tensor(provenance['normalization_std'], device=mean.device)
        if mean.shape != (len(ids),) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Explicit physical-reflectance normalization required')
        valid = observation['pixel_valid'][i, indices] & observation['band_valid'][i, indices, None, None]
        raw = observation['x'][i, indices] * std[indices, None, None] + mean[indices, None, None]
        if not torch.isfinite(raw[valid]).all():
            raise ValueError('Nonfinite visible RGB')
        raw = torch.where(valid, raw, 0)
        images.append((raw - raw.new_tensor([.485, .456, .406])[:, None, None]) /
                      raw.new_tensor([.229, .224, .225])[:, None, None])
        supports.append(valid)
    return torch.stack(images), torch.stack(supports)


def support_fraction(mask, kernel, stride, padding):
    weight = mask.new_ones(1, 1, kernel, kernel, dtype=torch.float32)
    count = F.conv2d(mask[:, None].float(), weight, stride=stride, padding=padding)
    total = F.conv2d(torch.ones_like(mask[:, None], dtype=torch.float32), weight, stride=stride, padding=padding)
    return (count / total.clamp_min(1))[:, 0]


def student_alignment_support(observation):
    # Patch2 followed by 3x3/s2: a 6x6 original-image receptive field, odd edges cropped.
    valid = (observation['pixel_valid'] & observation['band_valid'][:, :, None, None]).any(1)
    h, w = valid.shape[-2:]
    values = F.pad(valid[:, None].float(), (0, w % 2, 0, h % 2))
    total = F.pad(torch.ones_like(valid[:, None], dtype=torch.float32), (0, w % 2, 0, h % 2))
    values = F.avg_pool2d(values, 2, 2) * 4
    total = F.avg_pool2d(total, 2, 2) * 4
    weight = values.new_ones(1, 1, 3, 3)
    return (F.conv2d(values, weight, stride=2, padding=1) /
            F.conv2d(total, weight, stride=2, padding=1).clamp_min(1))[:, 0] >= .5


class DualRouteEncoder(USEv4):
    def __init__(self, route, source=V3):
        if route not in ('A', 'B', 'C0'):
            raise ValueError('Unknown route')
        super().__init__(source)
        self.route = route
        if route == 'A':
            self.residual_zero = nn.Conv2d(32, 32, 1)
            nn.init.zeros_(self.residual_zero.weight)
            nn.init.zeros_(self.residual_zero.bias)
        if route == 'C0':
            del self.adapter
            self.requires_grad_(False)

    def forward(self, observation, rgb_visible=None, rgb_valid=None, return_pyramid=True, diagnostics=False):
        fine, query, support, fine_valid, known, queries = self.frontend(observation)
        patch = self.mit.patch_embeddings[0]
        spectral = None
        if self.route != 'C0':
            support = propagate(support, self.adapter)
            spectral = self.adapter(query).masked_fill(~support[:, None], 0)
            if self.route == 'A':
                spectral = self.residual_zero(spectral).masked_fill(~support[:, None], 0)
        if self.route in ('A', 'C0'):
            if 'band_ids' not in observation:
                raise ValueError('Real RGB source mapping required')
            for ids in observation['band_ids']:
                rgb_indices(ids)
            if rgb_visible is None and rgb_valid is None:
                rgb_visible, rgb_valid = visible_rgb(observation)
            if rgb_visible is None or rgb_valid is None or rgb_valid.dtype != torch.bool or rgb_visible.shape != rgb_valid.shape or rgb_visible.shape != (fine.shape[0], 3, *fine.shape[-2:]):
                raise ValueError('Aligned RGB and per-band support required')
            # Refill invalid values before the original convolution, including explicit inputs.
            zero = rgb_visible.new_tensor([-.485/.229, -.456/.224, -.406/.225])[None, :, None, None]
            rgb_visible = torch.where(rgb_valid, rgb_visible, zero)
            visual_support = propagate(rgb_valid.any(1), patch.proj)
            visual = patch.proj(rgb_visible)
            injected = visual if spectral is None else visual + spectral
            support = visual_support if spectral is None else visual_support | support
        else:
            injected = spectral
        captured = []
        handle = patch.layer_norm.register_forward_hook(lambda module, args, out: captured.append(out))
        try:
            features, supports = staged(self.mit, injected, support, injected=True)
        finally:
            handle.remove()
        result = dict(fine=fine, context=features, valid=dict(fine=fine_valid, context=supports),
                      stride=dict(fine=1, context=[4, 8, 16, 32]),
                      spatial_metadata=observation.get('spatial_metadata'), metadata_used=known)
        result['patch_tokens'] = captured[0]
        if diagnostics:
            result.update(queries=queries, mit_input=injected, residual=spectral,
                          )
        return result


class PatchTeacher(nn.Module):
    """Training task only: the ADE patch projection and LN, without Stage1-4."""
    def __init__(self, encoder):
        super().__init__()
        self.patch = copy.deepcopy(encoder.mit.patch_embeddings[0]).requires_grad_(False).eval()
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, original):
        rgb, valid = visible_rgb(original)
        tokens, _, _ = self.patch(rgb)
        return tokens, support_fraction(valid.all(1), 7, 4, 3) >= .8


def alignment_loss(tokens, target, teacher_support, observation):
    support = teacher_support & student_alignment_support(observation)
    if tokens.shape != target.shape or tokens.shape[:2] != (support.shape[0], support.shape[1] * support.shape[2]):
        raise ValueError('Teacher/student grids differ')
    error = 1 - F.cosine_similarity(tokens, target.detach(), dim=-1)
    mask = support.flatten(1)
    counts = mask.sum(1)
    losses = torch.where(mask, error, 0).sum(1) / counts.clamp_min(1)
    return losses.mean(), dict(tokens=counts.detach().cpu().tolist(),
                               fraction=support.float().mean((1, 2)).detach().cpu().tolist(),
                               no_alignment=(counts == 0).detach().cpu().tolist())
