"""Permutation invariant, variable-band optical encoder with a full-grid path."""
import torch
from torch import nn
from torch.nn import functional as F


class UniversalSpectralEncoder(nn.Module):
    def __init__(self, width=32, heads=4, attention=True, queries=4,
                 metadata=True, hidden=32, relation_grid=2, identity='physical'):
        super().__init__()
        if width % heads or queries < 0 or relation_grid < 1:
            raise ValueError('Invalid USE dimensions')
        self.width, self.grid = width, relation_grid
        if identity not in ('physical', 'random'):
            raise ValueError('Unknown metadata identity')
        self.identity = identity
        self.tokenizer = nn.Sequential(nn.Conv2d(1, hidden, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(hidden, width, 1))
        self.norm = nn.LayerNorm(width)
        self.metadata = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width)) if metadata else None
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True) if attention else None
        self.queries = nn.Parameter(torch.randn(queries, width)*.02) if queries else None
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True) if queries else None
        self.project = nn.Linear(queries*width, width) if queries else nn.Identity()
        self.fuse = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, observation):
        x, bv, pv = (observation[k] for k in ('x', 'band_valid', 'pixel_valid'))
        if x.ndim != 4 or bv.shape != x.shape[:2] or pv.shape != x.shape:
            raise ValueError('Invalid batched optical shapes')
        valid = pv & bv[:, :, None, None]
        if not valid.flatten(1).any(1).all():
            raise ValueError('Every sample requires a valid observation')
        if not torch.isfinite(x[valid]).all():
            raise ValueError('Nonfinite valid observation')
        b, c, h, w = x.shape
        clean = torch.where(valid, x, 0)
        z = self.tokenizer(clean.reshape(b*c, 1, h, w)).reshape(b, c, self.width, h, w)
        z = self.norm(z.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        if self.metadata is not None:
            m, known = observation['band_metadata'], observation['known_mask']
            if self.identity == 'random':
                import hashlib
                m = m.clone()
                known = known.clone()
                for i, names in enumerate(observation['band_ids']):
                    for j, name in enumerate(names):
                        digest = hashlib.sha256(('USE1234'+name).encode()).digest()
                        m[i, j] = m.new_tensor([int.from_bytes(digest[k:k+4], 'big')/2**32 for k in (0,4,8)])
                        known[i, j] = True
            # Log scaling prevents GSD in metres dominating wavelengths in micrometres.
            desc = torch.cat((torch.where(known, torch.log1p(m), 0), known.to(m.dtype)), -1)
            z = z + self.metadata(desc)[:, :, :, None, None]
        z = torch.where(valid[:, :, None], z, 0)
        local = z.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        g = self.grid
        # Ceil pooling retains odd-sized edge pixels; divide pooled sums by valid counts.
        count = F.avg_pool2d(valid.float().reshape(b*c, 1, h, w), g, g, ceil_mode=True)
        pooled = F.avg_pool2d(z.reshape(b*c, self.width, h, w), g, g, ceil_mode=True)
        pooled = pooled / count.clamp_min(1e-8)
        ph, pw = pooled.shape[-2:]
        tokens = pooled.reshape(b, c, self.width, ph, pw).permute(0, 3, 4, 1, 2).reshape(-1, c, self.width)
        present = count.reshape(b, c, ph, pw).permute(0, 2, 3, 1).reshape(-1, c) > 0
        active = present.any(-1)
        # All-masked locations use a dummy token, then are zeroed after attention.
        safe = present.clone()
        safe[~active, 0] = True
        if self.attention is not None:
            tokens = tokens + self.attention(tokens, tokens, tokens, key_padding_mask=~safe, need_weights=False)[0]
        if self.pool is not None:
            q = self.queries[None].expand(tokens.shape[0], -1, -1)
            context = self.project(self.pool(q, tokens, tokens, key_padding_mask=~safe, need_weights=False)[0].flatten(1))
        else:
            context = (tokens * present[:, :, None]).sum(1) / present.sum(1).clamp_min(1)[:, None]
        context = torch.where(active[:, None], context, 0)
        context = context.reshape(b, ph, pw, self.width).permute(0, 3, 1, 2)
        context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=False)
        output_valid = valid.any(1)
        feature = local + self.fuse(local + context)
        return {'features': torch.where(output_valid[:, None], feature, 0), 'valid': output_valid}
