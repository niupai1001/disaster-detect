"""Window-local processing, complete coarse/global attention and local latent feedback."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .geometry import position_features


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_q = nn.LayerNorm(128)
        self.norm_k = nn.LayerNorm(128)
        self.attn = nn.MultiheadAttention(128, 4, batch_first=True, dropout=0.)
        self.norm = nn.LayerNorm(128)
        self.ff = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))

    def forward(self, q, kv, mask=None):
        out, _ = self.attn(self.norm_q(q), self.norm_k(kv), kv, key_padding_mask=mask, need_weights=False)
        x = q+out
        return x+self.ff(self.norm(x))


class SpatialBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = nn.Linear(50, 128)
        self.local = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 512, dropout=0., batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
        self.global_seed = nn.Parameter(torch.randn(32, 128)*.02)
        self.global_read = CrossBlock()
        self.global_self = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 512, dropout=0., batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
        self.feedback = CrossBlock()
        self.spectral = CrossBlock()

    def forward(self, summary, latent, latent_valid, index, centers, geometry, chunk_rows=None):
        x = summary+self.position(position_features(centers, geometry).to(summary.dtype))
        windows, inverse = torch.unique(torch.div(index, 8, rounding_mode='floor'), dim=0, return_inverse=True)
        slots = index[:, 1].remainder(8)*8+index[:, 0].remainder(8)
        padded = x.new_zeros(len(windows), 64, 128)
        mask = torch.ones(len(windows), 64, device=x.device, dtype=torch.bool)
        padded[inverse, slots] = x
        mask[inverse, slots] = False
        rows = int(windows[:, 1].max())+1 if chunk_rows is None else chunk_rows
        if type(rows) is not int or rows<1:
            raise ValueError('Positive window chunk rows required')
        processed = torch.zeros_like(padded)
        # Pass 1: whole windows (including a window-row halo) share the immutable grid.
        for start in range(int(windows[:, 1].min()), int(windows[:, 1].max())+1, rows):
            core = (windows[:, 1]>=start)&(windows[:, 1]<start+rows)
            halo = (windows[:, 1]>=start-1)&(windows[:, 1]<start+rows+1)
            selected = halo.nonzero().flatten()
            def run(z, m):
                return self.local(z, src_key_padding_mask=m)
            result = checkpoint(run, padded[selected], mask[selected], use_reentrant=False) if torch.is_grad_enabled() else run(padded[selected], mask[selected])
            processed = processed.index_copy(0, selected[core[selected]], result[core[selected]])
        fine = processed[inverse, slots]
        coarse_index, coarse_inverse = torch.unique(torch.div(index, 4, rounding_mode='floor'), dim=0, return_inverse=True)
        counts = torch.bincount(coarse_inverse, minlength=len(coarse_index))
        coarse = fine.new_zeros(len(coarse_index), 128).index_add(0, coarse_inverse, fine)/counts[:, None]
        global_latent = self.global_read(self.global_seed[None], coarse[None])
        global_latent = self.global_self(global_latent)[0]
        # Pass 2: feedback uses the same completed global set in every chunk.
        pieces = []
        block = max(64, rows*8*geometry.shape[1])
        for begin in range(0, len(fine), block):
            pieces.append(self.feedback(fine[None, begin:begin+block], global_latent[None])[0])
        fine = torch.cat(pieces)
        lookup = torch.full((geometry.shape[0]*geometry.shape[1],), -1, device=x.device, dtype=torch.long)
        lookup[index[:, 1]*geometry.shape[1]+index[:, 0]] = torch.arange(len(index), device=x.device)
        offsets = index.new_tensor([(dx, dy) for dy in (-1,0,1) for dx in (-1,0,1)])
        neighbor = index[:, None]+offsets
        inside = (neighbor>=0).all(-1)&(neighbor[..., 0]<geometry.shape[1])&(neighbor[..., 1]<geometry.shape[0])
        flat = (neighbor[..., 1]*geometry.shape[1]+neighbor[..., 0]).clamp(0, len(lookup)-1)
        found = lookup[flat]
        allowed = inside&(found>=0)
        latent = self.spectral(latent, fine[found.clamp_min(0)], ~allowed)
        latent = torch.where(latent_valid[..., None], latent, 0)
        return fine, latent, global_latent, dict(fine=len(fine), local_latent=int(latent_valid.sum()), coarse=len(coarse), global_tokens=32,
            max_attention_pairs=max(64*64, 32*len(coarse), 32*block), full_scene_attention=False)
