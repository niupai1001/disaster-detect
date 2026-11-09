"""IR1 joint local readout, exact kNN and geometry-conditioned 309-wide decoder."""
import time
import torch
from torch import nn
from .geometry import fourier2d


def exact_knn(points, centers, k=8):
    if points.is_cuda:
        torch.cuda.synchronize(points.device)
    start = time.perf_counter()
    if points.ndim!=2 or centers.ndim!=2 or points.shape[1]!=2 or centers.shape[1]!=2 or k<1:
        raise ValueError('2D target/candidate coordinates and positive k required')
    distance = ((points[:, None].double()-centers[None].double())**2).sum(-1)
    # Stable order resolves ties by canonical patch order, preserving full/chunk equivalence.
    index = torch.argsort(distance, dim=-1, stable=True)[:, :min(k, len(centers))]
    if points.is_cuda:
        torch.cuda.synchronize(points.device)
    return index, dict(distance_evaluations=len(points)*len(centers), candidates=len(centers),
                       queries=len(points), selected=index.numel(), seconds=time.perf_counter()-start)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(51, 128), nn.GELU(), nn.Linear(128, 128))
        self.relative = nn.Linear(16, 128)
        self.types = nn.Parameter(torch.randn(2, 128)*.02)
        self.local = nn.MultiheadAttention(128, 4, batch_first=True, dropout=0.)
        self.global_read = nn.MultiheadAttention(128, 4, batch_first=True, dropout=0.)
        self.output = nn.Sequential(nn.Linear(309, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, representation, points, gsd, condition, chunk_pixels=128):
        if not len(points) or chunk_pixels<1:
            raise ValueError('Nonempty target geometry and positive chunk size required')
        geometry = representation.geometry
        outputs, metrics = [], []
        for begin in range(0, len(points), chunk_pixels):
            target = points[begin:begin+chunk_pixels]
            patch = torch.floor((target.double()-target.new_tensor(geometry.origin))/geometry.patch_size)
            own_center = (patch+.5)*geometry.patch_size+target.new_tensor(geometry.origin)
            relative = ((target-own_center)/geometry.patch_size).to(condition.dtype)
            log_support = condition.new_tensor([gsd[0]/geometry.patch_size, gsd[1]/geometry.patch_size,
                                                 gsd[0]*gsd[1]/geometry.patch_size**2]).log()
            geo = torch.cat((fourier2d(relative), log_support.expand(len(target), -1)), -1)
            z = condition.expand(len(target), -1)
            q = self.query(torch.cat((z, geo), -1))[:, None]
            selected, metric = exact_knn(target, representation.centers)
            metrics.append(metric)
            if selected.shape[1]:
                pos = ((representation.centers[selected]-target[:, None])/geometry.patch_size).to(q.dtype)
                position = self.relative(fourier2d(pos))
                summary = representation.fine[selected]+position+self.types[0]
                latent = representation.latent[selected]+position[:, :, None]+self.types[1]
                kv = torch.cat((summary[:, :, None], latent), 2).flatten(1, 2)
                available = torch.cat((torch.ones_like(representation.latent_valid[selected][..., :1]),
                                       representation.latent_valid[selected]), -1).flatten(1)
                a_local, _ = self.local(q, kv, kv, key_padding_mask=~available, need_weights=False)
                has_latent = representation.latent_valid[selected].any(-1).any(-1).to(q.dtype)[:, None]
                has_local = torch.ones_like(has_latent)
            else:
                a_local = torch.zeros_like(q)
                has_local = has_latent = q.new_zeros(len(target), 1)
            global_kv = representation.global_latent[None].expand(len(target), -1, -1)
            a_global, _ = self.global_read(q, global_kv, global_kv, need_weights=False)
            joined = torch.cat((a_local[:, 0], a_global[:, 0], geo, z, has_local, has_latent), -1)
            outputs.append(self.output(joined).squeeze(-1))
        return torch.cat(outputs), metrics
