"""IR1 joint local readout, exact kNN and geometry-conditioned 309-wide decoder."""
import time
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import fourier2d
from .retrieval import ExactKDIndex


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

    def prepare(self, representation):
        # Linear K/V projections distribute over content + relative position + type.
        content=torch.cat(((representation.fine+self.types[0])[:,None],
                           representation.latent+self.types[1]),1)
        weight=self.local.in_proj_weight[128:]
        return (F.linear(content,weight,self.local.in_proj_bias[128:]),
                weight@self.relative.weight,F.linear(self.relative.bias,weight),
                F.linear(representation.global_latent,self.global_read.in_proj_weight[128:],
                         self.global_read.in_proj_bias[128:]))

    @staticmethod
    def attend(query,kv,attention,available=None,shared=False):
        q=F.linear(query,attention.in_proj_weight[:128],attention.in_proj_bias[:128])
        key,value=kv.split(128,-1)
        if shared:
            q=q.reshape(1,-1,4,32).transpose(1,2)
            key=key.reshape(1,-1,4,32).transpose(1,2)
            value=value.reshape(1,-1,4,32).transpose(1,2)
            result=F.scaled_dot_product_attention(q,key,value).transpose(1,2).reshape(-1,128)
        else:
            q=q.reshape(-1,1,4,32).transpose(1,2)
            key=key.reshape(len(q),-1,4,32).transpose(1,2)
            value=value.reshape(len(q),-1,4,32).transpose(1,2)
            result=F.scaled_dot_product_attention(q,key,value,attn_mask=available[:,None,None,:]).reshape(-1,128)
        return attention.out_proj(result)

    def forward(self, representation, points, gsd, condition, chunk_pixels=128, neighbors=None):
        if not len(points) or chunk_pixels<1:
            raise ValueError('Nonempty target geometry and positive chunk size required')
        geometry = representation.geometry
        outputs, metrics = [], []
        content,position_weight,position_bias,global_kv=getattr(representation,'_decoder_projection',None) or self.prepare(representation)
        if not hasattr(representation,"_retrieval_index"):
            representation._retrieval_index=ExactKDIndex(representation.centers)
        for begin in range(0, len(points), chunk_pixels):
            target = points[begin:begin+chunk_pixels]
            patch = torch.floor((target.double()-target.new_tensor(geometry.origin))/geometry.patch_size)
            own_center = (patch+.5)*geometry.patch_size+target.new_tensor(geometry.origin)
            relative = ((target-own_center)/geometry.patch_size).to(condition.dtype)
            log_support = condition.new_tensor([gsd[0]/geometry.patch_size, gsd[1]/geometry.patch_size,
                                                 gsd[0]*gsd[1]/geometry.patch_size**2]).log()
            geo = torch.cat((fourier2d(relative), log_support.expand(len(target), -1)), -1)
            z = condition.expand(len(target), -1) if condition.ndim==1 else condition[begin:begin+len(target)]
            q = self.query(torch.cat((z, geo), -1))[:, None]
            if neighbors is None:
                selected, metric = representation._retrieval_index.query(target)
            else:
                selected=neighbors[begin:begin+len(target)]
                metric=dict(distance_evaluations=0,candidates=len(representation.centers),queries=len(target),selected=selected.numel(),seconds=0.,cache_hit=True)
            metrics.append(metric)
            if selected.shape[1]:
                pos = ((representation.centers[selected]-target[:, None])/geometry.patch_size).to(q.dtype)
                position=F.linear(fourier2d(pos),position_weight,position_bias)
                kv=(content[selected]+position[:,:,None]).flatten(1,2)
                available = torch.cat((torch.ones_like(representation.latent_valid[selected][..., :1]),
                                       representation.latent_valid[selected]), -1).flatten(1)
                a_local=self.attend(q[:,0],kv,self.local,available)
                has_latent = representation.latent_valid[selected].any(-1).any(-1).to(q.dtype)[:, None]
                has_local = torch.ones_like(has_latent)
            else:
                a_local = torch.zeros_like(q[:,0])
                has_local = has_latent = q.new_zeros(len(target), 1)
            a_global=self.attend(q[:,0],global_kv,self.global_read,shared=True)
            joined = torch.cat((a_local, a_global, geo, z, has_local, has_latent), -1)
            outputs.append(self.output(joined).squeeze(-1))
        return torch.cat(outputs), metrics
