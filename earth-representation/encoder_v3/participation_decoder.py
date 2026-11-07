"""Optional A readout matrix execution; same neighbours, weights and learned MLP."""
import torch
from .model import ReflectanceDecoder
from .spatial_index import SpatialIndex


def aggregate_A_matrix(representation,targets,mask_bounds,*,index_type=SpatialIndex):
    r=representation
    if r.spatial_index is None or not isinstance(r.spatial_index,index_type):r.spatial_index=index_type(r)
    token=r.spatial_index.a_tokens(r,mask_bounds)
    if not len(token):return r.tokens.new_zeros(len(targets),128),r.tokens.new_zeros(len(targets),2),torch.zeros(len(targets),dtype=torch.bool,device=targets.device)
    relative=((targets[:,:2]+targets[:,2:])[:,None]/2-r.centers[token][None])/(r.bounds[token,2:]-r.bounds[token,:2])[None]
    weights=1/(1+relative.square().sum(-1));totals=weights.sum(1);supported=totals>0;weights=weights/totals[:,None].clamp_min(torch.finfo(weights.dtype).tiny)
    # QxK weights times Kx128 avoids the QxKx128 expanded feature tensor.
    features=weights.to(r.tokens.dtype)@r.tokens[token]
    position=(relative*weights[...,None]).sum(1).to(features.dtype)
    return features,position,supported


class MatrixADecoder(ReflectanceDecoder):
    def __init__(self,*,indexed=False,cell_indexed=False,bulk_geometry=False):
        super().__init__(indexed=indexed,cell_indexed=cell_indexed)
        if type(bulk_geometry) is not bool:raise ValueError('Explicit boolean geometry execution flag required')
        self.bulk_geometry=bulk_geometry

    def forward(self,representation,targets,srf_condition,*,kind,mask_bounds=None):
        if kind!='A':return super().forward(representation,targets,srf_condition,kind=kind,mask_bounds=mask_bounds)
        if not self.indexed:raise ValueError('Matrix A execution requires indexed Decoder')
        if targets.ndim!=2 or targets.shape[1]!=4 or srf_condition.shape!=(32,) or not bool((targets[:,2:]>targets[:,:2]).all()):raise ValueError('Invalid target support or SRF condition')
        index_type=SpatialIndex
        if self.bulk_geometry:
            from .participation_geometry import BulkSpatialIndex
            index_type=BulkSpatialIndex
        feature,position,supported=aggregate_A_matrix(representation,targets,mask_bounds,index_type=index_type)
        output=self.mlp(torch.cat((feature,position,srf_condition.expand(len(targets),-1)),-1))[:,0]
        return torch.where(supported,output,torch.nan),supported
