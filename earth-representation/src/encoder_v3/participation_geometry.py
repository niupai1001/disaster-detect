"""Candidate bulk metadata filtering; learned features never move to the CPU."""
import numpy as np
import shapely
import torch
from .spatial_index import SpatialIndex


class BulkSpatialIndex(SpatialIndex):
    def a_tokens(self,r,mask_bounds):
        if mask_bounds is None:raise ValueError('A requires explicit masked region')
        rule=getattr(r,'a_adjacency_rule',None) or 'legacy_center_outside'
        if rule not in ('legacy_center_outside','represented_boundary_v1'):raise ValueError('Unknown A adjacency rule')
        key=(rule,*mask_bounds.detach().cpu().tolist())
        if key not in self.a_candidates:
            candidates=self.tree.query(shapely.box(*key[1:]).boundary,predicate='intersects')
            valid=r.valid.detach().cpu().numpy();candidates=candidates[valid[candidates]]
            selected=np.empty(0,dtype=np.int64)
            if len(candidates):
                indices=candidates.tolist();lengths=[len(r.cells[i]) for i in indices];cells=torch.cat([r.cells[i] for i in indices]).detach().cpu().numpy();m=np.asarray(key[1:])
                intersects=(cells[:,0]<=m[2])&(cells[:,2]>=m[0])&(cells[:,1]<=m[3])&(cells[:,3]>=m[1]);inside=(cells[:,:2]>m[:2]).all(1)&(cells[:,2:]<m[2:]).all(1)
                touch=np.zeros(len(candidates),dtype=bool);np.logical_or.at(touch,np.repeat(np.arange(len(candidates)),lengths),intersects&~inside)
                if rule=='legacy_center_outside':
                    centers=r.centers.detach().cpu().numpy()[candidates];touch &= ~((centers[:,0]>m[0])&(centers[:,0]<m[2])&(centers[:,1]>m[1])&(centers[:,1]<m[3]))
                selected=candidates[touch]
            self.a_candidates[key]=torch.as_tensor(selected.copy(),dtype=torch.long,device=r.bounds.device)
        return self.a_candidates[key]
