"""Sparse exact geometry readout; bounds only prune, native cells determine areas."""
import numpy as np
import shapely
import torch
from .model import union_overlap


class SpatialIndex:
    def __init__(self, representation):
        boxes=representation.bounds.detach().cpu().numpy()
        self.tree=shapely.STRtree(shapely.box(boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]))
        self.a_candidates={}
        self.cell_tree=None

    def a_tokens(self,r,mask_bounds):
        device=r.bounds.device
        if mask_bounds is None:raise ValueError('A requires explicit masked region')
        rule=getattr(r,'a_adjacency_rule',None) or 'legacy_center_outside'
        if rule not in ('legacy_center_outside','represented_boundary_v1'):raise ValueError('Unknown A adjacency rule')
        key=(rule,*mask_bounds.detach().cpu().tolist())
        if key not in self.a_candidates:
            candidate=self.tree.query(shapely.box(*key[1:]).boundary,predicate='intersects')
            selected=[]
            for index in candidate:
                if not bool(r.valid[index]): continue
                c=r.cells[index];m=mask_bounds
                intersects=(c[:,0]<=m[2])&(c[:,2]>=m[0])&(c[:,1]<=m[3])&(c[:,3]>=m[1])
                inside=(c[:,:2]>m[:2]).all(1)&(c[:,2:]<m[2:]).all(1)
                if not bool((intersects & ~inside).any()):continue
                if rule=='legacy_center_outside':
                    center=r.centers[index]
                    if bool((center[0]>m[0])&(center[0]<m[2])&(center[1]>m[1])&(center[1]<m[3])):continue
                selected.append(int(index))
            self.a_candidates[key]=torch.tensor(selected,dtype=torch.long,device=device)
        return self.a_candidates[key]

    def edges(self, r, targets, kind, mask_bounds, *, cell_indexed=False):
        device=r.bounds.device
        if kind=='A':
            if mask_bounds is None: raise ValueError('A requires explicit masked region')
            token=self.a_tokens(r,mask_bounds)
            q=torch.arange(len(targets),device=device).repeat_interleave(len(token))
            token=token.repeat(len(targets))
            weights=None
        elif cell_indexed:
            if self.cell_tree is None:
                self.cell_bounds=torch.cat(r.cells)
                self.cell_tokens=torch.repeat_interleave(torch.arange(len(r.cells),device=device),torch.tensor([len(c) for c in r.cells],device=device))
                boxes=self.cell_bounds.detach().cpu().numpy()
                self.cell_tree=shapely.STRtree(shapely.box(boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]))
            boxes=targets.detach().cpu().numpy()
            pairs=self.cell_tree.query(shapely.box(boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]))
            q=torch.as_tensor(pairs[0].copy(),device=device,dtype=torch.long)
            cell=torch.as_tensor(pairs[1].copy(),device=device,dtype=torch.long)
            token=self.cell_tokens[cell]
            low=torch.maximum(targets[q,:2],self.cell_bounds[cell,:2])
            high=torch.minimum(targets[q,2:],self.cell_bounds[cell,2:])
            weights=(high-low).clamp_min(0).prod(-1)
            positive=(weights>0)&r.valid[token];q=q[positive];token=token[positive];weights=weights[positive]
        else:
            boxes=targets.detach().cpu().numpy()
            pairs=self.tree.query(shapely.box(boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]))
            q=torch.as_tensor(pairs[0].copy(),device=device,dtype=torch.long)
            token=torch.as_tensor(pairs[1].copy(),device=device,dtype=torch.long)
            valid=r.valid[token];q=q[valid];token=token[valid]
            weights=targets.new_zeros(len(q))
            for index in torch.unique(token).tolist():
                selected=token==index
                weights[selected]=union_overlap(r.cells[index],targets[q[selected]])
            positive=weights>0;q=q[positive];token=token[positive];weights=weights[positive]
        relative=((targets[q,:2]+targets[q,2:])/2-r.centers[token])/(r.bounds[token,2:]-r.bounds[token,:2])
        if weights is None: weights=1/(1+relative.square().sum(-1))
        return q,token,weights,relative


def aggregate(r,targets,*,kind,mask_bounds,cell_indexed=False):
    if r.spatial_index is None: r.spatial_index=SpatialIndex(r)
    q,token,weights,relative=r.spatial_index.edges(r,targets,kind,mask_bounds,cell_indexed=cell_indexed)
    totals=targets.new_zeros(len(targets)).index_add(0,q,weights)
    supported=totals>0
    weights=weights/totals[q].clamp_min(torch.finfo(totals.dtype).tiny)
    features=r.tokens.new_zeros(len(targets),128).index_add(0,q,r.tokens[token]*weights[:,None].to(r.tokens.dtype))
    position=targets.new_zeros(len(targets),2).index_add(0,q,relative*weights[:,None]).to(features.dtype)
    return features,position,supported
