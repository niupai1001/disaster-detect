"""Exact compiled 2D retrieval with canonical distance/index tie ordering."""
import time
import numpy as np
import torch
from scipy.spatial import cKDTree


class ExactKDIndex:
    def __init__(self, centers, leaf_size=32):
        self.centers=centers.detach().cpu().double().numpy().copy()
        if self.centers.ndim!=2 or self.centers.shape[1]!=2 or not np.isfinite(self.centers).all():
            raise ValueError('Finite 2D centers required')
        self.tree=cKDTree(self.centers,leafsize=leaf_size)

    def query(self,points,k=8):
        start=time.perf_counter();x=points.detach().cpu().double().numpy()
        if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all() or k<1:
            raise ValueError('Finite 2D targets and positive k required')
        count=len(self.centers);k=min(k,count);n=len(x)
        scans=0
        if k and n:
            # Oversampling handles ordinary grid ties in one vectorized sort.
            width=min(count,max(32,k+1))
            _,ids=self.tree.query(x,k=list(range(1,width+1)),eps=0,workers=1)
            distance=((x[:,None]-self.centers[ids])**2).sum(-1);scans=distance.size
            order=np.lexsort((ids,distance),axis=1)
            ids=np.take_along_axis(ids,order,axis=1)
            distance=np.take_along_axis(distance,order,axis=1)
            indices=ids[:,:k].copy()
            if width<count:
                ambiguous=np.flatnonzero(np.isclose(distance[:,-1],distance[:,k-1],rtol=1e-13,atol=0))
                # Include every boundary tie, including arbitrarily many duplicate centers.
                radius=np.nextafter(np.sqrt(distance[ambiguous,k-1])*(1+1e-13),np.inf)
                candidates=self.tree.query_ball_point(x[ambiguous],radius,eps=0)
                for row,neighbors in zip(ambiguous,candidates):
                    neighbors=np.asarray(neighbors,dtype=np.int64)
                    d=((x[row]-self.centers[neighbors])**2).sum(-1);scans+=len(d)
                    indices[row]=neighbors[np.lexsort((neighbors,d))[:k]]
        else:indices=np.empty((n,k),dtype=np.int64)
        result=torch.as_tensor(indices,device=points.device)
        if points.is_cuda:torch.cuda.synchronize(points.device)
        return result,dict(distance_evaluations=int(scans),candidates=count,queries=n,selected=n*k,
            seconds=time.perf_counter()-start,bbox_tests=0,index='scipy_exact_ckdtree',
            distance_evaluation_scope='explicit_canonical_refinement_only; library_internal_traversal_unavailable')


class ViewRetrievalCache:
    """Integer neighbor indices for scored native pixels, scoped to one representation."""
    def __init__(self,representation,targets):
        self.representation=representation;self.maps={};self.metrics=[]
        self.index=ExactKDIndex(representation.centers)
        grouped={}
        for channel,mask,_ in targets:grouped.setdefault(channel.grid,[]).append(mask)
        for grid,masks in grouped.items():
            if len(masks)<2:continue
            start=time.perf_counter();union=masks[0].clone()
            for mask in masks[1:]:union.logical_or_(mask)
            rc=union.nonzero();host=rc.cpu();parts=[];scans=boxes=0
            a,_,x0,_,e,y0=grid.affine
            for begin in range(0,len(host),16384):
                rows=host[begin:begin+16384]
                points=torch.stack((x0+(rows[:,1].double()+.5)*a,y0+(rows[:,0].double()+.5)*e),-1)
                indices,m=self.index.query(points);parts.append(indices);scans+=m['distance_evaluations'];boxes+=m['bbox_tests']
            indices=torch.cat(parts).to(rc.device)
            lookup=torch.full((grid.shape[0]*grid.shape[1],),-1,device=rc.device,dtype=torch.long)
            lookup[rc[:,0]*grid.shape[1]+rc[:,1]]=torch.arange(len(rc),device=rc.device)
            self.maps[grid]=(lookup,indices)
            if rc.is_cuda:torch.cuda.synchronize(rc.device)
            self.metrics.append(dict(distance_evaluations=scans,candidates=len(representation.centers),queries=len(rc),seconds=time.perf_counter()-start,bbox_tests=boxes,cache_build=True))

    def get(self,grid,linear):
        if grid not in self.maps:return None
        lookup,indices=self.maps[grid]
        return indices[lookup[linear]]
