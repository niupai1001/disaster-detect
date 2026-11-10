"""Bounded cache of pure geometric edges; no pixel values, masks or learned features."""
from collections import OrderedDict
import torch
from .geometry import native_cells
from .native_patch import MICRO,MAX_EDGES_PER_CELL


class GeometryCache:
    def __init__(self,limit_bytes=256*1024**2):
        self.limit_bytes=limit_bytes;self.bytes=0;self.items=OrderedDict();self.builds=0;self.hits=0

    def plan(self,grid,geometry,device,begin,end):
        key=(grid,geometry.origin,geometry.patch_size,geometry.shape,str(device),begin,end)
        if key in self.items:
            self.hits+=1;self.items.move_to_end(key);return self.items[key]
        cells=native_cells(grid,device,begin,end);cell=geometry.patch_size/MICRO
        origin=cells.new_tensor(geometry.origin)
        low=torch.floor((cells[:,:2]-origin)/cell).long()
        high=torch.ceil((cells[:,2:]-origin)/cell).long()-1
        spans=high-low+1;lengths=spans.prod(-1)
        if int(lengths.max())>MAX_EDGES_PER_CELL:raise ValueError('Native cell geometry exceeds supported edge expansion')
        source=torch.repeat_interleave(torch.arange(len(cells),device=device),lengths)
        starts=torch.repeat_interleave(lengths.cumsum(0)-lengths,lengths)
        offset=torch.arange(int(lengths.sum()),device=device)-starts
        nx=spans[:,0][source];i=low[:,0][source]+offset%nx;j=low[:,1][source]+offset//nx
        lower=origin+torch.stack((i,j),-1).to(cells.dtype)*cell
        upper=lower+cell
        area=(torch.minimum(cells[source,2:],upper)-torch.maximum(cells[source,:2],lower)).clamp_min(0).prod(-1)
        px=torch.div(i,MICRO,rounding_mode='floor');py=torch.div(j,MICRO,rounding_mode='floor')
        destination=(py*geometry.shape[1]+px)*16+(j-py*MICRO)*MICRO+(i-px*MICRO)
        if bool((destination<0).any()) or bool((destination>=geometry.shape[0]*geometry.shape[1]*16).any()):
            raise ValueError('Native geometry falls outside immutable scene grid')
        result=dict(source=source,destination=destination,area=area,cell_area=cell*cell)
        size=sum(t.numel()*t.element_size() for t in result.values() if isinstance(t,torch.Tensor))
        self.builds+=1
        if size<=self.limit_bytes:
            while self.items and self.bytes+size>self.limit_bytes:
                _,old=self.items.popitem(last=False);self.bytes-=sum(t.numel()*t.element_size() for t in old.values() if isinstance(t,torch.Tensor))
            self.items[key]=result;self.bytes+=size
        return result
