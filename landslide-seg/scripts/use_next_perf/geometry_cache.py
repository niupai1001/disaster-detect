"""Bounded CPU memo for deterministic geometry; candidate only, no production import."""
from collections import OrderedDict


class GeometryMemo:
    def __init__(self,compute,max_bytes=256*1024**2):
        self.compute=compute;self.max_bytes=max_bytes;self.entries=OrderedDict();self.bytes=0
    def __call__(self,asset,grid=None):
        key=(asset.sample_grid['crs'],tuple(asset.sample_grid['transform']),tuple(asset.image.shape[-2:]),str(grid))
        if key in self.entries:
            self.entries.move_to_end(key);return self.entries[key][0]
        value=self.compute(asset,grid);matrix=value[0]
        size=matrix.indices().numel()*matrix.indices().element_size()+matrix.values().numel()*matrix.values().element_size()
        if size<=self.max_bytes:
            while self.entries and self.bytes+size>self.max_bytes:
                _,(_,old_size)=self.entries.popitem(last=False);self.bytes-=old_size
            self.entries[key]=(value,size);self.bytes+=size
        return value
