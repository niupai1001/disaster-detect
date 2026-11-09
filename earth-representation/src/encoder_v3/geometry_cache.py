"""Bounded, copy-isolated native geometry reuse; never caches learned outputs."""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import threading
from . import layout,streaming


_native_layout=layout.native_layout
_fields={'indices','lengths','xy','represented','represented_lengths','receptive','receptive_lengths','centers','bounds'}


def copy_layout(value):
    if value is None:return None
    if set(value)!=_fields:raise ValueError('Unknown geometry schema; refuse caching')
    return {key:item.copy() for key,item in value.items()}


class NativeGeometryCache:
    def __init__(self,*,data_version,max_bytes=128*1024**2):
        if not data_version or type(max_bytes) is not int or max_bytes<1:raise ValueError('Frozen data version and positive cache bound required')
        self.version=hashlib.sha256(json.dumps(data_version,sort_keys=True).encode()+Path(layout.__file__).read_bytes()).hexdigest()
        self.max_bytes=max_bytes;self.entries=OrderedDict();self.bytes=0;self.hits=0;self.misses=0;self.evictions=0;self.lock=threading.RLock()

    def key(self,band,patch_m,origin_xy):
        digest=hashlib.sha256(json.dumps([self.version,list(band.reflectance.shape),band.crs,list(band.affine),float(patch_m),list(origin_xy)],sort_keys=True).encode())
        for tensor in (band.valid_mask,band.visible):
            digest.update(b'none' if tensor is None else tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def __call__(self,band,patch_m,origin_xy):
        key=self.key(band,patch_m,origin_xy)
        with self.lock:
            if key in self.entries:
                self.hits+=1;value,size=self.entries.pop(key);self.entries[key]=(value,size)
                return copy_layout(value)
            self.misses+=1
        value=_native_layout(band,patch_m,origin_xy)
        size=64 if value is None else sys.getsizeof(value)+sum(sys.getsizeof(v)+(sum(sys.getsizeof(x) for x in v) if isinstance(v,list) else 0) for v in value.values())
        if size>self.max_bytes:return value
        with self.lock:
            if key not in self.entries:
                while self.entries and self.bytes+size>self.max_bytes:
                    _,(_,removed)=self.entries.popitem(last=False);self.bytes-=removed;self.evictions+=1
                self.entries[key]=(value,size);self.bytes+=size
            return copy_layout(self.entries[key][0])

    def summary(self):
        with self.lock:return dict(hits=self.hits,misses=self.misses,evictions=self.evictions,entries=len(self.entries),estimated_retained_bytes=self.bytes,max_bytes=self.max_bytes,version=self.version)

    @contextmanager
    def installed(self):
        if layout.native_layout is not _native_layout or streaming.native_layout is not _native_layout:raise RuntimeError('Nested or foreign geometry patch refused')
        layout.native_layout=self;streaming.native_layout=self
        try:yield self
        finally:layout.native_layout=_native_layout;streaming.native_layout=_native_layout
