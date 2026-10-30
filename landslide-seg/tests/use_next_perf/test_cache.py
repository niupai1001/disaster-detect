from pathlib import Path
import sys
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'scripts'),str(Path(__file__).resolve().parents[2]/'tests/use_next_p2')]
from use_next_perf.geometry_cache import GeometryMemo
from test_scale import asset,area_grid


def test_exact_cache_and_bounded_eviction():
    a=asset();calls=[]
    def compute(a,g=None):calls.append(1);return area_grid(a,g)
    cache=GeometryMemo(compute);first=cache(a);second=cache(a)
    assert len(calls)==1 and torch.equal(first[0].values(),second[0].values())
    tiny=GeometryMemo(compute,max_bytes=1);tiny(a);tiny(a)
    assert not tiny.entries and tiny.bytes==0


def test_disk_restart_and_corruption(tmp_path):
    from use_next_perf.disk_geometry import DiskGeometry
    a=asset();calls=[]
    def compute(a,g=None):calls.append(1);return area_grid(a,g)
    first=DiskGeometry(compute,tmp_path,'algorithm');value=first(a)
    restarted=DiskGeometry(compute,tmp_path,'algorithm');other=restarted(a)
    assert len(calls)==1 and torch.equal(value[0].values(),other[0].values())
    next(tmp_path.glob('*.pt')).write_bytes(b'corrupted')
    rebuilt=restarted(a)
    assert len(calls)==2 and restarted.invalid==1 and torch.equal(value[0].values(),rebuilt[0].values())
