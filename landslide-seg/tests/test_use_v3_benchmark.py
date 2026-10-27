import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from benchmark_use_encoder import output_bytes


def test_output_bytes():
    assert output_bytes(dict(a=torch.zeros(2,3),b=[torch.zeros(4,dtype=torch.bool)]))==28


def test_attention_mac_counter():
    from benchmark_use_encoder import theoretical_macs
    from landslide_seg.models.use_v3 import SpatialBlock
    class Wrapper(torch.nn.Module):
        def __init__(self): super().__init__(); self.block=SpatialBlock(32)
        def forward(self,o): return self.block(o['x'],o['valid'])
    m=Wrapper()
    result=theoretical_macs(m,dict(x=torch.randn(1,32,4,4),valid=torch.ones(1,4,4,dtype=torch.bool)))
    # QKV+output+two FFN projections, plus both attention matmuls.
    assert result['macs']==16*32*32*8+2*16*16*32
