from pathlib import Path
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'tests/use_next_p2')]
from test_scale import asset
from use_next_recovery.geometry import area_grid
from landslide_seg.models.use_next_p2.geometry import area_grid as original


@pytest.mark.parametrize('rotation',[False,True])
def test_exact_geometry(rotation):
    a=asset(rotation);first=original(a);second=area_grid(a)
    assert first[1:]==second[1:]
    assert torch.equal(first[0].indices(),second[0].indices())
    assert torch.equal(first[0].values(),second[0].values())
