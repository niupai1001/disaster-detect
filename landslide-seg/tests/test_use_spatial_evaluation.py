from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_use_spatial_reconstruction import error_totals


def test_constant_bias_has_pixel_error_but_no_neighbor_difference_error():
    target=torch.tensor([[[[0.,1.],[2.,3.]]]])
    result=error_totals(target+1,target,torch.ones_like(target,dtype=torch.bool))
    assert result['absolute'].item()==4 and result['squared'].item()==4
    assert result['pixels'].item()==4 and result['edges'].item()==4
    assert result['edge_absolute'].item()==0


def test_invalid_target_pixels_never_enter_pixel_or_edge_totals():
    target=torch.tensor([[[[0.,float('nan')],[2.,float('nan')]]]])
    valid=torch.isfinite(target)
    result=error_totals(torch.zeros_like(target),target,valid)
    assert result['pixels'].item()==2 and result['edges'].item()==1
    assert result['absolute'].item()==2 and result['edge_absolute'].item()==2
    assert all(torch.isfinite(value).all() for value in result.values())
