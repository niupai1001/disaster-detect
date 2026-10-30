import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from landslide_seg.models.use_next_v2 import EOAsset,FeatureLevel,USEOutput
from landslide_seg.models.use_next_v2.physical import area_grid as legacy_grid
from landslide_seg.models.use_next_v2.pipeline import USEModel
from landslide_seg.models.use_next_p2.encoder import ScaleEncoder,ScaleUSE,grid_for,mask_asset,scalar_spacing
from landslide_seg.models.use_next_p2.geometry import area_grid
from landslide_seg.models.use_next_p2.tasks import masks,band_view,patch_targets


def asset(rotation=False):
    x=torch.sin(torch.arange(2*24*24).reshape(2,24,24).float()/19)*.1+.2
    return EOAsset('s2',('red','nir'),x,torch.tensor([[.665,.03,10.],[.842,.1,20.]]),torch.ones(2,3,dtype=torch.bool),
        dict(crs='EPSG:32648',transform=(0.,-10.,500000.,10.,0.,3000000.) if rotation else (10.,0.,500000.,0.,-10.,3000000.)),
        (10.,20.),(None,None),torch.ones_like(x,dtype=torch.bool),'2024-01-01','surface_reflectance',dict(band_ids=['red','nir']))


@pytest.mark.parametrize('rotation',[False,True])
def test_exact_vector_geometry(rotation):
    a=asset(rotation);old=legacy_grid(a);new=area_grid(a)
    assert old[1:]==new[1:]
    assert torch.equal(old[0].indices(),new[0].indices()) and torch.equal(old[0].values(),new[0].values())


def test_identical_initial_parameters_and_target():
    torch.manual_seed(1234);pixel=ScaleEncoder('pixel')
    torch.manual_seed(1234);physical=ScaleEncoder('physical')
    assert all(torch.equal(v,physical.state_dict()[k]) for k,v in pixel.state_dict().items())
    a=asset();grid=grid_for(a)
    first=pixel(a,grid=grid);second=physical(a,grid=grid)
    assert first.metric_local.tensor.shape==second.metric_local.tensor.shape
    assert torch.equal(first.metric_patch.valid_mask,second.metric_patch.valid_mask)
    assert not torch.allclose(first.global_embedding,second.global_embedding)
    calibration={'s2/red,nir':dict(mean=[0.,0.],std=[1.,1.])}
    target,_=patch_targets(a,grid,calibration)
    assert torch.isfinite(target[0]).all() and (target[1]>0).any()


@pytest.mark.parametrize('mode',['pixel','physical'])
def test_mask_perturbation_and_invalid_nan(mode):
    torch.set_num_threads(4);a=asset();hidden,grid=masks(a,1234)
    masked=mask_asset(a,hidden,grid);changed=copy.deepcopy(a)
    changed.image[~masked.pixel_valid]=999
    model=ScaleEncoder(mode).eval()
    assert torch.equal(model(a,hidden_cells=hidden,grid=grid).global_embedding,
                       model(changed,hidden_cells=hidden,grid=grid).global_embedding)
    a.image[0,0,0]=float('nan');a.pixel_valid[0,0,0]=False
    assert torch.isfinite(model(a,grid=grid).global_embedding).all()


def test_rotated_semantic_spacing_wrapper(monkeypatch):
    level=FeatureLevel(torch.zeros(2,2,2),torch.ones(2,2,dtype=torch.bool),(0.,-10.,0.,10.,0.,0.),0.,None,'s2',None,False)
    monkeypatch.setattr(USEModel,'forward',lambda *a,**kw:USEOutput(semantic_levels={'dofa':copy.deepcopy(level)}))
    model=ScaleUSE.__new__(ScaleUSE);torch.nn.Module.__init__(model)
    assert model(asset(True),grid_for(asset(True))).semantic_levels['dofa'].grid_spacing_m==10
    assert scalar_spacing((0.,-10.,0.,10.,0.,0.))==10


def test_unknown_effective_resolution_and_sar_no_target():
    a=asset();a.modality='s1';a.channel_names=('vv','vh');a.radiometry='gamma0_linear'
    target,reason=patch_targets(a,grid_for(a),{})
    assert target is None and reason.startswith('NOT_AVAILABLE_SAR')
